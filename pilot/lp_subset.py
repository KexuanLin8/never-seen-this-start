"""LIBERO-plus stratified-subset evaluation for Evo-1 style websocket servers.

Samples a fixed fraction of tasks from every (suite, perturbation category) cell
with a deterministic seed, so every method is evaluated on exactly the same
subset. The per-episode rollout logic is copied from the official
evo_libero_plus_clients.runtime so subset numbers stay comparable with the
official full evaluation. Results go to JSONL (one line per episode), which
makes runs resumable and shardable across several client processes.

Run from the libero-plus-eval/ directory inside the libero_plus env:

  python /path/to/lp_subset.py run --tag evo1_official --shard 0 --num-shards 4
  python /path/to/lp_subset.py summarize --tag evo1_official
"""

import os
import argparse
import asyncio
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from evo_libero_plus_clients.config import SUITE_MAX_STEPS  # noqa: E402

CATEGORIES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "layout": "Objects Layout",
}
SUITES = tuple(SUITE_MAX_STEPS)
LIBERO_DUMMY_ACTION = [0.0] * 7


def load_classification():
    from libero.libero import get_libero_path

    path = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_subset(frac, seed, suites, categories):
    """Deterministic stratified sample: list of (suite, category, task_id)."""
    table = load_classification()
    subset = []
    for suite in suites:
        for cat in categories:
            full_name = CATEGORIES[cat]
            ids = sorted(t["id"] - 1 for t in table[suite] if t["category"] == full_name)
            rng = random.Random(f"{seed}-{suite}-{cat}")
            k = max(1, round(frac * len(ids)))
            subset.extend((suite, cat, tid) for tid in sorted(rng.sample(ids, k)))
    return subset


def results_path(out_root, tag, shard):
    return Path(out_root) / tag / f"results_shard{shard}.jsonl"


def load_done(tag_dir):
    """Episodes finished by ANY shard of this tag, so re-sharding never repeats work."""
    done = set()
    for path in Path(tag_dir).glob("results_shard*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task_id"]))
    return done


async def run_episode(ws, helper, env, task_description, max_steps, horizon, initial_state, save_frames):
    """Mirrors EvaluationClient.run_suite's inner episode loop (official protocol)."""
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    frames, executed_steps = [], 0
    for _ in range(max_steps // horizon):
        executed_steps += horizon
        await ws.send(json.dumps(helper.obs_to_json_dict(obs, str(task_description))))
        actions = np.array(json.loads(await ws.recv()))
        for i in range(horizon):
            action = actions[i].tolist()
            action[6] = -1 if action[6] > 0.5 else 1
            obs, _, done, _ = env.step(action[:7])
            if save_frames:
                frames.append(np.hstack([np.rot90(obs["agentview_image"], 2),
                                         np.rot90(obs["robot0_eye_in_hand_image"], 2)]))
            if done:
                return True, executed_steps, frames
    return False, executed_steps, frames


async def run_shard(args):
    import imageio
    import websockets
    from libero.libero import benchmark
    from evo_libero_plus_clients.runtime import EvaluationClient

    np.random.seed(args.seed)
    random.seed(args.seed)

    subset = build_subset(args.frac, args.subset_seed, args.suites, args.categories)
    mine = subset[args.shard::args.num_shards]
    out_file = results_path(args.out_root, args.tag, args.shard)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_file.parent)
    todo = [x for x in mine if (x[0], x[2]) not in done]
    print(f"[shard {args.shard}/{args.num_shards}] subset={len(subset)} mine={len(mine)} "
          f"done={len(mine) - len(todo)} todo={len(todo)}", flush=True)

    # Reuse the official preprocessing / env construction without its logging side effects.
    helper = EvaluationClient.__new__(EvaluationClient)
    helper.config = SimpleNamespace(seed=args.seed)

    suites = {s: benchmark.get_benchmark_dict()[s]() for s in {x[0] for x in todo}}
    async with websockets.connect(args.server_url, max_size=100_000_000, ping_interval=None) as ws:
        for n, (suite, cat, task_id) in enumerate(todo):
            task_suite = suites[suite]
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            env, task_description = helper.get_libero_env(task)
            t0 = time.time()
            try:
                success, steps, frames = await run_episode(
                    ws, helper, env, task_description, SUITE_MAX_STEPS[suite],
                    args.horizon, init_states[0], args.save_video)
            except ValueError as exc:  # invalid action raised by the env, counted as failure
                print(f"env error on {suite}/{task_id}: {exc}", flush=True)
                success, steps, frames = False, -1, []
            finally:
                env.close()
            rec = {"suite": suite, "category": cat, "task_id": task_id, "success": bool(success),
                   "steps": steps, "seconds": round(time.time() - t0, 1), "instruction": str(task_description)}
            with out_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if args.save_video and frames:
                vdir = out_file.parent / "videos" / suite / cat
                vdir.mkdir(parents=True, exist_ok=True)
                tag = "S" if success else "F"
                imageio.mimsave(vdir / f"task{task_id}_{tag}.mp4", frames, fps=30)
            print(f"[shard {args.shard}] {n + 1}/{len(todo)} {suite}/{cat}/{task_id} "
                  f"{'SUCCESS' if success else 'FAIL'} {rec['seconds']}s", flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(args):
    tag_dir = Path(args.out_root) / args.tag
    recs = {}
    for f in sorted(tag_dir.glob("results_shard*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(r["suite"], r["task_id"])] = r  # dedupe on resume
    expected = len(build_subset(args.frac, args.subset_seed, args.suites, args.categories))
    cell = defaultdict(lambda: [0, 0])
    for r in recs.values():
        for key in [(r["suite"], r["category"]), ("ALL", r["category"]), (r["suite"], "ALL"), ("ALL", "ALL")]:
            cell[key][0] += r["success"]
            cell[key][1] += 1

    cats = list(args.categories) + ["ALL"]
    print(f"tag={args.tag}  episodes={len(recs)}/{expected}  (success %, n)")
    print(f"{'suite':<16}" + "".join(f"{c:>13}" for c in cats))
    for s in list(args.suites) + ["ALL"]:
        row = f"{s:<16}"
        for c in cats:
            k, n = cell[(s, c)]
            row += f"{(100 * k / n if n else float('nan')):>8.1f}({n:>3})"
        print(row)
    k, n = cell[("ALL", "ALL")]
    lo, hi = wilson(k, n)
    print(f"overall {100 * k / max(n, 1):.2f}%  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]")
    # Category-average (official LIBERO-plus 'avg.' style: mean over categories)
    cat_means = [100 * cell[("ALL", c)][0] / cell[("ALL", c)][1] for c in args.categories if cell[("ALL", c)][1]]
    if cat_means:
        print(f"mean over categories: {sum(cat_means) / len(cat_means):.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summarize", "list"])
    p.add_argument("--tag", required=True, help="method/run name, e.g. evo1_official")
    p.add_argument("--out-root", default=os.path.join(os.environ.get("W", "work"), "outputs/lp_subset"))
    p.add_argument("--server-url", default="ws://127.0.0.1:9000")
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--subset-seed", type=int, default=2026)
    p.add_argument("--seed", type=int, default=0, help="env seed, same as official default")
    p.add_argument("--horizon", type=int, default=15, help="official LIBERO-plus default")
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args()

    if args.cmd == "run":
        asyncio.run(run_shard(args))
    elif args.cmd == "summarize":
        summarize(args)
    else:
        subset = build_subset(args.frac, args.subset_seed, args.suites, args.categories)
        counts = defaultdict(int)
        for s, c, _ in subset:
            counts[(s, c)] += 1
        for (s, c), n in sorted(counts.items()):
            print(s, c, n)
        print("total", len(subset))


if __name__ == "__main__":
    main()
