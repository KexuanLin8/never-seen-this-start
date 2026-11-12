"""P1 diagnostic rollout client (works in both the `libero` py3.8 env and the `libero_plus` py3.10 env).

Every episode follows the official LIBERO-plus rollout protocol (horizon 15,
10 dummy steps, per-suite max env steps) on BOTH benches, so clean-LIBERO and
LIBERO-plus episodes are directly comparable. Interventions:

  --prompt-mode   orig | counterfactual (instruction of the next task in the suite) | empty
  --state-mode    orig | zero (normalized mid-range, server side) | freeze (first state forever)
                  | offset (adds --state-offset dx dy dz metres to the eef position)
  --mask-views    0 (agentview) and/or 1 (wrist), blanked + masked server side
  --ctx-mask      hide padding and/or masked-view tokens from the action-head cross-attention
  --trim-padding  do not pad the VLM input to 1024 tokens
  --branches      counterfactual forward passes logged at every step (no effect on execution)
  --attn-first    log action-head attention mass at the first policy step

Flow noise is seeded per (suite, task, init, step), so the same episode gets the same
noise under every condition (paired comparisons). Results: one JSONL line per episode,
resumable and shardable.

Examples (run inside the bench's env; for libero_plus run from libero-plus-eval/):
  python p1_client.py --bench libero --condition C0_full --branches no_lang no_state no_agent no_wrist --attn-first
  python p1_client.py --bench libero --condition C1_lang_cf --prompt-mode counterfactual
  python p1_client.py --bench libero_plus --condition P0_full --frac 0.05 --branches no_lang no_state no_agent no_wrist
"""

import os
import argparse
import asyncio
import json
import math
import random
import time
import zlib
from pathlib import Path

import numpy as np

SUITE_MAX_STEPS = {"libero_spatial": 660, "libero_object": 840, "libero_goal": 900, "libero_10": 1560}
CATEGORIES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "layout": "Objects Layout",
}
DUMMY_ACTION = [0.0] * 7


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def raw_state(obs):
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"])).astype(np.float64)


def make_request(obs, prompt, state):
    """Same payload as the official clients (images flipped, dummy third view)."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy = np.zeros((448, 448, 3), dtype=np.uint8)
    return {
        "image": [img.astype(np.uint8).tolist(), wrist.astype(np.uint8).tolist(), dummy.tolist()],
        "state": state.tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }


def episode_seed(suite, task_id, init_idx):
    return zlib.crc32(f"{suite}/{task_id}/{init_idx}".encode()) % 1_000_000


def build_episodes(args, benchmark, get_libero_path):
    """List of (suite, category, task_id, init_idx)."""
    eps = []
    if args.bench == "libero":
        for suite in args.suites:
            n = benchmark.get_benchmark_dict()[suite]().n_tasks
            for tid in range(n):
                for k in range(args.inits_per_task):
                    eps.append((suite, "clean", tid, k))
        return eps
    # libero_plus: the P0 baseline subset (20%, seed 2026), then a nested sub-sample of it
    path = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    for suite in args.suites:
        for cat in args.categories:
            ids = sorted(t["id"] - 1 for t in table[suite] if t["category"] == CATEGORIES[cat])
            base = sorted(random.Random(f"2026-{suite}-{cat}").sample(ids, max(1, round(0.2 * len(ids)))))
            random.Random(f"p1-{suite}-{cat}").shuffle(base)
            k = max(1, round(min(1.0, args.frac / 0.2) * len(base)))
            eps.extend((suite, cat, tid, 0) for tid in sorted(base[:k]))
    return eps


async def run_episode(ws, env, args, suite, task_id, init_idx, instruction, init_state):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    base_seed = episode_seed(suite, task_id, init_idx)
    frozen = None
    log = {"init_eef_pos": obs["robot0_eef_pos"].tolist(), "init_eef_quat": obs["robot0_eef_quat"].tolist(),
           "init_joint_pos": np.asarray(obs["robot0_joint_pos"]).tolist() if "robot0_joint_pos" in obs else None,
           "dists": [], "attn": None}
    executed = 0
    for step in range(SUITE_MAX_STEPS[suite] // args.horizon):
        state = raw_state(obs)
        if args.state_mode == "freeze":
            frozen = state if frozen is None else frozen
            state = frozen
        elif args.state_mode == "offset":
            state = state.copy()
            state[:3] += np.array(args.state_offset)
        req = make_request(obs, instruction, state)
        exec_spec = {}
        if args.mask_views:
            exec_spec["mask_views"] = list(args.mask_views)
        if args.state_mode == "zero":
            exec_spec["state"] = "zero"
        if args.ctx_mask:
            exec_spec["ctx_mask"] = list(args.ctx_mask)
        if args.trim_padding:
            exec_spec["trim_padding"] = True
        if args.pad_to:
            exec_spec["pad_to"] = args.pad_to
        req["p1"] = {"seed": base_seed * 1000 + step, "exec": exec_spec, "branches": args.branches,
                     "return_attn": bool(args.attn_first and step == 0), "dist_horizon": args.horizon}
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
        actions = np.array(resp["actions"])
        if "branch_dist" in resp:
            log["dists"].append(resp["branch_dist"])
        if resp.get("attn") is not None:
            log["attn"] = resp["attn"]
        for i in range(args.horizon):
            action = actions[i].tolist()
            action[6] = -1 if action[6] > 0.5 else 1
            try:
                obs, _, done, _ = env.step(action[:7])
            except ValueError:  # env horizon (1000 steps) reached before SUITE_MAX_STEPS: a failure
                log["env_horizon_hit"] = True
                return False, executed, log
            executed += 1
            if done:
                return True, executed, log
    return False, executed, log


async def main_async(args):
    import websockets
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(0)
    random.seed(0)
    out_dir = Path(args.out_root) / args.bench / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    out_file = out_dir / f"results_shard{args.shard}.jsonl"

    done = set()
    for f in out_dir.glob("results_shard*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task_id"], r["init_idx"]))

    episodes = build_episodes(args, benchmark, get_libero_path)
    mine = episodes[args.shard::args.num_shards]
    todo = [e for e in mine if (e[0], e[2], e[3]) not in done]
    print(f"[{args.condition} shard {args.shard}/{args.num_shards}] total={len(episodes)} "
          f"mine={len(mine)} todo={len(todo)}", flush=True)

    suites = {s: benchmark.get_benchmark_dict()[s]() for s in {e[0] for e in todo}}
    async with websockets.connect(args.server_url, max_size=100_000_000, ping_interval=None) as ws:
        for n, (suite, cat, task_id, init_idx) in enumerate(todo):
            ts = suites[suite]
            task = ts.get_task(task_id)
            orig_instruction = task.language
            if args.prompt_mode == "counterfactual":
                instruction = ts.get_task((task_id + 1) % ts.n_tasks).language
            elif args.prompt_mode == "empty":
                instruction = ""
            else:
                instruction = orig_instruction
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=448, camera_widths=448)
            env.seed(0)
            t0 = time.time()
            try:
                init_state = ts.get_task_init_states(task_id)[init_idx]
                success, steps, log = await run_episode(ws, env, args, suite, task_id, init_idx,
                                                        str(instruction), init_state)
            except ValueError as exc:
                print(f"env error {suite}/{task_id}: {exc}", flush=True)
                success, steps, log = False, -1, {}
            finally:
                env.close()
            rec = {"bench": args.bench, "condition": args.condition, "suite": suite, "category": cat,
                   "task_id": task_id, "init_idx": init_idx, "success": bool(success), "steps": steps,
                   "seconds": round(time.time() - t0, 1), "instruction": str(instruction),
                   "orig_instruction": str(orig_instruction)}
            rec.update(log)
            with out_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{args.condition} s{args.shard}] {n + 1}/{len(todo)} {suite}/{cat}/{task_id}/{init_idx} "
                  f"{'SUCCESS' if success else 'FAIL'} {rec['seconds']}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["libero", "libero_plus"], required=True)
    p.add_argument("--condition", required=True, help="run name, e.g. C0_full")
    p.add_argument("--server-url", default="ws://127.0.0.1:9100")
    p.add_argument("--out-root", default=os.path.join(os.environ.get("W", "work"), "outputs/p1"))
    p.add_argument("--suites", nargs="+", default=list(SUITE_MAX_STEPS), choices=list(SUITE_MAX_STEPS))
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    p.add_argument("--inits-per-task", type=int, default=5, help="libero bench: init states per task")
    p.add_argument("--frac", type=float, default=0.05, help="libero_plus: fraction of all tasks (<=0.2)")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--prompt-mode", choices=["orig", "counterfactual", "empty"], default="orig")
    p.add_argument("--state-mode", choices=["orig", "zero", "freeze", "offset"], default="orig")
    p.add_argument("--state-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    p.add_argument("--mask-views", nargs="*", type=int, default=[])
    p.add_argument("--branches", nargs="*", default=[],
                   choices=["no_lang", "no_state", "no_agent", "no_wrist"])
    p.add_argument("--ctx-mask", nargs="*", default=[], choices=["padding", "masked_view"],
                   help="hide these context segments from the action-head cross-attention")
    p.add_argument("--trim-padding", action="store_true", help="do not pad the VLM sequence to 1024 tokens")
    p.add_argument("--pad-to", type=int, default=0, help="pad the VLM sequence to this many tokens (default 1024)")
    p.add_argument("--attn-first", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
