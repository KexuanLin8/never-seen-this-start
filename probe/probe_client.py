"""Model-agnostic rollout client for the diagnosis (talks to probe_server.py).

Runs in the `libero` (py3.8) env for clean LIBERO and the `libero_plus` (py3.10) env for LIBERO-plus.
Every model is evaluated with the same LIBERO-plus style episode budget (per-suite max env steps,
10 settle steps), while rendering resolution, settle action and replanning horizon follow each
model's official evaluation:
  evo1         --render-res 448 --dummy-gripper 0  --horizon 15
  vla_adapter  --render-res 256 --dummy-gripper -1 --horizon 8
  lerobot      --render-res 256 --dummy-gripper -1 --horizon 0   (0 = the checkpoint's n_action_steps)

Interventions (same semantics for every model, applied by the server):
  --prompt-mode orig|counterfactual|empty, --state-mode orig|zero, --mask-views 0 1 (black image),
  --branches no_lang no_state no_agent no_wrist  (logged action changes, no effect on execution)
  --homing-ref ref.json  teleport the arm to the suite's nominal joint configuration before the
                         policy starts (homing test on LIBERO-plus robot-init perturbations)
  --disturb-at N         mid-episode push after N env steps (where homing cannot help), see push_arm()
  --chain-starts f.json  task chaining: start from the arm configuration in which ANOTHER task of the
                         suite ended (make_chain_starts.py); collision-free candidates only

Results: <out-root>/<bench>/<model-tag>/<condition>/results_shard*.jsonl, resumable and shardable.
"""

import os
import argparse
import asyncio
import json
import math
import random
import re
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


# LIBERO-plus builds task.language from the variant's file name, so non-language variants carry the
# perturbation parameters as a text suffix, e.g. "... place it on the plate view 0 0 100 0 0 initstate 484",
# "... light 12", "... tb 19", "... noise 14", "... add 10", "... moved level1 sample1".
# --prompt-mode clean strips that suffix (LLM-rewritten language variants carry no suffix and are kept).
_PLUS_SUFFIX = re.compile(
    r"\s+(?:view(?:\s+-?\d+(?:\.\d+)?){1,6}"
    r"|(?:initstate|light|table|tb|noise|add|level)\s+-?\d+(?:\.\d+)?"
    r"|(?:moved\s+)?level\d+(?:\s+sample\d+)?"
    r"|sample\d+)\s*$")


def clean_instruction(text):
    prev = None
    while prev != text:
        prev, text = text, _PLUS_SUFFIX.sub("", text)
    return text.strip()


def episode_seed(suite, task_id, init_idx):
    return zlib.crc32(f"{suite}/{task_id}/{init_idx}".encode()) % 1_000_000


def home_robot(env, q):
    """Teleport the arm joints to q (objects untouched) and re-anchor the controller.

    The OSC controller caches the end-effector pose; without a forced update its next goal is the
    stale (perturbed) pose and it drags the arm back, so refresh it before resetting the goal.
    """
    rs = env.env
    robot = rs.robots[0]
    q = np.asarray(q, dtype=np.float64)
    for i, name in enumerate(robot.robot_joints):  # keep the target inside the joint limits
        jid = rs.sim.model.joint_name2id(name)
        if rs.sim.model.jnt_limited[jid]:
            lo, hi = rs.sim.model.jnt_range[jid]
            q[i] = np.clip(q[i], lo + 1e-3, hi - 1e-3)
    rs.sim.data.qpos[robot._ref_joint_pos_indexes] = q
    rs.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    rs.sim.forward()
    ctrl = robot.controller
    if hasattr(ctrl, "update"):
        ctrl.update(force=True)
    if hasattr(ctrl, "reset_goal"):
        ctrl.reset_goal()


def build_episodes(args, benchmark, get_libero_path):
    """(suite, category, task_id, init_idx); LIBERO-plus uses the P1 nested subset of the P0 baseline subset."""
    eps = []
    if args.bench == "libero":
        for suite in args.suites:
            for tid in range(benchmark.get_benchmark_dict()[suite]().n_tasks):
                eps.extend((suite, "clean", tid, k) for k in range(args.inits_per_task))
        return eps
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


def flipped(img):
    return np.ascontiguousarray(img[::-1, ::-1])


def arm_contacts(env):
    """Bodies outside the robot (and its mount) that the arm or gripper currently touches."""
    sim, hits = env.env.sim, set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom1]) or ""
        b2 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom2]) or ""
        r1, r2 = b1.startswith(("robot0_", "gripper0_")), b2.startswith(("robot0_", "gripper0_"))
        other = b2 if r1 else b1
        if r1 != r2 and not other.startswith("mount0"):
            hits.add(other)
    return hits


def chain_start(env, obs, chain, suite, task_id, init_idx, frac=1.0):
    """Teleport the arm to the end configuration of another task of the suite (first collision-free
    candidate in a per-episode random order); frac < 1 stops part of the way from the nominal start to it
    (joint-space interpolation). Returns (q0, info)."""
    q0 = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
    cands = [(int(t), j) for t, qs in chain[suite].items() if int(t) != task_id for j in range(len(qs))]
    rng = np.random.RandomState(episode_seed(suite, task_id, init_idx) + 11)
    rng.shuffle(cands)
    base = arm_contacts(env)
    for tries, (t, j) in enumerate(cands, 1):
        home_robot(env, q0 + frac * (np.asarray(chain[suite][str(t)][j]) - q0))
        if not arm_contacts(env) - base:
            return q0, {"chain_src_task": t, "chain_src_demo": j, "tries": tries, "frac": frac}
    home_robot(env, q0)  # nothing fits: keep the nominal start and flag it
    return q0, {"chain_src_task": None, "tries": len(cands)}


async def run_episode(ws, env, args, suite, task_id, init_idx, instruction, init_state, homing_q, chain=None):
    env.reset()
    obs = env.set_init_state(init_state)
    start_info = None
    if args.start_perturb > 0:  # controlled start-configuration perturbation in the clean scene
        q0 = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
        d = np.random.RandomState(episode_seed(suite, task_id, init_idx) + 7).normal(size=q0.shape)
        home_robot(env, q0 + d / np.linalg.norm(d) * args.start_perturb)
        start_info = {"magnitude": args.start_perturb}
    if chain is not None:  # task chaining: start where another task of the suite ended
        q0, start_info = chain_start(env, obs, chain, suite, task_id, init_idx, args.chain_frac)
    if homing_q is not None:
        home_robot(env, homing_q)
    dummy = [0.0] * 6 + [args.dummy_gripper]
    for _ in range(10):
        obs, _, _, _ = env.step(dummy)
    home_err = None
    if homing_q is not None:  # second teleport after settling, then read a fresh observation
        home_robot(env, homing_q)
        obs = env.env._get_observations(force_update=True)
        home_err = float(np.linalg.norm(np.asarray(obs["robot0_joint_pos"]) - np.asarray(homing_q)))

    if start_info is not None:
        start_info["joint_disp"] = float(np.linalg.norm(np.asarray(obs["robot0_joint_pos"]) - q0))
    log = {"start_perturb": start_info, "home_err": home_err, "init_eef_pos": obs["robot0_eef_pos"].tolist(),
           "init_eef_quat": np.asarray(obs["robot0_eef_quat"]).tolist(),
           "init_joint_pos": np.asarray(obs["robot0_joint_pos"]).tolist() if "robot0_joint_pos" in obs else None,
           "dists": []}
    exec_spec = {}
    if args.mask_views:
        exec_spec["mask_views"] = list(args.mask_views)
    if args.state_mode == "zero":
        exec_spec["state"] = "zero"
    base_seed = episode_seed(suite, task_id, init_idx)
    horizon = args.horizon
    executed, step = 0, 0
    max_steps = SUITE_MAX_STEPS[suite]
    disturbed = args.disturb_at <= 0
    last_grip = args.dummy_gripper
    bias = np.zeros(3)
    if args.state_mode == "bias":  # constant calibration-style error on the eef position, random direction per episode
        d = np.random.RandomState(base_seed + 1).normal(size=3)
        bias = d / np.linalg.norm(d) * args.state_bias
        log["state_bias"] = bias.tolist()
    while executed < max_steps:
        if not disturbed and executed >= args.disturb_at:
            disturbed = True
            try:
                obs, done = push_arm(env, obs, args, base_seed, last_grip, log)
            except ValueError:
                log["env_horizon_hit"] = True
                return False, executed, log
            executed += args.disturb_steps
            if done:
                return True, executed, log
        state = raw_state(obs)
        state[:3] += bias
        req = {"image": [flipped(obs["agentview_image"]).tolist(), flipped(obs["robot0_eye_in_hand_image"]).tolist()],
               "state": state.tolist(), "prompt": instruction,
               "p1": {"seed": base_seed * 10000 + step, "exec": exec_spec,
                      "branches": args.branches if step % args.branch_every == 0 else [],
                      "dist_horizon": horizon or 0}}
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
        if horizon == 0:
            horizon = int(resp["default_horizon"])
        if "branch_dist" in resp:
            log["dists"].append(resp["branch_dist"])
        actions = np.asarray(resp["actions"])
        for i in range(min(horizon, len(actions))):
            try:
                obs, _, done, _ = env.step(actions[i].tolist())
            except ValueError:  # LIBERO env horizon (1000 steps) reached: failure
                log["env_horizon_hit"] = True
                return False, executed, log
            last_grip = float(actions[i][6])
            executed += 1
            if done:
                return True, executed, log
            if executed >= max_steps or (not disturbed and executed >= args.disturb_at):
                break  # re-query the policy after the push (remaining chunk is discarded)
        step += 1
    return False, executed, log


def push_arm(env, obs, args, seed, grip, log):
    """Mid-episode disturbance: a constant random push of the end effector for disturb_steps
    controller steps (position direction + small rotation, seeded per episode), gripper held at
    its last command so a grasped object is not released. Returns (obs, done)."""
    rng = np.random.RandomState(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    rot = rng.uniform(-1.0, 1.0, size=3) * args.disturb_rot
    act = np.concatenate([direction * args.disturb_scale, rot, [grip]]).tolist()
    q0 = np.asarray(obs["robot0_joint_pos"])
    p0 = np.asarray(obs["robot0_eef_pos"])
    done = False
    for _ in range(args.disturb_steps):
        obs, _, done, _ = env.step(act)
        if done:
            break
    log["disturb"] = {"at": args.disturb_at, "steps": args.disturb_steps, "scale": args.disturb_scale,
                      "rot": args.disturb_rot, "grip_held": grip,
                      "joint_disp": float(np.linalg.norm(np.asarray(obs["robot0_joint_pos"]) - q0)),
                      "eef_disp": float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - p0))}
    return obs, done


async def main_async(args):
    import websockets
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(0)
    random.seed(0)
    out_dir = Path(args.out_root) / args.bench / args.model_tag / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    out_file = out_dir / f"results_{args.file_tag or f'shard{args.shard}'}.jsonl"
    homing = json.loads(Path(args.homing_ref).read_text()) if args.homing_ref else None
    chain = json.loads(Path(args.chain_starts).read_text()) if args.chain_starts else None

    done = set()
    for f in out_dir.glob("results_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task_id"], r["init_idx"]))
    episodes = build_episodes(args, benchmark, get_libero_path)
    mine = episodes[args.shard::args.num_shards]
    todo = [e for e in mine if (e[0], e[2], e[3]) not in done]
    print(f"[{args.model_tag}/{args.condition} shard {args.shard}/{args.num_shards}] total={len(episodes)} "
          f"mine={len(mine)} todo={len(todo)}", flush=True)

    suites = {s: benchmark.get_benchmark_dict()[s]() for s in {e[0] for e in todo}}
    async with websockets.connect(args.server_url, max_size=100_000_000, ping_interval=None) as ws:
        for n, (suite, cat, task_id, init_idx) in enumerate(todo):
            ts = suites[suite]
            task = ts.get_task(task_id)
            orig = task.language
            if args.prompt_mode == "counterfactual":
                instruction = ts.get_task((task_id + 1) % ts.n_tasks).language
            elif args.prompt_mode == "empty":
                instruction = ""
            elif args.prompt_mode == "clean":
                instruction = clean_instruction(orig)
            else:
                instruction = orig
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=args.render_res,
                                     camera_widths=args.render_res)
            env.seed(0)
            t0 = time.time()
            try:
                success, steps, log = await run_episode(
                    ws, env, args, suite, task_id, init_idx, str(instruction),
                    ts.get_task_init_states(task_id)[init_idx], homing[suite] if homing else None, chain)
            finally:
                env.close()
            rec = {"bench": args.bench, "model": args.model_tag, "condition": args.condition, "suite": suite,
                   "category": cat, "task_id": task_id, "init_idx": init_idx, "success": bool(success),
                   "steps": steps, "seconds": round(time.time() - t0, 1), "instruction": str(instruction),
                   "orig_instruction": str(orig)}
            rec.update(log)
            with out_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{args.model_tag}/{args.condition} s{args.shard}] {n + 1}/{len(todo)} "
                  f"{suite}/{cat}/{task_id}/{init_idx} {'SUCCESS' if success else 'FAIL'} {rec['seconds']}s",
                  flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["libero", "libero_plus"], required=True)
    p.add_argument("--model-tag", required=True, help="evo1 | vla_adapter | smolvla | pi05 ...")
    p.add_argument("--condition", required=True)
    p.add_argument("--server-url", default="ws://127.0.0.1:9200")
    p.add_argument("--out-root", default=os.path.join(os.environ.get("W", "work"), "outputs/probe"))
    p.add_argument("--suites", nargs="+", default=list(SUITE_MAX_STEPS), choices=list(SUITE_MAX_STEPS))
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    p.add_argument("--inits-per-task", type=int, default=5)
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--render-res", type=int, default=256)
    p.add_argument("--dummy-gripper", type=float, default=-1.0)
    p.add_argument("--horizon", type=int, default=0, help="0 = server's default_horizon")
    p.add_argument("--prompt-mode", choices=["orig", "counterfactual", "empty", "clean"], default="orig",
                   help="clean = strip LIBERO-plus perturbation-parameter suffixes from the instruction")
    p.add_argument("--state-mode", choices=["orig", "zero", "bias"], default="orig",
                   help="bias = add a constant eef-position error of --state-bias metres (random direction per episode)")
    p.add_argument("--state-bias", type=float, default=0.02)
    p.add_argument("--mask-views", nargs="*", type=int, default=[])
    p.add_argument("--branches", nargs="*", default=[], choices=["no_lang", "no_state", "no_agent", "no_wrist"])
    p.add_argument("--branch-every", type=int, default=1,
                   help="compute branches on every k-th policy query (use >1 for horizon-1 policies)")
    p.add_argument("--homing-ref", default="", help="json {suite: [7 joint angles]}")
    p.add_argument("--start-perturb", type=float, default=0.0,
                   help="clean LIBERO: perturb the start joint configuration by this many rad (random direction)")
    p.add_argument("--chain-starts", default="", help="json from make_chain_starts.py (task-chaining starts)")
    p.add_argument("--chain-frac", type=float, default=1.0,
                   help="fraction of the way from the nominal start to the chained end configuration")
    p.add_argument("--disturb-at", type=int, default=0, help="push the arm after this many env steps (0 = off)")
    p.add_argument("--disturb-steps", type=int, default=10, help="controller steps the push lasts")
    p.add_argument("--disturb-scale", type=float, default=0.3, help="normalized OSC position action of the push")
    p.add_argument("--disturb-rot", type=float, default=0.1, help="max normalized OSC rotation action of the push")
    p.add_argument("--file-tag", default="", help="results file suffix (default shard<i>); "
                   "use a unique tag when several clients of one condition run with shard 0/1, e.g. per suite")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
