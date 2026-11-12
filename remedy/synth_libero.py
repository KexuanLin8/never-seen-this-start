"""Synthesize LIBERO demonstrations that start from diverse arm configurations (the remedy).

Diagnosis: every LIBERO demo of a suite starts from the SAME arm configuration, so policies never learn
how to begin a task from anywhere else (LIBERO-plus robot-init failures disappear with homing).

For each original demo (LIBERO hdf5, which stores the full simulator state):
  mode "start" (ours): restore the demo's initial simulator state, perturb the arm joints by a random
      direction x magnitude in [--mag-min, --mag-max] rad, reject the sample if the arm would disturb
      the scene, drive the end effector back to the demo's start pose with a proportional OSC
      controller (the "recovery" segment), then replay the demo's actions (no-ops skipped).
  mode "mid" (DART-style baseline): replay the demo, push the arm at a random time, drive it back to
      the pose the demo had at that time, then continue the replay.
  mode "replay": plain replay (sanity check of the pipeline; should succeed for ~all demos).
  mode "chain": like "start", but the arm starts where a demo of ANOTHER task of the suite ended
      (task chaining; --chain-pool holds demos 10-49, evaluation uses demos 0-9). Needs a longer
      recovery, e.g. --recovery-steps 400.
Only episodes whose replay ends in task success are kept.

Recorded per step (observation BEFORE the action, same conventions as the LeRobot LIBERO datasets
used to train Evo-1 / SmolVLA): agentview + wrist images 256x256 rotated by 180 deg, state = eef pos +
axis-angle + gripper qpos (8), action = 7-d OSC delta + gripper (-1 open / +1 close, raw LIBERO;
build_lerobot_v21.py converts it to the IPEC convention 1 open / 0 close).

Each kept episode goes to <staging>/<suite>/<episode_id>/{agent.mp4, wrist.mp4, data.npz, meta.json};
build_lerobot_v21.py turns the staging folder into a LeRobot v2.1 dataset.

Run in the `libero` env (py3.8). Shard across processes with --shard/--num-shards, e.g.
  python synth_libero.py --suite libero_spatial --mode start --per-demo 1 --shard 0 --num-shards 8
"""

import os
import argparse
import json
import math
import time
import zlib
from pathlib import Path

import h5py
import imageio
import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
POS_SCALE, ROT_SCALE = 0.05, 0.5  # OSC_POSE output_max: metres / radians per unit action


# ----------------------------------------------------------------------------- helpers
def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def quat2mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat2rotvec(R):
    angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


def raw_state(obs):
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"])).astype(np.float32)


def is_noop(action, prev_action):
    """Same rule as the OpenVLA LIBERO regeneration (the "_no_noops" datasets)."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < 1e-4
    return np.linalg.norm(action[:-1]) < 1e-4 and action[-1] == prev_action[-1]


class Recorder:
    def __init__(self):
        self.agent, self.wrist, self.states, self.actions, self.phase = [], [], [], [], []

    def step(self, env, obs, action, phase):
        self.agent.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        self.wrist.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        self.states.append(raw_state(obs))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.phase.append(phase)
        return env.step(list(map(float, action)))


def robot_handles(env):
    rs = env.env
    robot = rs.robots[0]
    return rs, robot


def non_robot_qpos(rs, robot):
    mask = np.ones(rs.sim.model.nq, dtype=bool)
    mask[robot._ref_joint_pos_indexes] = False
    mask[robot._ref_gripper_joint_pos_indexes] = False
    return rs.sim.data.qpos[mask].copy()


def arm_contacts(rs):
    """Contacts between the arm/gripper and anything that is not part of the robot or its mount."""
    sim, hits = rs.sim, []
    robot_prefix = ("robot0_", "gripper0_")
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom1]) or ""
        b2 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom2]) or ""
        r1, r2 = b1.startswith(robot_prefix), b2.startswith(robot_prefix)
        other = b2 if r1 else b1
        if r1 != r2 and not other.startswith("mount0"):
            hits.append(other)
    return hits


def teleport_joints(env, q):
    rs, robot = robot_handles(env)
    idx = robot._ref_joint_pos_indexes
    lo, hi = [], []
    for name in robot.robot_joints:  # same order as _ref_joint_pos_indexes
        jid = rs.sim.model.joint_name2id(name)
        r = rs.sim.model.jnt_range[jid]
        limited = rs.sim.model.jnt_limited[jid]
        lo.append(r[0] if limited else -np.inf)
        hi.append(r[1] if limited else np.inf)
    q = np.clip(q, np.array(lo) + 1e-3, np.array(hi) - 1e-3)
    rs.sim.data.qpos[idx] = q
    rs.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    rs.sim.forward()
    if hasattr(robot.controller, "update"):
        robot.controller.update(force=True)
    if hasattr(robot.controller, "reset_goal"):
        robot.controller.reset_goal()


def drive_to_pose(env, obs, rec, target_pos, target_quat, gripper, max_steps, gain, pos_tol, rot_tol, phase):
    """Proportional OSC control of the end effector to (target_pos, target_quat); recorded."""
    Rt = quat2mat(target_quat)
    for _ in range(max_steps):
        dp = target_pos - obs["robot0_eef_pos"]
        rv = mat2rotvec(Rt @ quat2mat(obs["robot0_eef_quat"]).T)
        if np.linalg.norm(dp) < pos_tol and np.linalg.norm(rv) < rot_tol:
            return obs, True
        a = np.concatenate([np.clip(gain * dp / POS_SCALE, -1, 1), np.clip(gain * rv / ROT_SCALE, -1, 1), [gripper]])
        obs, _, done, _ = rec.step(env, obs, a, phase)
        if done:
            return obs, True
    dp = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
    rv = np.linalg.norm(mat2rotvec(Rt @ quat2mat(obs["robot0_eef_quat"]).T))
    return obs, (dp < 2 * pos_tol and rv < 2 * rot_tol)


def write_video(path, frames, fps):
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])


# ----------------------------------------------------------------------------- one episode
def synth_episode(env, orig_states, orig_actions, mode, rng, args, chain_cands=None):
    """Returns (Recorder, info) for a successful episode, or (None, info) if rejected / failed."""
    rs, robot = robot_handles(env)
    env.reset()
    obs = env.set_init_state(orig_states[0])
    info = {"mode": mode}
    rec = Recorder()

    if mode in ("start", "chain"):
        q0 = np.array(obs["robot0_joint_pos"], dtype=np.float64)
        p0, quat0 = np.array(obs["robot0_eef_pos"]), np.array(obs["robot0_eef_quat"])
        # reference: how the scene settles over the same 5 steps WITHOUT perturbing the arm (some scenes
        # drift by themselves), so that only the arm's own effect on the objects is measured below
        for _ in range(5):
            env.step([0.0] * 6 + [-1.0])
        scene0 = non_robot_qpos(rs, robot)
        env.reset()
        obs = env.set_init_state(orig_states[0])
        base_hits = set(arm_contacts(rs))  # contacts that already exist at the nominal start (e.g. base/table)
        if mode == "start":
            d = rng.normal(size=q0.shape)
            target = q0 + d / np.linalg.norm(d) * rng.uniform(args.mag_min, args.mag_max)
            tries = [target]
        else:  # chain: where another task of the suite ended (a few candidates, first collision-free one)
            tries = [np.asarray(chain_cands[i]) for i in rng.choice(len(chain_cands), args.chain_tries, replace=False)]
        for target in tries:
            teleport_joints(env, target)
            hits = set(arm_contacts(rs)) - base_hits  # the new configuration must not intersect the table or objects
            if not hits:
                break
            teleport_joints(env, q0)
        info.update(magnitude=float(np.linalg.norm(target - q0)))
        if hits:
            info.update(reject="arm_collision", contact_with=sorted(set(hits))[:5])
            return None, info
        for _ in range(5):  # let the scene settle with the arm at its new configuration
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        moved = float(np.abs(non_robot_qpos(rs, robot) - scene0).max())
        info.update(joint_disp=float(np.linalg.norm(np.array(obs["robot0_joint_pos"]) - q0)),
                    eef_disp=float(np.linalg.norm(np.array(obs["robot0_eef_pos"]) - p0)), scene_moved=moved)
        if moved > args.max_scene_move:  # safety net only: objects knocked over / pushed away
            info["reject"] = "scene_disturbed"
            return None, info
        obs, ok = drive_to_pose(env, obs, rec, p0, quat0, -1.0, args.recovery_steps, args.gain,
                                args.pos_tol, args.rot_tol, "recovery")
        info["recovery_len"] = len(rec.actions)
        if not ok:
            info["reject"] = "recovery_not_converged"
            return None, info

    push_at = None
    if mode == "mid":
        n_real = sum(1 for i, a in enumerate(orig_actions) if not is_noop(a, orig_actions[i - 1] if i else None))
        push_at = int(rng.uniform(0.1, 0.6) * n_real)
        info["push_at"] = push_at

    prev, k = None, 0
    for a in orig_actions:
        if is_noop(a, prev):
            continue
        prev = a
        if push_at is not None and k == push_at:
            target_pos, target_quat = np.array(obs["robot0_eef_pos"]), np.array(obs["robot0_eef_quat"])
            d = rng.normal(size=3)
            push = np.concatenate([d / np.linalg.norm(d) * args.push_scale, rng.uniform(-1, 1, 3) * args.push_rot,
                                   [a[-1]]])
            for _ in range(args.push_steps):  # the push itself is NOT recorded (it is an external disturbance)
                obs, _, _, _ = env.step(list(map(float, push)))
            obs, ok = drive_to_pose(env, obs, rec, target_pos, target_quat, float(a[-1]), args.recovery_steps,
                                    args.gain, args.pos_tol, args.rot_tol, "recovery")
            if not ok:
                info["reject"] = "recovery_not_converged"
                return None, info
        obs, _, done, _ = rec.step(env, obs, a, "demo")
        k += 1
        if done:
            info.update(success=True, length=len(rec.actions))
            return rec, info
    info.update(success=False, reject="replay_failed", length=len(rec.actions))
    return None, info


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--mode", choices=["start", "mid", "replay", "chain"], default="start")
    p.add_argument("--chain-pool", default=os.path.join(os.environ.get("W", "work"), "outputs/probe/chain_pool_train.json"),
                   help="chain mode: end configurations of the demos NOT used for evaluation (make_chain_starts.py)")
    p.add_argument("--chain-tries", type=int, default=5, help="chain mode: candidates tried before rejecting")
    p.add_argument("--hdf5-dir", default=os.path.join(os.environ.get("W", "work"), "data/libero_hdf5"))
    p.add_argument("--staging", default=os.path.join(os.environ.get("W", "work"), "data/synth_staging"))
    p.add_argument("--tag", default="", help="staging sub-folder (default <mode>)")
    p.add_argument("--per-demo", type=int, default=1, help="synthetic episodes attempted per original demo")
    p.add_argument("--max-demos-per-task", type=int, default=0, help="0 = all demos (for quick tests use 2)")
    p.add_argument("--mag-min", type=float, default=0.05)
    p.add_argument("--mag-max", type=float, default=0.40)
    p.add_argument("--max-scene-move", type=float, default=0.02,
                   help="safety net on object qpos change during settling (contacts are the primary check)")
    p.add_argument("--recovery-steps", type=int, default=120)
    p.add_argument("--gain", type=float, default=0.6)
    p.add_argument("--pos-tol", type=float, default=0.004)
    p.add_argument("--rot-tol", type=float, default=math.radians(2.0))
    p.add_argument("--push-scale", type=float, default=0.3)
    p.add_argument("--push-rot", type=float, default=0.1)
    p.add_argument("--push-steps", type=int, default=10)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345, help="different from every evaluation seed")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    ts = benchmark.get_benchmark_dict()[a.suite]()
    pool = json.loads(Path(a.chain_pool).read_text())[a.suite] if a.mode == "chain" else None
    jobs = []
    for tid in range(ts.n_tasks):
        task = ts.get_task(tid)
        f = Path(a.hdf5_dir) / a.suite / f"{task.name}_demo.hdf5"
        if not f.exists():
            raise SystemExit(f"missing demo file {f}")
        with h5py.File(f, "r") as h:
            demos = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1]))
        if a.max_demos_per_task:
            demos = demos[: a.max_demos_per_task]
        jobs.extend((tid, str(f), dk, k) for dk in demos for k in range(a.per_demo))
    mine = jobs[a.shard::a.num_shards]
    out_root = Path(a.staging) / (a.tag or a.mode) / a.suite
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"log_shard{a.shard}.jsonl"
    done = set()
    if log_path.exists():
        done = {json.loads(l)["episode_id"] for l in log_path.read_text().splitlines() if l.strip()}
    print(f"[{a.suite}/{a.mode} shard {a.shard}/{a.num_shards}] jobs={len(mine)} already={len(done)}", flush=True)

    env, cur_tid = None, None
    for tid, fpath, demo_key, k in mine:
        ep_id = f"t{tid:02d}_{demo_key}_k{k}"
        if ep_id in done:
            continue
        if tid != cur_tid:
            if env is not None:
                env.close()
            task = ts.get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(0)
            cur_tid, language = tid, task.language
        with h5py.File(fpath, "r") as h:
            states = h["data"][demo_key]["states"][()]
            actions = h["data"][demo_key]["actions"][()]
        rng = np.random.RandomState((a.seed * 1000003 + zlib.crc32(ep_id.encode())) % (2 ** 31))
        t0 = time.time()
        try:
            cands = [q for t, qs in pool.items() if int(t) != tid for q in qs] if pool else None
            rec, info = synth_episode(env, states, actions, a.mode, rng, a, cands)
        except Exception as exc:  # keep going; log the failure
            rec, info = None, {"reject": f"exception: {type(exc).__name__}: {exc}"}
        info.update(episode_id=ep_id, task_id=tid, demo=demo_key, seconds=round(time.time() - t0, 1))
        if rec is not None:
            d = out_root / ep_id
            d.mkdir(exist_ok=True)
            write_video(d / "agent.mp4", rec.agent, a.fps)
            write_video(d / "wrist.mp4", rec.wrist, a.fps)
            np.savez_compressed(d / "data.npz", state=np.stack(rec.states), action=np.stack(rec.actions),
                                phase=np.array(rec.phase))
            (d / "meta.json").write_text(json.dumps({"task": language, **info}))
            info["kept"] = True
        else:
            info["kept"] = False
        with log_path.open("a") as fh:
            fh.write(json.dumps(info) + "\n")
        print(f"{ep_id} kept={info['kept']} {info.get('reject', '')} {info['seconds']}s", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
