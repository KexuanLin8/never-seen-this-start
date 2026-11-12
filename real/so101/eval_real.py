"""Real-robot evaluation of a trained policy (run in the lerobot-so101 environment, with policy_server.py running).

  python eval_real.py --model orig --port 6001 --condition nominal
  python eval_real.py --model startchain --port 6002 --condition perturbed
  conditions: nominal (HOME) | perturbed (held-out test poses) | chain (held-out end poses of the other task)

Each condition has 20 trials: tasks A and B x 5 cube positions x 2. The schedule is fixed, so both models face the same
task, cube position and start pose. Finished trials are skipped when the command is run again.

Keys during a trial: S = success, F = failure, X = stop without recording this trial.
Results: results/real_eval.jsonl, desk video of each trial in results/videos/.
Try-outs: add --results results/tryout.jsonl so that they do not count.
"""

import argparse
import json
import time
from multiprocessing.connection import Client
from pathlib import Path

from collect_data import FPS, HERE, JOINTS, TASKS, Rig, joint_distance_rad, key, load_poses, write_video

AUTH = b"so101-eval"
N_TRIALS, N_POSITIONS = 20, 5
MAX_BODY_STEP, MAX_GRIPPER_STEP = 6.0, 10.0  # per control step (20 Hz): 120 deg/s and 200 points/s


def schedule(condition, poses, n_trials=N_TRIALS):
    """The fixed trial list of a condition: (task, cube position, start pose name, start pose)."""
    home = poses["home"]
    trials = []
    for i in range(n_trials):
        task = "AB"[i % 2]
        position = (i // 2) % N_POSITIONS + 1
        if condition == "nominal":
            name, pose = "home", home
        elif condition == "perturbed":
            k = i % len(poses["test"])
            name, pose = f"test{k}", poses["test"][k]
        else:
            other = "B" if task == "A" else "A"
            pool = poses["chain_test"][other]
            k = (i // 2) % len(pool)
            name, pose = f"end{other}{k}", pool[k]
        trials.append({"trial": i, "task": task, "position": position, "start": name,
                       "pose": {**pose, "gripper": home["gripper"]}})
    return trials


def limit_step(target, previous, limits):
    """Rate- and range-limit a policy action, so that a bad prediction cannot jerk the arm."""
    out = {}
    for j in JOINTS:
        step = MAX_GRIPPER_STEP if j == "gripper" else MAX_BODY_STEP
        value = max(previous[j] - step, min(previous[j] + step, target[j]))
        out[j] = max(limits[j][0], min(limits[j][1], value))
    return out


def done_trials(results, model, condition):
    if not results.exists():
        return set()
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    return {r["trial"] for r in rows if r["model"] == model and r["condition"] == condition}


def run_policy(rig, conn, task, seed0, horizon, max_steps):
    """Execute the policy until the operator judges the trial. Returns (result, steps, mean inference ms, desk frames)."""
    desk_video, latencies, steps, query = [], [], 0, 0
    command = rig.observed()
    print("Running. S = success, F = failure, X = stop.", flush=True)
    while steps < max_steps:
        obs = rig.observed()
        desk, wrist = rig.frames()
        conn.send({"desk": desk, "wrist": wrist, "state": [obs[j] for j in JOINTS], "task": TASKS[task], "seed": seed0 + query})
        reply = conn.recv()
        latencies.append(reply["ms"])
        query += 1
        for action in reply["actions"][:horizon]:
            tick = time.perf_counter()
            pressed = key()
            if pressed in ("s", "f", "x"):
                return pressed, steps, sum(latencies) / len(latencies), desk_video
            command = limit_step({j: float(action[i]) for i, j in enumerate(JOINTS)}, command, rig.limits)
            desk_video.append(rig.frames()[0])
            rig.send(command)
            steps += 1
            time.sleep(max(0, 1 / FPS - (time.perf_counter() - tick)))
    print("Time limit reached: press S or F to judge the trial.", flush=True)
    while True:
        pressed = key()
        if pressed in ("s", "f", "x"):
            return pressed, steps, sum(latencies) / len(latencies), desk_video
        time.sleep(0.05)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="name recorded in the results; must match the server's checkpoint folder")
    p.add_argument("--port", type=int, default=6001)
    p.add_argument("--condition", choices=["nominal", "perturbed", "chain"], required=True)
    p.add_argument("--horizon", type=int, default=10, help="actions executed per prediction")
    p.add_argument("--max-seconds", type=float, default=40)
    p.add_argument("--speed", type=float, default=25, help="speed of the moves between trials, degrees/second")
    p.add_argument("--results", default="results/real_eval.jsonl")
    p.add_argument("--trials", type=int, default=N_TRIALS, help="trials of this condition (40 adds trials 20-39)")
    p.add_argument("--leader-port", default="COM3")
    p.add_argument("--follower-port", default="COM4")
    a = p.parse_args()

    results = HERE / a.results
    poses = load_poses()
    home = poses["home"]
    trials = [t for t in schedule(a.condition, poses, a.trials) if t["trial"] not in done_trials(results, a.model, a.condition)]
    print(f"{a.model} / {a.condition}: {len(trials)} trials left", flush=True)
    if not trials:
        return
    conn = Client(("127.0.0.1", a.port), authkey=AUTH)
    conn.send({"cmd": "info"})
    served = conn.recv()["ckpt"]
    if served != a.model:
        raise SystemExit(f"The server on port {a.port} runs '{served}', not '{a.model}'.")
    (results.parent / "videos").mkdir(parents=True, exist_ok=True)
    with Rig(a, use_leader=False) as rig:
        rig.move(home, a.speed)
        for t in trials:
            print(f"\n=== trial {t['trial']}: task {t['task']} '{TASKS[t['task']]}', cube on position {t['position']}. "
                  f"Place it and press Enter (q + Enter quits).", flush=True)
            if input().strip().lower() == "q":
                break
            rig.move(t["pose"], a.speed)
            time.sleep(0.5)
            result, steps, ms, video = run_policy(rig, conn, t["task"], 1000 * t["trial"], a.horizon, int(a.max_seconds * FPS))
            if result == "x":
                print("Trial not recorded.", flush=True)
                rig.move(home, a.speed)
                break
            name = f"{a.model}_{a.condition}_{t['trial']:02d}"
            if video:
                write_video(results.parent / "videos" / f"{name}.mp4", video)
            row = {"model": a.model, "condition": a.condition, "trial": t["trial"], "task": t["task"], "position": t["position"],
                   "start": t["start"], "start_rad": joint_distance_rad(t["pose"], home), "success": result == "s", "steps": steps,
                   "inference_ms": ms}
            with results.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {'SUCCESS' if result == 's' else 'FAILURE'} ({steps / FPS:.1f} s, inference {ms:.0f} ms)", flush=True)
            rig.move(home, a.speed)
    conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
