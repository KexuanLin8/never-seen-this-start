"""SO-101 data collection for the real-robot start-robustness experiment.

Reuses the controller and calibration of teleoperate_smooth_start.py. Three steps:

  python collect_data.py poses    teleoperate; H saves HOME, P a training start pose, T a test start pose, Q quits
  python collect_data.py check    slowly visits every saved pose and returns, nothing recorded (safety dry run)
  python collect_data.py record   the collection loop (tasks A and B alternate):
        1. you place the cube on the announced dot and press Enter
        2. the arm goes to a start pose (not recorded) and drives back to HOME (RECORDED: recovery clip), twice
        3. you bring the leader roughly to HOME and hold it still, then demonstrate the task (RECORDED); S saves, D discards
  python collect_data.py record --episodes 3 --out data/trial    (short trial)

Same idea as the simulation: every recovery clip ends at HOME in the same scene as the demonstration that follows,
so a clip + the demonstration form one synthetic episode, without extra human demonstrations.
The end pose of each demonstration feeds the chain pool (recovery clips of the OTHER task); every 5th is kept for testing.

Everything runs at 20 Hz because the wrist camera delivers about 20 frames/s.
Output per episode: <out>/ep_XXXX/{meta.json, demo.npz, demo_desk.mp4, demo_wrist.mp4, rec0.npz, rec0_desk.mp4, ...}
"""

import argparse
import json
import math
import msvcrt
import shutil
import sys
import time
from pathlib import Path

import av
import numpy as np

from teleoperate_smooth_start import JOINTS, Settings, SmoothController, calibrated_limits, configure_at_current_pose

HERE = Path(__file__).resolve().parent
POSES = HERE / "setup" / "poses.json"
FPS = 20
TASKS = {"A": "put the cube in the box", "B": "put the cube on the plate"}
DOTS = 9  # cube positions on the table, numbered 1-9
# How close the leader must be to HOME before a demonstration starts. The remaining body-joint difference is kept
# as a constant offset, so the follower does not move at the start and every demonstration begins exactly at HOME.
LEADER_TOL, GRIPPER_TOL = 15.0, 15.0


# ------------------------------------------------------------------ pure helpers (no hardware)
def min_jerk(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def path_duration(start, goal, speed):
    """Seconds for a joint-space move: the largest body-joint change at `speed` degrees/second, at least 1 s."""
    return max(1.0, max(abs(goal[j] - start[j]) for j in JOINTS[:-1]) / speed)


def interpolate(start, goal, tau):
    s = min_jerk(min(1.0, max(0.0, tau)))
    return {j: start[j] + (goal[j] - start[j]) * s for j in JOINTS}


def joint_distance_rad(a, b):
    return math.radians(math.sqrt(sum((a[j] - b[j]) ** 2 for j in JOINTS[:-1])))


def leader_offset(lead, home):
    """Constant offset that maps the current leader pose onto HOME (body joints only; the gripper stays absolute)."""
    return {j: (home[j] - lead[j] if j != "gripper" else 0.0) for j in JOINTS}


def load_poses():
    if POSES.exists():
        return json.loads(POSES.read_text())
    return {"home": None, "train": [], "test": [], "chain": {"A": [], "B": []}, "chain_test": {"A": [], "B": []}}


def save_poses(p):
    POSES.parent.mkdir(exist_ok=True)
    POSES.write_text(json.dumps(p, indent=1))


def write_video(path, frames, fps=FPS):
    with av.open(str(path), "w") as out:
        s = out.add_stream("libx264", rate=fps)
        s.height, s.width, s.pix_fmt = frames[0].shape[0], frames[0].shape[1], "yuv420p"
        s.options = {"crf": "20"}
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)


class Segment:
    """One recorded stretch: state and frames observed BEFORE the action of that step."""

    def __init__(self):
        self.state, self.action, self.t, self.desk, self.wrist = [], [], [], [], []

    def add(self, observed, command, desk, wrist, now):
        self.state.append([observed[j] for j in JOINTS])
        self.action.append([command[j] for j in JOINTS])
        self.t.append(now)
        self.desk.append(desk)
        self.wrist.append(wrist)

    def save(self, folder, name):
        np.savez_compressed(folder / f"{name}.npz", state=np.array(self.state, np.float32),
                            action=np.array(self.action, np.float32), t=np.array(self.t))
        write_video(folder / f"{name}_desk.mp4", self.desk)
        write_video(folder / f"{name}_wrist.mp4", self.wrist)


def key():
    return msvcrt.getwch().lower() if msvcrt.kbhit() else ""


# ------------------------------------------------------------------ hardware
class Rig:
    def __init__(self, a, cameras=True, use_leader=True):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        class CurrentPoseFollower(SO101Follower):
            def configure(self):
                configure_at_current_pose(self)

        root = HERE / "calibration"
        self.leader = None
        if use_leader:
            self.leader = SO101Leader(SO101LeaderConfig(port=a.leader_port, id="my_leader",
                                                        calibration_dir=root / "leader", use_degrees=True,
                                                        num_read_retries=6))
        self.follower = CurrentPoseFollower(SO101FollowerConfig(port=a.follower_port, id="my_follower",
                                                                calibration_dir=root / "follower", use_degrees=True,
                                                                max_relative_target=None, num_read_retries=6))
        self.limits = calibrated_limits(self.follower)
        self.cams = {}
        if cameras:
            from camera_names import open_cameras
            self.cams = open_cameras()

    def __enter__(self):
        if self.leader is not None:
            self.leader.connect(calibrate=False)
        self.follower.connect(calibrate=False)
        for c in self.cams.values():
            c.connect()
            c.async_read(timeout_ms=2000)
        return self

    def __exit__(self, *exc):
        print("Stopping: support the follower, torque is about to be released.", flush=True)
        for c in self.cams.values():
            c.disconnect()
        for dev in (self.follower, self.leader):
            if dev is not None and dev.bus.is_connected:
                dev.disconnect()

    @staticmethod
    def _read(read, who):
        """Serial reads can fail once in a while; retry so that one bad packet does not end the session."""
        for attempt in range(3):
            try:
                return read()
            except ConnectionError as exc:
                print(f"  {who} read failed ({attempt + 1}/3): {str(exc)[-40:]}", flush=True)
                error = exc
                time.sleep(0.01)
        raise error

    def observed(self):
        o = self._read(self.follower.get_observation, "follower")
        return {j: o[j + ".pos"] for j in JOINTS}

    def leader_pose(self):
        o = self._read(self.leader.get_action, "leader")
        return {j: o[j + ".pos"] for j in JOINTS}

    def send(self, command):
        self.follower.send_action({j + ".pos": command[j] for j in JOINTS})

    def frames(self):
        return self.cams["desk"].read_latest(), self.cams["wrist"].read_latest()

    def move(self, goal, speed, seg=None):
        """Smooth joint-space move to `goal`; a Segment records it."""
        start = self.observed()
        duration = path_duration(start, goal, speed)
        t0 = time.perf_counter()
        while True:
            tick = time.perf_counter()
            tau = (tick - t0) / duration
            command = interpolate(start, goal, tau)
            obs = self.observed()
            if seg is not None:
                seg.add(obs, command, *self.frames(), tick)
            self.send(command)
            if tau >= 1:
                return
            time.sleep(max(0, 1 / FPS - (time.perf_counter() - tick)))

    def teleop(self, on_step, offset=None):
        """Leader-follower loop with the smooth start; on_step(observed, command, now, following) returns True to stop."""
        settings = Settings(fps=FPS)
        now = time.perf_counter()
        ctrl = SmoothController(self.observed(), self.limits, settings, now)
        if offset is not None:  # the offset already maps the leader onto the follower: no alignment phase, only the speed ramp
            ctrl.follow_started = now
        last = now
        time.sleep(1 / FPS)
        while True:
            tick = time.perf_counter()
            obs, desired = self.observed(), self.leader_pose()
            if offset:
                desired = {j: desired[j] + offset[j] for j in JOINTS}
            now = time.perf_counter()
            command = ctrl.step(obs, desired, now, now - last)
            last = now
            stop = on_step(obs, command, now, ctrl.phase == "FOLLOW")
            self.send(command)
            if stop:
                return obs
            time.sleep(max(0, 1 / FPS - (time.perf_counter() - tick)))


# ------------------------------------------------------------------ commands
def cmd_poses(a):
    p = load_poses()
    with Rig(a, cameras=False) as rig:
        print("Keep the leader still until FOLLOW. Then: H = HOME, P = training start pose, T = test start pose, Q = quit.")
        announced = [False]

        def step(obs, command, now, following):
            if following and not announced[0]:
                announced[0] = True
                print("FOLLOW: you can move now.", flush=True)
            k = key() if following else ""
            if k == "h":
                p["home"] = obs
                print("HOME saved.", flush=True)
            elif k in ("p", "t"):
                if not p["home"]:
                    print("Save HOME first (H).", flush=True)
                    return False
                name = "train" if k == "p" else "test"
                p[name].append(obs)
                print(f"{name} pose {len(p[name])}: {joint_distance_rad(obs, p['home']):.2f} rad from HOME", flush=True)
            if k in ("h", "p", "t"):
                save_poses(p)
            return k == "q"

        rig.teleop(step)
    print(f"HOME set: {bool(p['home'])}, training poses: {len(p['train'])}, test poses: {len(p['test'])}")


def cmd_check(a):
    p = load_poses()
    with Rig(a, cameras=False) as rig:
        rig.move(p["home"], 20)
        for name in ("train", "test"):
            for i, pose in enumerate(p[name], 1):
                print(f"{name} pose {i}/{len(p[name])}  ({joint_distance_rad(pose, p['home']):.2f} rad)", flush=True)
                rig.move(pose, 25)
                time.sleep(0.5)
                rig.move(p["home"], 25)
        print("All poses visited. If the arm touched anything, delete that pose from setup/poses.json.")


def wait_for_leader(rig, home):
    print("Bring the LEADER roughly to the HOME pose and hold it still (it starts automatically when close)...", flush=True)
    t, since = 0.0, None
    while True:
        lead = rig.leader_pose()
        worst = max(JOINTS[:-1], key=lambda j: abs(lead[j] - home[j]))
        err = abs(lead[worst] - home[worst])
        gerr = abs(lead["gripper"] - home["gripper"])
        now = time.perf_counter()
        since = (since or now) if err <= LEADER_TOL and gerr <= GRIPPER_TOL else None
        if since is not None and now - since >= 0.5:
            return leader_offset(lead, home)
        if now - t > 1:
            t = now
            print(f"  largest difference {err:.0f} deg at {worst} (needs {LEADER_TOL:.0f}), gripper {gerr:.0f}", flush=True)
        time.sleep(0.05)


def record_demo(rig, offset, task):
    seg = Segment()
    print(f"TASK {task}: {TASKS[task]}. Press S when done (stay where you are), D to discard.", flush=True)
    state = {"go": False, "result": None}

    def step(obs, command, now, following):
        if following and not state["go"]:
            state["go"] = True
            print("GO", flush=True)
        if state["go"]:
            seg.add(obs, command, *rig.frames(), now)
        k = key()
        if k in ("s", "d") and state["go"]:
            state["result"] = k
            return True
        return False

    end = rig.teleop(step, offset)
    return seg, end, state["result"] == "s"


def cmd_record(a):
    p = load_poses()
    if not p["home"] or not p["train"]:
        sys.exit("Run 'poses' first: HOME and training poses are needed.")
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    for folder in out.glob("ep_*"):  # an episode interrupted before saving has no meta.json
        if not (folder / "meta.json").exists():
            shutil.rmtree(folder)
            print(f"removed unfinished {folder.name}", flush=True)
    n = len(list(out.glob("ep_*")))
    rng = np.random.default_rng(1000 + n)
    home = p["home"]
    with Rig(a) as rig:
        rig.move(home, 20)
        done = 0
        while done < a.episodes:
            task = "AB"[n % 2]
            dot = (n // 2) % DOTS + 1
            print(f"\n=== episode {n}: task {task} '{TASKS[task]}'. Put the cube on dot {dot}, press Enter (q + Enter quits).")
            if input().strip().lower() == "q":
                break
            folder = out / f"ep_{n:04d}"
            folder.mkdir()
            meta = {"task": TASKS[task], "task_id": task, "dot": dot, "clips": []}
            other = "B" if task == "A" else "A"
            for c in range(a.clips):  # clip 0: small perturbation; clip 1: where the other task ended, if known
                pool, kind = p["train"], "small"
                if c % 2 == 1 and p["chain"][other]:
                    pool, kind = p["chain"][other], "chain"
                pose = dict(pool[int(rng.integers(len(pool)))])
                pose["gripper"] = home["gripper"]
                print(f"  recovery clip {c + 1}/{a.clips} ({kind}, {joint_distance_rad(pose, home):.2f} rad)", flush=True)
                rig.move(pose, a.speed)
                time.sleep(0.3)
                seg = Segment()
                rig.move(home, a.speed, seg)
                seg.save(folder, f"rec{c}")
                meta["clips"].append({"kind": kind, "rad": joint_distance_rad(pose, home), "steps": len(seg.t)})
            offset = wait_for_leader(rig, home)
            seg, end, keep = record_demo(rig, offset, task)
            if keep:
                seg.save(folder, "demo")
                meta["steps"] = len(seg.t)
                (folder / "meta.json").write_text(json.dumps(meta, indent=1))
                seen = len(p["chain"][task]) + len(p["chain_test"][task])
                bucket = "chain_test" if seen % 5 == 4 else "chain"  # every 5th end pose is held out for testing
                p[bucket][task].append(end)
                save_poses(p)
                n += 1
                done += 1
                print(f"  saved: {len(seg.t)} steps ({len(seg.t) / FPS:.0f} s). Total {n}.", flush=True)
            else:
                shutil.rmtree(folder)
                print("  discarded.", flush=True)
            print("  Let go of the leader. The arm returns to HOME.", flush=True)
            rig.move(home, 25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["poses", "check", "record"])
    p.add_argument("--leader-port", default="COM3")
    p.add_argument("--follower-port", default="COM4")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--clips", type=int, default=2, help="recovery clips per demonstration")
    p.add_argument("--speed", type=float, default=40, help="recovery speed, degrees/second")
    a = p.parse_args()
    {"poses": cmd_poses, "check": cmd_check, "record": cmd_record}[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
