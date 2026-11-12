"""SO-101 teleoperation: measured slow alignment, then faster tracking.

Uses the existing calibration files. No cameras or automatic calibration.
Run --help without opening a serial port. Joint-space limits are not collision avoidance.
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class Settings:
    fps: float = 30
    startup_speed: float = 15
    follow_speed: float = 90
    startup_gripper_speed: float = 25
    follow_gripper_speed: float = 150
    startup_lead: float = 10
    follow_lead: float = 30
    startup_gripper_lead: float = 15
    follow_gripper_lead: float = 35
    angle_tolerance: float = 3
    gripper_tolerance: float = 5
    settle_seconds: float = 0.6
    ramp_seconds: float = 1
    startup_timeout: float = 60
    max_gap: float = 0.25


class SmoothController:
    """Pure controller; switching depends on measured pose, never just a timer."""

    def __init__(self, initial, limits, settings, now):
        self.settings = settings
        self.limits = limits
        self.validate(initial)
        self.command = dict(initial)
        self.started = now
        self.close_since = None
        self.follow_started = None
        self.max_angle_error = 0
        self.gripper_error = 0
        self.limited = []

    @staticmethod
    def validate(pose):
        if set(pose) != set(JOINTS) or not all(math.isfinite(pose[name]) for name in JOINTS):
            raise ValueError("Expected six finite SO-101 joint positions.")

    @property
    def phase(self):
        return "ALIGN" if self.follow_started is None else "FOLLOW"

    def step(self, observed, desired, now, dt):
        self.validate(observed)
        self.validate(desired)
        s = self.settings
        if not math.isfinite(dt) or dt < 0:
            raise RuntimeError(f"Invalid control interval {dt!r}; stopping.")
        if dt > s.max_gap:
            raise RuntimeError(f"Control gap {dt:.6f}s exceeded limit; stopping instead of catching up.")
        if dt == 0:
            # A repeated clock sample is not a stall. Hold without inventing a
            # time step, advancing the target, or completing the settle window.
            return dict(self.command)
        target = {name: clamp(desired[name], *self.limits[name]) for name in JOINTS}
        self.limited = [name for name in JOINTS if target[name] != desired[name]]
        self.max_angle_error = max(abs(target[name] - observed[name]) for name in JOINTS[:-1])
        self.gripper_error = abs(target["gripper"] - observed["gripper"])
        if self.follow_started is None:
            close = self.max_angle_error <= s.angle_tolerance and self.gripper_error <= s.gripper_tolerance
            self.close_since = (now if self.close_since is None else self.close_since) if close else None
            if self.close_since is not None and now - self.close_since >= s.settle_seconds:
                self.follow_started = now
            elif now - self.started >= s.startup_timeout:
                raise RuntimeError("Alignment timed out. Check pose, obstructions and calibration; fast mode was not enabled.")
        mix = 0 if self.follow_started is None else clamp((now - self.follow_started) / s.ramp_seconds, 0, 1)
        next_command = {}
        for name in JOINTS:
            gripper = name == "gripper"
            slow = s.startup_gripper_speed if gripper else s.startup_speed
            fast = s.follow_gripper_speed if gripper else s.follow_speed
            lead_slow = s.startup_gripper_lead if gripper else s.startup_lead
            lead_fast = s.follow_gripper_lead if gripper else s.follow_lead
            distance = (slow + (fast - slow) * mix) * dt
            lead = lead_slow + (lead_fast - lead_slow) * mix
            # Intersect rate, measured tracking-error and calibrated position bounds.
            low = max(self.command[name] - distance, observed[name] - lead, self.limits[name][0])
            high = min(self.command[name] + distance, observed[name] + lead, self.limits[name][1])
            if low > high:
                raise RuntimeError(f"{name}: measured pose moved outside tracking bounds; stopping.")
            next_command[name] = clamp(target[name], low, high)
        self.command = next_command
        return dict(next_command)


def configure_at_current_pose(robot):
    """Configure with torque off; acknowledge all hold targets before enabling."""
    bus = robot.bus
    if not robot.calibration or not robot.is_calibrated:
        raise RuntimeError("Follower calibration missing or inconsistent. Use the original calibration command first.")
    bus.disable_torque()
    # Do not use torque_disabled(): it enables torque even if configuration fails.
    bus.configure_motors()
    for name in bus.motors:
        bus.set_operating_mode("position", name)
        bus.write("P_Coefficient", name, robot.config.position_p_coefficient)
        bus.write("I_Coefficient", name, robot.config.position_i_coefficient)
        bus.write("D_Coefficient", name, robot.config.position_d_coefficient)
        if name == "gripper":
            bus.write("Max_Torque_Limit", name, 500)
            bus.write("Protection_Current", name, 250)
            bus.write("Overload_Torque", name, 25)
    current = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    for name in bus.motors:
        # Unicast writes wait for acknowledgements, unlike the normal sync_write.
        bus.write("Goal_Position", name, current[name], normalize=False)
        if bus.read("Goal_Position", name, normalize=False) != current[name]:
            raise RuntimeError(f"{name}: initial hold target could not be verified.")
    bus.enable_torque()


def calibrated_limits(robot):
    limits = {}
    for name in JOINTS:
        c = robot.calibration[name]
        if c.range_min >= c.range_max:
            raise ValueError(f"Invalid calibration range: {name}")
        if name == "gripper":
            limits[name] = (0, 100)
        else:
            half_span = (c.range_max - c.range_min) * 180 / 4095
            limits[name] = (-half_span, half_span)
    return limits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default="COM3")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--startup-speed", type=float, default=15, help="Initial body target rate, degrees/second")
    parser.add_argument("--follow-speed", type=float, default=90, help="Tracking body target rate, degrees/second")
    parser.add_argument("--startup-gripper-speed", type=float, default=25, help="Initial gripper target rate, percent points/second")
    parser.add_argument("--follow-gripper-speed", type=float, default=150, help="Tracking gripper target rate, percent points/second")
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args(argv)
    for key in ("startup_speed", "follow_speed", "startup_gripper_speed", "follow_gripper_speed", "startup_timeout", "fps"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if not 10 <= args.fps <= 60:
        parser.error("--fps must be between 10 and 60")
    if args.leader_port.upper() == args.follower_port.upper():
        parser.error("Leader and follower ports must differ")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Lazy imports keep --help and controller tests independent of hardware/ML.
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    class CurrentPoseFollower(SO101Follower):
        def configure(self):
            configure_at_current_pose(self)

    root = Path(__file__).resolve().parent / "calibration"
    leader = SO101Leader(SO101LeaderConfig(
        port=args.leader_port, id="my_leader", calibration_dir=root / "leader", use_degrees=True,
    ))
    follower = CurrentPoseFollower(SO101FollowerConfig(
        port=args.follower_port, id="my_follower", calibration_dir=root / "follower",
        use_degrees=True, max_relative_target=None,
    ))
    # max_relative_target is replaced by the controller's rate and measured-error bounds.
    if not leader.calibration or not follower.calibration:
        raise RuntimeError(f"Expected existing my_leader.json and my_follower.json under {root}")
    limits = calibrated_limits(follower)
    settings = Settings(**{key: getattr(args, key) for key in (
        "fps", "startup_speed", "follow_speed", "startup_gripper_speed", "follow_gripper_speed", "startup_timeout",
    )})
    try:
        print("Support the follower during connection; clear its travel path. Ctrl+C stops.", flush=True)
        print(f"Leader={args.leader_port}; follower={args.follower_port}; {args.fps:g} Hz", flush=True)
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise RuntimeError("Leader calibration mismatch. Recalibrate before teleoperation.")
        follower.connect(calibrate=False)
        observation = follower.get_observation()
        initial = {name: observation[name + ".pos"] for name in JOINTS}
        # perf_counter uses the high-resolution Windows performance counter.
        # Python 3.12's monotonic clock can repeat within a ~15.6ms tick.
        now = time.perf_counter()
        controller = SmoothController(initial, limits, settings, now)
        last_command_time = now
        status_time = now - 1
        print("ALIGN: keep the leader still while the follower approaches slowly.", flush=True)
        # Start the first movement after a real control period has elapsed.
        time.sleep(1 / settings.fps)
        while True:
            tick = time.perf_counter()
            observation = follower.get_observation()
            desired_action = leader.get_action()
            observed = {name: observation[name + ".pos"] for name in JOINTS}
            desired = {name: desired_action[name + ".pos"] for name in JOINTS}
            now = time.perf_counter()
            phase = controller.phase
            command = controller.step(observed, desired, now, now - last_command_time)
            follower.send_action({name + ".pos": command[name] for name in JOINTS})
            last_command_time = now
            if controller.phase != phase:
                print("FOLLOW: alignment confirmed; ramping up tracking speed over 1 second.", flush=True)
            if now - status_time >= 1:
                print(f"{controller.phase}: max angle error={controller.max_angle_error:.1f} deg; "
                      f"gripper error={controller.gripper_error:.1f} pts; "
                      f"range-clipped={','.join(controller.limited) or 'none'}", flush=True)
                status_time = now
            time.sleep(max(0, 1 / settings.fps - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print("\nStopping; support the follower as torque is released.", flush=True)
    finally:
        # Try the follower first, and always try both even if one cleanup fails.
        for device, name in ((follower, "follower"), (leader, "leader")):
            if device.bus.is_connected:
                try:
                    device.disconnect()
                except Exception as error:
                    print(f"CLEANUP FAILED ({name}): {error}. Support the arm and disconnect servo power.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
