"""Offline control and startup sequencing tests. No serial ports are opened."""

from types import SimpleNamespace
import unittest

from teleoperate_smooth_start import (
    JOINTS, Settings, SmoothController, configure_at_current_pose,
)


def pose(value=0):
    return {name: value for name in JOINTS}


class ControllerTests(unittest.TestCase):
    def make_controller(self, settings=None):
        limits = {name: (-180, 180) for name in JOINTS}
        limits["gripper"] = (0, 100)
        return SmoothController(pose(), limits, settings or Settings(), 0)

    def test_large_start_offset_has_no_first_target_jump(self):
        c = self.make_controller()
        action = c.step(pose(), pose(100), 0.1, 0.1)
        self.assertAlmostEqual(action["shoulder_pan"], 1.5)
        self.assertAlmostEqual(action["gripper"], 2.5)

    def test_repeated_startup_timestamp_holds_then_resumes_at_limited_rate(self):
        c = self.make_controller()
        self.assertEqual(c.step(pose(), pose(100), 0, 0), pose())
        self.assertIsNone(c.close_since)
        self.assertEqual(c.phase, "ALIGN")
        action = c.step(pose(), pose(100), 1 / 30, 1 / 30)
        self.assertAlmostEqual(action["shoulder_pan"], 0.5)
        self.assertAlmostEqual(action["gripper"], 25 / 30)

    def test_zero_interval_does_not_complete_settle_window(self):
        c = self.make_controller()
        c.step(pose(), pose(), 0.1, 0.1)
        c.step(pose(), pose(), 1, 0)
        self.assertEqual(c.phase, "ALIGN")
        self.assertEqual(c.close_since, 0.1)

    def test_negative_and_nonfinite_intervals_stop(self):
        for dt in (-0.01, float("nan"), float("inf")):
            with self.subTest(dt=dt):
                c = self.make_controller()
                with self.assertRaisesRegex(RuntimeError, "Invalid control interval"):
                    c.step(pose(), pose(), 0.1, dt)

    def test_fast_mode_requires_actual_arrival(self):
        c = self.make_controller()
        for i in range(1, 101):
            action = c.step(pose(), pose(90), i * 0.1, 0.1)
            self.assertLessEqual(action["shoulder_pan"], 10)
        self.assertEqual(c.phase, "ALIGN")

    def test_settle_window_resets_after_moving_away(self):
        c = self.make_controller()
        c.step(pose(), pose(), 0.1, 0.1)
        c.step(pose(), pose(), 0.5, 0.1)
        c.step(pose(), pose(8), 0.6, 0.1)
        c.step(pose(), pose(), 0.7, 0.1)
        c.step(pose(), pose(), 1.2, 0.1)
        self.assertEqual(c.phase, "ALIGN")
        c.step(pose(), pose(), 1.4, 0.1)
        self.assertEqual(c.phase, "FOLLOW")

    def test_transition_ramps_then_allows_faster_tracking(self):
        c = self.make_controller()
        c.step(pose(), pose(), 0.1, 0.1)
        c.step(pose(), pose(), 0.8, 0.1)
        first = c.step(pose(), pose(100), 0.9, 0.1)
        self.assertGreater(first["shoulder_pan"], 1.5)
        self.assertLess(first["shoulder_pan"], 9)
        later = c.step(first, pose(100), 1.9, 0.1)
        self.assertAlmostEqual(later["shoulder_pan"] - first["shoulder_pan"], 9)

    def test_simple_lagging_plant_reaches_follow_mode(self):
        c = self.make_controller()
        observed = pose()
        for i in range(1, 251):
            command = c.step(observed, pose(60), i / 30, 1 / 30)
            observed = {name: observed[name] + 0.6 * (command[name] - observed[name]) for name in JOINTS}
        self.assertEqual(c.phase, "FOLLOW")
        self.assertLess(max(abs(observed[name] - 60) for name in JOINTS), 1)

    def test_stalled_arm_times_out_without_enabling_fast_mode(self):
        c = self.make_controller(Settings(startup_timeout=1))
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            c.step(pose(), pose(60), 1.1, 0.1)
        self.assertEqual(c.phase, "ALIGN")

    def test_nonfinite_input_and_stale_cycle_stop(self):
        c = self.make_controller()
        bad = pose()
        bad["elbow_flex"] = float("nan")
        with self.assertRaises(ValueError):
            c.step(pose(), bad, 0.1, 0.1)
        with self.assertRaisesRegex(RuntimeError, "gap"):
            c.step(pose(), pose(), 1, 0.5)

    def test_calibration_bounds_clip_out_of_range_leader(self):
        c = self.make_controller()
        for i in range(1, 101):
            observed = dict(c.command)
            c.step(observed, pose(-500), i * 0.1, 0.1)
        self.assertEqual(c.command["gripper"], 0)
        self.assertGreaterEqual(c.command["shoulder_pan"], -180)
        self.assertIn("shoulder_pan", c.limited)


class StartupTests(unittest.TestCase):
    def fake_robot(self, fail_write=False, mismatch=False):
        class Bus:
            def __init__(self):
                self.motors = dict.fromkeys(JOINTS)
                self.events = []
                self.positions = {name: 2000 + i for i, name in enumerate(JOINTS)}

            def disable_torque(self): self.events.append("torque_off")
            def configure_motors(self): self.events.append("configure")
            def set_operating_mode(self, mode, name): self.events.append(("mode", name))
            def sync_read(self, *args, **kwargs): return dict(self.positions)
            def write(self, register, name, value, **kwargs):
                if register == "Goal_Position":
                    if fail_write and name == "elbow_flex":
                        raise ConnectionError("simulated write failure")
                    self.events.append(("goal", name, value))
            def read(self, register, name, **kwargs): return self.positions[name]
            def enable_torque(self): self.events.append("torque_on")

        return SimpleNamespace(
            bus=Bus(), calibration={"exists": True}, is_calibrated=not mismatch,
            config=SimpleNamespace(position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32),
        )

    def test_all_current_targets_written_before_torque_on(self):
        robot = self.fake_robot()
        configure_at_current_pose(robot)
        events = robot.bus.events
        self.assertEqual(events[0], "torque_off")
        self.assertEqual(events[-1], "torque_on")
        goals = [event for event in events if isinstance(event, tuple) and event[0] == "goal"]
        self.assertEqual({event[1]: event[2] for event in goals}, robot.bus.positions)

    def test_write_failure_does_not_reenable_torque(self):
        robot = self.fake_robot(fail_write=True)
        with self.assertRaises(ConnectionError):
            configure_at_current_pose(robot)
        self.assertNotIn("torque_on", robot.bus.events)

    def test_calibration_mismatch_prevents_configuration(self):
        robot = self.fake_robot(mismatch=True)
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            configure_at_current_pose(robot)
        self.assertEqual(robot.bus.events, [])


if __name__ == "__main__":
    unittest.main()
