"""Offline tests of collect_data.py (no arm, no camera): python test_collect_data.py"""

import tempfile
from pathlib import Path

import av
import numpy as np

import collect_data as c

A = {j: 0.0 for j in c.JOINTS}
B = {j: 30.0 for j in c.JOINTS}

assert c.min_jerk(0) == 0 and abs(c.min_jerk(1) - 1) < 1e-9 and abs(c.min_jerk(0.5) - 0.5) < 1e-9
assert c.interpolate(A, B, 0) == A and c.interpolate(A, B, 1) == B and c.interpolate(A, B, 2) == B
assert c.path_duration(A, B, 30) == 1.0 and c.path_duration(A, {**B, "elbow_flex": 90.0}, 30) == 3.0
assert c.path_duration(A, {**A, "gripper": 100.0}, 30) == 1.0  # the gripper does not set the duration
assert abs(c.joint_distance_rad(A, {**A, "elbow_flex": 180 / np.pi}) - 1.0) < 1e-9
assert abs(c.joint_distance_rad(A, {**A, "gripper": 50.0})) == 0  # the gripper is not a joint angle

off = c.leader_offset({**A, "elbow_flex": 10.0, "gripper": 40.0}, {**A, "gripper": 50.0})
assert off["elbow_flex"] == -10.0 and off["shoulder_pan"] == 0.0 and off["gripper"] == 0.0



class FlakyLeader:
    def __init__(self, failures):
        self.failures = failures

    def get_action(self):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("motor 1: Parsing error")
        return {j + ".pos": 1.0 for j in c.JOINTS}


class Fake:
    pass


rig = Fake()
rig._read = c.Rig._read
rig.leader = FlakyLeader(2)
assert c.Rig.leader_pose(rig)["elbow_flex"] == 1.0  # two bad reads are survived
rig.leader = FlakyLeader(3)
try:
    c.Rig.leader_pose(rig)
    raise AssertionError("persistent failure must raise")
except ConnectionError:
    pass

seg = c.Segment()
rng = np.random.default_rng(0)
for i in range(25):
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    seg.add(A, B, img, img, i / c.FPS)
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    seg.save(d, "demo")
    with np.load(d / "demo.npz") as z:
        assert z["state"].shape == (25, 6) and z["action"].shape == (25, 6) and z["action"][0, 0] == 30
    for cam in ("desk", "wrist"):
        with av.open(str(d / f"demo_{cam}.mp4")) as v:
            frames = [f for f in v.decode(v.streams.video[0])]
        assert len(frames) == 25 and frames[0].width == 640 and frames[0].height == 480, len(frames)
print("all collect_data tests passed")
