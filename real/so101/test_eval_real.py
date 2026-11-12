"""Offline tests of eval_real.py (no arm, no camera, no GPU): python test_eval_real.py"""

import threading
from multiprocessing.connection import Client, Listener

import collect_data as c
import eval_real as e

P = {j: 0.0 for j in c.JOINTS}
poses = {"home": {**P, "gripper": 50.0}, "test": [{**P, "elbow_flex": float(i)} for i in range(7)],
         "chain_test": {"A": [{**P, "wrist_flex": float(i)} for i in range(6)], "B": [{**P, "wrist_roll": float(i)} for i in range(6)]}}

for cond in ("nominal", "perturbed", "chain"):
    t = e.schedule(cond, poses)
    assert len(t) == 20 and t == e.schedule(cond, poses)  # the same for every model
    assert sum(x["task"] == "A" for x in t) == 10 and {x["position"] for x in t} == {1, 2, 3, 4, 5}
    assert all(x["pose"]["gripper"] == 50.0 for x in t)  # every trial starts with the gripper as at HOME
assert all(x["start"] == "home" for x in e.schedule("nominal", poses))
chain = e.schedule("chain", poses)
assert chain[0]["task"] == "A" and chain[0]["start"].startswith("endB") and chain[1]["start"].startswith("endA")

limits = {j: (-100.0, 100.0) for j in c.JOINTS}
out = e.limit_step({**P, "elbow_flex": 50.0, "gripper": 80.0, "shoulder_pan": -500.0}, P, limits)
assert out["elbow_flex"] == e.MAX_BODY_STEP and out["gripper"] == e.MAX_GRIPPER_STEP and out["shoulder_pan"] == -e.MAX_BODY_STEP
near = {**P, "elbow_flex": 99.0}
assert e.limit_step({**P, "elbow_flex": 200.0}, near, limits)["elbow_flex"] == 100.0

with Listener(("127.0.0.1", 6099), authkey=e.AUTH) as li:
    def serve():
        with li.accept() as conn:
            conn.send({"got": conn.recv()})
    th = threading.Thread(target=serve)
    th.start()
    with Client(("127.0.0.1", 6099), authkey=e.AUTH) as cl:
        cl.send({"cmd": "info"})
        assert cl.recv() == {"got": {"cmd": "info"}}
    th.join()
print("all eval_real tests passed")
