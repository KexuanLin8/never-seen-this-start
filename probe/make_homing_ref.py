"""Build the nominal joint configuration per suite for the homing test.

Uses LIBERO-plus episodes from NON-robot categories (their robot initial state is unperturbed)
that logged init_joint_pos (P1 run PF_full, camera category). Prints the spread so we can check
that the nominal configuration is well defined (std should be ~0).

  python make_homing_ref.py --p1-root $W/outputs/p1 --out $W/outputs/probe/homing_ref.json
"""

import os
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1-root", default=os.path.join(os.environ.get("W", "work"), "outputs/p1"))
    p.add_argument("--out", default=os.path.join(os.environ.get("W", "work"), "outputs/probe/homing_ref.json"))
    a = p.parse_args()

    joints = defaultdict(list)
    for f in Path(a.p1_root).glob("libero_plus/*/results_shard*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["category"] != "robot" and r.get("init_joint_pos"):
                    joints[r["suite"]].append(r["init_joint_pos"])
    if not joints:
        raise SystemExit("no non-robot episodes with init_joint_pos found")
    ref = {}
    for suite, js in sorted(joints.items()):
        arr = np.array(js)
        ref[suite] = arr.mean(axis=0).tolist()
        print(f"{suite:<15} n={len(arr)}  max per-joint std={arr.std(axis=0).max():.4f} rad")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(ref, indent=2))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
