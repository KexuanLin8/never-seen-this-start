"""Convert the actions of already-built synthetic LeRobot datasets to the IPEC/Evo-1 convention, in place.

compare_datasets.py showed the bug that collapsed FT-start / FT-mid: synth_libero.py recorded the raw LIBERO
gripper command (-1 open / +1 close) while the IPEC datasets Evo-1 trains on use 1 = open / 0 = close.
Arm dimensions are also clipped to +-0.9375 (the range of the original teleoperated actions; the
recovery controller saturated at +-1). States and images were checked to match and are not touched.

Idempotent: a converted dataset gets meta/action_convention.json and is skipped afterwards.

  python fix_synth_actions.py $W/data/synth_lerobot/*

Run in the Evo1 env (pandas + pyarrow).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARM_LIMIT = 0.9375


def convert(action):
    a = np.stack(action.to_list()).astype(np.float32)
    if not np.isin(a[:, 6], (-1.0, 1.0)).all():
        raise ValueError(f"gripper values not in {{-1, 1}}: {np.unique(a[:, 6])[:5]}")
    a[:, :6] = np.clip(a[:, :6], -ARM_LIMIT, ARM_LIMIT)
    a[:, 6] = (1.0 - a[:, 6]) / 2.0  # -1 open -> 1, +1 close -> 0
    return a


def episode_stats(arr):
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(), "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(), "count": [int(len(arr))]}


def fix(root):
    marker = root / "meta" / "action_convention.json"
    if marker.exists():
        print(f"skip (already converted): {root}")
        return
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    new_actions = {}
    for f in files:  # convert everything in memory first: either all files change or none
        new_actions[f] = convert(pd.read_parquet(f)["action"])
    stats = {}
    for f, a in new_actions.items():
        df = pd.read_parquet(f)
        df["action"] = list(a)
        df.to_parquet(f, index=False)
        stats[int(df["episode_index"].iloc[0])] = episode_stats(a)
    ep_path = root / "meta" / "episodes_stats.jsonl"
    rows = [json.loads(l) for l in ep_path.read_text().splitlines() if l.strip()]
    for r in rows:
        r["stats"]["action"] = stats[r["episode_index"]]
    ep_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    marker.write_text(json.dumps({"gripper": "1 open / 0 close", "arm_clip": ARM_LIMIT}))
    allg = np.concatenate([a[:, 6] for a in new_actions.values()])
    print(f"converted {root}: {len(files)} episodes, gripper values {np.unique(allg)}, open fraction {allg.mean():.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        fix(Path(p))
