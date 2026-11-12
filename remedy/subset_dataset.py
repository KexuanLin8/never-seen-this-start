"""Keep every k-th episode of a LeRobot v2.1 dataset (for the synthetic-data ratio ablation).

  python subset_dataset.py --src $W/data/synth_lerobot/start_libero_spatial \
                           --dst $W/data/synth_lerobot/starthalf_libero_spatial --every 2
  python subset_dataset.py --all-start --every 2      (the four start_<suite> datasets -> starthalf_<suite>)

Videos are copied unchanged; episode and frame indices are renumbered. Run in the Evo1 env.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

W = Path("$W")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
KEYS = ("observation.images.image", "observation.images.wrist_image")


def subset(src, dst, every):
    if dst.exists():
        print(f"skip (exists): {dst}")
        return
    (dst / "meta").mkdir(parents=True)
    for name in ("tasks.jsonl", "stats.json", "action_convention.json"):
        if (src / "meta" / name).exists():
            shutil.copy2(src / "meta" / name, dst / "meta" / name)
    eps = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    st = {json.loads(l)["episode_index"]: json.loads(l)
          for l in (src / "meta" / "episodes_stats.jsonl").read_text().splitlines() if l.strip()}
    keep = eps[::every]
    new_eps, new_st, total = [], [], 0
    for j, e in enumerate(keep):
        i = e["episode_index"]
        oc, nc = f"chunk-{i // 1000:03d}", f"chunk-{j // 1000:03d}"
        df = pd.read_parquet(src / "data" / oc / f"episode_{i:06d}.parquet")
        df["episode_index"] = j
        df["index"] = range(total, total + len(df))
        out = dst / "data" / nc / f"episode_{j:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        for key in KEYS:
            v = dst / "videos" / nc / key / f"episode_{j:06d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "videos" / oc / key / f"episode_{i:06d}.mp4", v)
        new_eps.append({**e, "episode_index": j})
        new_st.append({**st[i], "episode_index": j})
        total += len(df)
    info = json.loads((src / "meta" / "info.json").read_text())
    n = len(new_eps)
    info.update(total_episodes=n, total_frames=total, total_videos=2 * n, total_chunks=(n - 1) // 1000 + 1,
                splits={"train": f"0:{n}"})
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (dst / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in new_eps))
    (dst / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in new_st))
    print(f"{dst}: {n} of {len(eps)} episodes, {total} frames")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src")
    p.add_argument("--dst")
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--all-start", action="store_true")
    a = p.parse_args()
    if a.all_start:
        for s in SUITES:
            subset(W / "data" / "synth_lerobot" / f"start_{s}", W / "data" / "synth_lerobot" / f"starthalf_{s}", a.every)
    else:
        subset(Path(a.src), Path(a.dst), a.every)


if __name__ == "__main__":
    main()
