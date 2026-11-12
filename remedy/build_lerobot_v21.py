"""Turn synth_libero.py staging episodes into a LeRobot v2.1 dataset (the layout Evo-1 trains on).

  python build_lerobot_v21.py --staging $W/data/synth_staging/start/libero_spatial \
      --template $W/data/libero_lerobot_parquet/libero_spatial_no_noops_1.0.0_lerobot \
      --out $W/data/synth_lerobot/start_libero_spatial

Layout written (same as IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot):
  data/chunk-XXX/episode_XXXXXX.parquet          observation.state, action, timestamp, frame_index,
                                                  episode_index, index, task_index
  videos/chunk-XXX/observation.images.image/episode_XXXXXX.mp4        (agentview, rotated 180 deg)
  videos/chunk-XXX/observation.images.wrist_image/episode_XXXXXX.mp4  (wrist, rotated 180 deg)
  meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl, meta/episodes_stats.jsonl
meta/stats.json (normalization) is written separately by prepare_evo1_finetune.py so that fine-tuning
reuses the released checkpoint's statistics. Also prints a synthesis summary from the staging logs.

Run in the Evo1 env (pandas + pyarrow).
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fix_synth_actions import convert  # raw LIBERO action -> IPEC/Evo-1 convention (gripper 1 open / 0 close)

CHUNK = 1000


def summarize(staging):
    rows = []
    for f in sorted(staging.glob("log_shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        print("no staging logs found")
        return
    kept = [r for r in rows if r.get("kept")]
    print(f"attempted={len(rows)} kept={len(kept)} ({100 * len(kept) / len(rows):.1f}%)")
    print("rejects:", dict(Counter(r.get("reject", "-") for r in rows if not r.get("kept"))))
    for key in ("magnitude", "joint_disp", "eef_disp", "recovery_len", "length"):
        v = [r[key] for r in kept if r.get(key) is not None]
        if v:
            print(f"  {key:<13} median={np.median(v):.3f}  p10={np.percentile(v, 10):.3f}  p90={np.percentile(v, 90):.3f}")


def episode_stats(arr):
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(), "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(), "count": [int(len(arr))]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--staging", required=True, help=".../synth_staging/<mode>/<suite>")
    p.add_argument("--template", required=True, help="an IPEC LeRobot LIBERO dataset folder (meta/info.json is copied)")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--copy", action="store_true", help="copy videos instead of moving them out of staging")
    a = p.parse_args()

    staging, out = Path(a.staging), Path(a.out)
    summarize(staging)
    eps = sorted(d for d in staging.iterdir() if (d / "meta.json").exists())
    if not eps:
        raise SystemExit("no kept episodes in staging")
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    (out / "meta").mkdir(parents=True)

    tasks, episodes, ep_stats, total = {}, [], [], 0
    for i, d in enumerate(eps):
        meta = json.loads((d / "meta.json").read_text())
        npz = np.load(d / "data.npz")
        state, action = npz["state"].astype(np.float32), convert(pd.Series(list(npz["action"])))
        T = len(action)
        tid = tasks.setdefault(meta["task"], len(tasks))
        chunk = i // CHUNK
        df = pd.DataFrame({
            "observation.state": list(state),
            "action": list(action),
            "timestamp": (np.arange(T) / a.fps).astype(np.float32),
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, i, dtype=np.int64),
            "index": np.arange(total, total + T, dtype=np.int64),
            "task_index": np.full(T, tid, dtype=np.int64),
        })
        pq = out / "data" / f"chunk-{chunk:03d}" / f"episode_{i:06d}.parquet"
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
        for src, key in (("agent.mp4", "observation.images.image"), ("wrist.mp4", "observation.images.wrist_image")):
            dst = out / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{i:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copy2 if a.copy else shutil.move)(str(d / src), str(dst))
        episodes.append({"episode_index": i, "tasks": [meta["task"]], "length": T, "source": d.name})
        ep_stats.append({"episode_index": i, "stats": {"observation.state": episode_stats(state),
                                                       "action": episode_stats(action)}})
        total += T

    info = json.loads((Path(a.template) / "meta" / "info.json").read_text())
    n = len(episodes)
    info.update(total_episodes=n, total_frames=total, total_tasks=len(tasks), total_videos=2 * n,
                total_chunks=(n - 1) // CHUNK + 1, chunks_size=CHUNK, fps=a.fps, splits={"train": f"0:{n}"})
    for key in ("observation.images.image", "observation.images.wrist_image"):
        if key in info.get("features", {}):
            info["features"][key].setdefault("info", {}).update({"video.codec": "h264", "video.fps": a.fps})
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": v, "task": k}) + "\n" for k, v in tasks.items()))
    (out / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (out / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats))
    (out / "meta" / "action_convention.json").write_text(json.dumps({"gripper": "1 open / 0 close", "arm_clip": 0.9375}))
    print(f"wrote {out}: {n} episodes, {total} frames, {len(tasks)} tasks")


if __name__ == "__main__":
    main()
