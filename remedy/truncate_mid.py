"""Build the corrected DART-style baseline datasets (midv2_<suite>) from the existing mid_<suite> datasets.

Problem with the first mid datasets: the push itself is not recorded, so the recorded episode jumps from the
pre-push frame to the displaced arm. Evo-1 predicts 50-step action chunks, so every frame in the 50 steps
before the push is labelled with recovery actions that cannot be predicted from its observation (label noise);
FT-mid lost 21 points on clean LIBERO. midv2 keeps each episode only from the first recovery frame on, i.e.
"start mid-task from a displaced arm, recover, finish" -- consistent observations and labels, the same
structure as the start-diversified episodes.

  python truncate_mid.py            (all four suites; writes $W/data/synth_lerobot/midv2_<suite>)

Needs the staging folders (data.npz with the per-frame phase). Run in the Evo1 env (pandas, pyarrow, av).
"""

import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd

W = Path("$W")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
FPS = 20


def cut_video(src, dst, k):
    with av.open(str(src)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    frames = frames[k:]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(dst), "w") as out:
        s = out.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return len(frames)


def episode_stats(arr):
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(), "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(), "count": [int(len(arr))]}


def build(suite):
    src = W / "data" / "synth_lerobot" / f"mid_{suite}"
    dst = W / "data" / "synth_lerobot" / f"midv2_{suite}"
    staging = W / "data" / "synth_staging" / "mid" / suite
    if dst.exists():
        print(f"skip (exists): {dst}")
        return
    (dst / "meta").mkdir(parents=True)
    for name in ("tasks.jsonl", "stats.json", "action_convention.json"):
        if (src / "meta" / name).exists():
            shutil.copy2(src / "meta" / name, dst / "meta" / name)
    eps = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    new_eps, new_stats, total, cut = [], [], 0, []
    for e in eps:
        i = e["episode_index"]
        chunk = f"chunk-{i // 1000:03d}"
        phase = np.load(staging / e["source"] / "data.npz")["phase"]
        df = pd.read_parquet(src / "data" / chunk / f"episode_{i:06d}.parquet")
        if len(df) != len(phase):
            raise SystemExit(f"length mismatch in {e['source']}: parquet {len(df)} vs phase {len(phase)}")
        k = int(np.argmax(phase == "recovery"))
        df = df.iloc[k:].reset_index(drop=True)
        T = len(df)
        df["frame_index"] = np.arange(T, dtype=np.int64)
        df["timestamp"] = (np.arange(T) / FPS).astype(np.float32)
        df["index"] = np.arange(total, total + T, dtype=np.int64)
        out = dst / "data" / chunk / f"episode_{i:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        for key in ("observation.images.image", "observation.images.wrist_image"):
            n = cut_video(src / "videos" / chunk / key / f"episode_{i:06d}.mp4",
                          dst / "videos" / chunk / key / f"episode_{i:06d}.mp4", k)
            if n != T:
                raise SystemExit(f"video/parquet length mismatch in episode {i}: {n} vs {T}")
        new_eps.append({**e, "length": T})
        st = np.stack(df["observation.state"].to_list())
        ac = np.stack(df["action"].to_list())
        new_stats.append({"episode_index": i, "stats": {"observation.state": episode_stats(st),
                                                         "action": episode_stats(ac)}})
        total += T
        cut.append(k)
    info = json.loads((src / "meta" / "info.json").read_text())
    info.update(total_frames=total)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (dst / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in new_eps))
    (dst / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in new_stats))
    print(f"{dst}: {len(new_eps)} episodes, {total} frames (dropped a median of {int(np.median(cut))} "
          f"pre-push frames per episode)", flush=True)


if __name__ == "__main__":
    for s in SUITES:
        build(s)
