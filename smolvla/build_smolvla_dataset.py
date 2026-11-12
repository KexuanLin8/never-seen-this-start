"""Build a LeRobot dataset for SmolVLA fine-tuning from our LeRobot v2.1 (IPEC-format) datasets.

Conversions to the convention of the released SmolVLA LIBERO checkpoint (checked with inspect_smolvla.py):
  observation.images.wrist_image -> observation.images.image2 (images unchanged: same 256 px, same 180 deg rotation)
  observation.state              -> unchanged (eef pos + axis-angle + 2 gripper qpos)
  action gripper 1 open / 0 close -> -1 open / +1 close (raw LIBERO command); arm dimensions unchanged
The dataset is written with the INSTALLED LeRobot's writer (LeRobotDataset.create / add_frame / save_episode),
so it has whatever on-disk version that LeRobot trains on. Afterwards meta/stats.json is replaced by the
checkpoint's normalisation statistics, so fine-tuning keeps the released normalisation (as for Evo-1).

  python build_smolvla_dataset.py --out $W/data/smolvla/smol_orig  --src $W/data/libero_lerobot_full/*_no_noops_1.0.0_lerobot
  python build_smolvla_dataset.py --out $W/data/smolvla/smol_start --src <the same four> $W/data/synth_lerobot/start_*
  python build_smolvla_dataset.py --out $W/data/smolvla/smol_smoke --max-episodes 3 --src <one dataset>   (smoke test)

Run in the lerobot env.
"""

import os
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

KEYS = {"observation.images.image": "observation.images.image", "observation.images.wrist_image": "observation.images.image2"}


def frames(path):
    """Decoded frames; a damaged stream yields the frames before the damage (the caller decides what to keep)."""
    out = []
    try:
        with av.open(str(path)) as c:
            for f in c.decode(c.streams.video[0]):
                out.append(f.to_ndarray(format="rgb24"))
    except av.error.FFmpegError as exc:
        print(f"!! damaged video {path}: {len(out)} frames decoded ({type(exc).__name__})", flush=True)
    return out


def checkpoint_stats(ckpt):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    pre, _ = make_pre_post_processors(PreTrainedConfig.from_pretrained(ckpt), pretrained_path=ckpt)
    for step in getattr(pre, "steps", []):
        stats = getattr(step, "stats", None)
        if stats and "action" in stats:
            return {k: {sk: np.asarray(sv, dtype=float).tolist() for sk, sv in v.items()} for k, v in stats.items()}
    raise SystemExit("no normaliser statistics found in the checkpoint")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, help="LeRobot v2.1 dataset folders (IPEC format)")
    p.add_argument("--out", required=True)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--ckpt", default=os.path.join(os.environ.get("W", "work"), "ckpt/smolvla_libero"))
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-episodes", type=int, default=0, help="per source dataset; 0 = all (small value: smoke test)")
    a = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    img = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
    features = {"observation.images.image": img, "observation.images.image2": img,
                "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
                "action": {"dtype": "float32", "shape": (7,), "names": None}}
    ds = LeRobotDataset.create(repo_id=a.repo_id or f"local/{out.name}", fps=a.fps, features=features, root=out,
                               robot_type="panda", use_videos=True, image_writer_threads=4)
    n_ep = skipped = 0
    for src in map(Path, a.src):
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
                 for l in (src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()}
        files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
        if a.max_episodes:
            files = files[: a.max_episodes]
        for f in files:
            df = pd.read_parquet(f)
            vids = {dst: frames(src / "videos" / f.parent.name / key / f"{f.stem}.mp4") for key, dst in KEYS.items()}
            T = min(len(df), *(len(v) for v in vids.values()))
            if T < 0.9 * len(df):  # a demo cut short would teach an unfinished task
                print(f"!! skipped episode {f} ({T} of {len(df)} frames readable)", flush=True)
                skipped += 1
                continue
            act = np.stack(df["action"].to_list()).astype(np.float32)
            if not np.isin(act[:, 6], (0.0, 1.0)).all():
                raise SystemExit(f"{f}: gripper not in {{0, 1}} (unexpected convention)")
            act[:, 6] = 1.0 - 2.0 * act[:, 6]
            st = np.stack(df["observation.state"].to_list()).astype(np.float32)
            task = tasks[int(df["task_index"].iloc[0])]
            for t in range(T):
                frame = {"observation.images.image": vids["observation.images.image"][t],
                         "observation.images.image2": vids["observation.images.image2"][t],
                         "observation.state": st[t], "action": act[t], "task": task}
                try:
                    ds.add_frame(frame)
                except TypeError:  # older LeRobot: task passed separately
                    task_ = frame.pop("task")
                    ds.add_frame(frame, task=task_)
            ds.save_episode()
            n_ep += 1
            if n_ep % 100 == 0:
                print(f"{n_ep} episodes ({src.name})", flush=True)
    if hasattr(ds, "finalize"):
        ds.finalize()
    stats = checkpoint_stats(a.ckpt)
    meta = out / "meta" / "stats.json"
    old = json.loads(meta.read_text()) if meta.exists() else {}
    for k in ("observation.state", "action"):
        old[k] = {**old.get(k, {}), **stats[k]}
    meta.write_text(json.dumps(old, indent=2))
    print(f"wrote {out}: {n_ep} episodes ({skipped} skipped); meta/stats.json state/action <- checkpoint statistics")


if __name__ == "__main__":
    main()
