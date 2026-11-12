"""Build the LeRobot datasets of the real-robot experiment from collect_data.py recordings.

  orig        the human demonstrations only
  startchain  the demonstrations + one synthetic episode per recovery clip (clip followed by the demonstration
              recorded right after it, in the same scene)

Both variants share the same human demonstrations. To keep the comparison fair they also share one normalisation:
build `orig` first, then pass --stats-from <orig dataset> when building `startchain`.

  python build_real_dataset.py --raw $W/data/real_raw --variant orig --out $W/data/real/real_orig
  python build_real_dataset.py --raw $W/data/real_raw --variant startchain --out $W/data/real/real_startchain \
      --stats-from $W/data/real/real_orig

Run in the lerobot env (writes with the installed LeRobot's dataset format).
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np

FPS = 20


def frames(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def load(folder, name):
    """(state, action, desk frames, wrist frames) of one recorded segment, all of equal length."""
    with np.load(folder / f"{name}.npz") as z:
        state, action = z["state"].astype(np.float32), z["action"].astype(np.float32)
    desk, wrist = frames(folder / f"{name}_desk.mp4"), frames(folder / f"{name}_wrist.mp4")
    if not len(state) == len(action) == len(desk) == len(wrist):
        raise SystemExit(f"{folder.name}/{name}: lengths differ {len(state)} {len(action)} {len(desk)} {len(wrist)}")
    return state, action, desk, wrist


def join(a, b):
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), a[2] + b[2], a[3] + b[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="folder with ep_XXXX recordings")
    p.add_argument("--variant", choices=["orig", "startchain"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats-from", default="", help="dataset whose state/action statistics are copied")
    p.add_argument("--desk-key", default="observation.images.camera1")
    p.add_argument("--wrist-key", default="observation.images.camera2")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all (small value: smoke test)")
    a = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    img = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]}
    features = {a.desk_key: img, a.wrist_key: img,
                "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
                "action": {"dtype": "float32", "shape": (6,), "names": None}}
    ds = LeRobotDataset.create(repo_id=f"local/{out.name}", fps=FPS, features=features, root=out,
                               robot_type="so101_follower", use_videos=True, image_writer_threads=4)
    episodes = sorted(e for e in Path(a.raw).glob("ep_*") if (e / "meta.json").exists())
    if a.max_episodes:
        episodes = episodes[: a.max_episodes]
    n_written = 0
    for ep in episodes:
        meta = json.loads((ep / "meta.json").read_text())
        demo = load(ep, "demo")
        seqs = [demo]
        if a.variant == "startchain":
            seqs += [join(load(ep, f"rec{k}"), demo) for k in range(len(meta["clips"]))]
        for state, action, desk, wrist in seqs:
            for t in range(len(state)):
                ds.add_frame({a.desk_key: desk[t], a.wrist_key: wrist[t], "observation.state": state[t],
                              "action": action[t], "task": meta["task"]})
            ds.save_episode()
            n_written += 1
        print(f"{ep.name}: total {n_written} episodes", flush=True)
    ds.finalize()
    if a.stats_from:
        src = json.loads((Path(a.stats_from) / "meta" / "stats.json").read_text())
        meta_path = out / "meta" / "stats.json"
        stats = json.loads(meta_path.read_text())
        for k in ("observation.state", "action"):
            stats[k] = src[k]
        meta_path.write_text(json.dumps(stats, indent=2))
        print(f"state/action statistics copied from {a.stats_from}")
    print(f"wrote {out}: {n_written} episodes from {len(episodes)} recordings")


if __name__ == "__main__":
    main()
