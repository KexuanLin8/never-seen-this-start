"""Compare a synthesized LeRobot dataset against the original IPEC dataset of the same suite, field by field.

FT-start / FT-mid collapsed (~7% even on clean LIBERO) while FT-orig did not, so the synthetic episodes
must disagree with the originals in some convention. This prints everything that could differ:
action / state ranges per dimension, gripper values, lengths, video size / fps / frame count, and whether
the first agent-view frame matches the original upright or rotated by 180 deg.

  python compare_datasets.py --suite libero_spatial --synth start

Run in the Evo1 env (pandas, pyarrow, av).
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

W = Path("$W")


def load_eps(root, n):
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))[:n]
    return [(f, pd.read_parquet(f)) for f in files]


def first_frame(root, ep_file, key):
    vid = root / "videos" / ep_file.parent.name / key / (ep_file.stem + ".mp4")
    with av.open(str(vid)) as c:
        s = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
        return frames[0], len(frames), float(s.average_rate or 0), s.codec_context.name


def corr(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def describe(name, root, eps):
    print(f"\n===== {name}: {root}")
    info = json.loads((root / "meta" / "info.json").read_text())
    print("info.json fps:", info.get("fps"), " features:",
          {k: v.get("shape") for k, v in info.get("features", {}).items()})
    act = np.concatenate([np.stack(df["action"].to_list()) for _, df in eps])
    st = np.concatenate([np.stack(df["observation.state"].to_list()) for _, df in eps])
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("action dim", act.shape[1], "\n  min ", act.min(0), "\n  max ", act.max(0), "\n  mean", act.mean(0))
    print("  gripper (last dim) unique values:", np.unique(np.round(act[:, -1], 3))[:10])
    print("state dim", st.shape[1], "\n  min ", st.min(0), "\n  max ", st.max(0))
    print("  first-frame state of episode 0:", np.array(eps[0][1]["observation.state"].iloc[0]))
    print("  first 3 actions of episode 0:\n", np.stack(eps[0][1]["action"].to_list()[:3]))
    print("lengths:", [len(df) for _, df in eps])
    for key in ("observation.images.image", "observation.images.wrist_image"):
        img, n, fps, codec = first_frame(root, eps[0][0], key)
        print(f"video {key}: shape={img.shape} frames={n} (parquet rows {len(eps[0][1])}) fps={fps} codec={codec}")
    return first_frame(root, eps[0][0], "observation.images.image")[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--synth", default="start", help="start | mid")
    p.add_argument("--n", type=int, default=5)
    a = p.parse_args()
    # take both paths from the dataset config the fine-tuning run actually used
    import yaml
    cfg = yaml.safe_load((W / "ft" / f"evo1_ft_{a.synth}" / "dataset_config.yaml").read_text())
    groups = next(iter(cfg["data_groups"].values()))
    paths = {k: Path(v["path"]) for k, v in groups.items() if a.suite in k}
    print("datasets of this suite in the fine-tuning config:", {k: str(v) for k, v in paths.items()})
    orig = next(v for k, v in paths.items() if "synth" not in k)
    syn = next(v for k, v in paths.items() if "synth" in k)
    img_o = describe("ORIGINAL", orig, load_eps(orig, a.n))
    img_s = describe("SYNTH", syn, load_eps(syn, a.n))
    if img_o.shape == img_s.shape:
        print(f"\nfirst agent frame correlation synth vs orig: as-is={corr(img_s, img_o):.3f}  "
              f"synth rotated 180={corr(img_s[::-1, ::-1], img_o):.3f}  (the higher one is the right orientation)")
    else:
        print(f"\n!! image shapes differ: orig {img_o.shape} synth {img_s.shape}")
    out = W / "outputs" / "check"
    out.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.imwrite(out / f"{a.suite}_orig.png", img_o)
        imageio.imwrite(out / f"{a.suite}_{a.synth}.png", img_s)
        print(f"saved first frames to {out}")
    except Exception as exc:
        print("could not save png:", exc)


if __name__ == "__main__":
    main()
