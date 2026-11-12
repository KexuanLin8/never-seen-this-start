"""Quality check of recorded episodes (no hardware needed).

  python check_recording.py data/trial        report per episode + outputs/recording_check/ep_XXXX.jpg

Reports the real control rate, whether frames stall, how far each recovery clip ends from the demonstration's first
state (they are spliced together), and how much the arm idles at the start and end of the demonstration.
"""

import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def frames(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def idle_steps(state, tol=1.0):
    """Leading and trailing steps during which no body joint moved more than tol degrees."""
    moved = np.abs(state[:, :5] - state[0, :5]).max(1) > tol
    lead = int(np.argmax(moved)) if moved.any() else len(state)
    moved = np.abs(state[:, :5] - state[-1, :5]).max(1) > tol
    trail = int(np.argmax(moved[::-1])) if moved.any() else len(state)
    return lead, trail


def repeated(frame_list):
    """Fraction of consecutive frames that are (almost) identical: a sign of a stalled camera."""
    small = [f[::8, ::8].astype(np.int16) for f in frame_list]
    return float(np.mean([np.abs(a - b).mean() < 0.5 for a, b in zip(small, small[1:])]))


def check(ep, out):
    meta = json.loads((ep / "meta.json").read_text())
    with np.load(ep / "demo.npz") as z:
        s, a, t = z["state"], z["action"], z["t"]
    desk, wrist = frames(ep / "demo_desk.mp4"), frames(ep / "demo_wrist.mp4")
    dt = np.diff(t)
    lead, trail = idle_steps(s)
    print(f"\n{ep.name}: task {meta['task_id']} dot {meta['dot']}, demo {len(s)} steps = {len(s) / 20:.1f} s")
    print(f"  rate {1 / dt.mean():.1f} Hz, longest gap {dt.max() * 1000:.0f} ms; video frames desk {len(desk)} wrist {len(wrist)}"
          f"; repeated frames desk {repeated(desk):.0%} wrist {repeated(wrist):.0%}")
    print(f"  idle at start {lead / 20:.1f} s, at end {trail / 20:.1f} s; gripper range {s[:, 5].min():.0f}..{s[:, 5].max():.0f}"
          f"; command-state lag (median) {np.median(np.abs(a[:, :5] - s[:, :5])):.1f} deg")
    tiles = [desk[0], desk[len(desk) // 2], desk[-1], wrist[len(wrist) // 2]]
    for k in range(len(meta["clips"])):
        with np.load(ep / f"rec{k}.npz") as z:
            rs = z["state"]
        gap = np.abs(rs[-1, :5] - s[0, :5])
        rd = frames(ep / f"rec{k}_desk.mp4")
        print(f"  clip {k} ({meta['clips'][k]['kind']}): {len(rs)} steps, start-to-end travel {np.abs(rs[0, :5] - rs[-1, :5]).max():.0f} deg, "
              f"end vs demo start: max {gap.max():.1f} deg ({JOINTS[int(gap.argmax())]})")
        tiles += [rd[0], rd[-1]]
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
    out.mkdir(parents=True, exist_ok=True)
    sheet = np.concatenate(rows, 0)[::2, ::2]
    cv2.imwrite(str(out / f"{ep.name}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main():
    root = HERE / (sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    out = HERE / "outputs" / "recording_check"
    for ep in sorted(root.glob("ep_*")):
        check(ep, out)
    print(f"\nsheets in {out}: row 1 = demo first / middle / last desk frame, middle wrist frame; then recovery clip first / last")


if __name__ == "__main__":
    main()
