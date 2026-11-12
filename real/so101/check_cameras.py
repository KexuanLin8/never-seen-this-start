"""Check both cameras together, the way recording will use them.

Opens the desk and the wrist camera at the same time, reads for a few seconds, reports the real frame rate of
each and saves one picture per camera to outputs/camera_check/. No arm is moved.

  python check_cameras.py

Cameras are found by name (camera_names.py), so USB re-plugging cannot swap them.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2

from camera_names import open_cameras

OUT = Path(__file__).parent / "outputs" / "camera_check"


def measure(name, cam, seconds, result):
    n, last, t0 = 0, None, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        frame = cam.async_read(timeout_ms=1000)
        if last is None or frame is not last:
            n += 1
            last = frame
    result[name] = (n / seconds, last)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=5)
    a = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cams = open_cameras(a.fps)
    for name, cam in cams.items():
        cam.connect()
        print(f"{name}: connected")
    result = {}
    threads = [threading.Thread(target=measure, args=(n, c, a.seconds, result)) for n, c in cams.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for name, cam in cams.items():
        fps, frame = result[name]
        path = OUT / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"{name}: {frame.shape[1]}x{frame.shape[0]}, {fps:.1f} frames/s -> {path}")
        cam.disconnect()


if __name__ == "__main__":
    main()
