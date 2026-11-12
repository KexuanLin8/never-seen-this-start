"""Open the cameras by device NAME, because Windows renumbers them after re-plugging or a reboot.

DirectShow lists video devices in the order OpenCV's CAP_DSHOW backend numbers them, so the position of a name in
that list is the camera index to use for this session.
"""

import av
import av.logging
from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

DESK = "C20-C28B-1.0"
WRIST = "Integrated Camera"


def video_devices():
    """Names of the DirectShow video devices, in index order."""
    av.logging.set_level(av.logging.INFO)
    with av.logging.Capture(True) as logs:
        try:
            av.open("video=dummy", format="dshow", options={"list_devices": "true"})
        except av.error.FFmpegError:
            pass  # expected: listing the devices makes FFmpeg stop with an exit status
    av.logging.set_level(None)
    messages = [m.strip() for _, _, m in logs]
    return [m.strip('"') for m, kind in zip(messages, messages[1:])
            if m.startswith('"') and kind == "(video"]


def index_of(name, devices=None):
    devices = video_devices() if devices is None else devices
    if devices.count(name) != 1:
        raise RuntimeError(f"Camera '{name}' must appear exactly once, found devices: {devices}. Check its USB cable.")
    return devices.index(name)


def open_cameras(fps=30):
    """{'desk': camera, 'wrist': camera}, not yet connected."""
    devices = video_devices()
    print("Video devices:", devices, flush=True)
    return {role: OpenCVCamera(OpenCVCameraConfig(index_or_path=index_of(name, devices), fps=fps, width=640, height=480,
                                                  fourcc="MJPG", backend=Cv2Backends.DSHOW))
            for role, name in (("desk", DESK), ("wrist", WRIST))}
