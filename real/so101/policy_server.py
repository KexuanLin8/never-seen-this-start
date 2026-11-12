"""Serve a trained policy to eval_real.py over localhost. Runs in the GPU environment (lerobot-gpu, LeRobot 0.6.1).

The arm and the cameras stay in the tested lerobot-so101 environment; only the network runs here.

  python policy_server.py --ckpt models\\orig --port 6001
  python policy_server.py --ckpt models\\startchain --port 6002

Answers {"cmd": "info"} and {"desk", "wrist", "state", "task", "seed"} with the action chunk (T x 6, in the units of the
recorded data: degrees, gripper 0-100).
"""

import argparse
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

AUTH = b"so101-eval"
DESK_KEY, WRIST_KEY = "observation.images.camera1", "observation.images.camera2"


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="folder with config.json and model.safetensors")
    p.add_argument("--port", type=int, default=6001)
    a = p.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    if hasattr(cfg, "load_vlm_weights"):  # every weight is in the checkpoint: do not download the base VLM again
        cfg.load_vlm_weights = False
    policy = get_policy_class(cfg.type).from_pretrained(a.ckpt, config=cfg).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt)
    print(f"loaded {a.ckpt}: {cfg.type}, chunk {policy.config.chunk_size}, "
          f"GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)

    @torch.inference_mode()
    def predict(req):
        batch = {DESK_KEY: to_tensor(req["desk"]), WRIST_KEY: to_tensor(req["wrist"]),
                 "observation.state": torch.tensor(req["state"], dtype=torch.float32).reshape(1, -1), "task": [req["task"]]}
        batch = pre(batch)
        policy.reset()
        torch.manual_seed(req["seed"])
        chunk = policy.predict_action_chunk(batch)
        steps = [post(chunk[:, t]) for t in range(chunk.shape[1])]
        return torch.stack([s.reshape(-1) for s in steps]).float().cpu().numpy()

    with Listener(("127.0.0.1", a.port), authkey=AUTH) as listener:
        print(f"ready on port {a.port}", flush=True)
        while True:
            with listener.accept() as conn:
                print("client connected", flush=True)
                try:
                    while True:
                        req = conn.recv()
                        if req.get("cmd") == "info":
                            conn.send({"ckpt": Path(a.ckpt).name})
                            continue
                        t0 = time.perf_counter()
                        actions = predict(req)
                        conn.send({"actions": actions, "ms": (time.perf_counter() - t0) * 1000})
                except EOFError:
                    print("client disconnected", flush=True)


if __name__ == "__main__":
    main()
