"""Print what the SmolVLA fine-tuning pipeline must match: LeRobot version, the checkpoint's input/output
features, normalisation statistics (incl. the gripper range of the action), and the training entry point.

  $W/conda/envs/lerobot/bin/python inspect_smolvla.py [--ckpt $W/ckpt/smolvla_libero]
"""

import os
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(os.environ.get("W", "work"), "ckpt/smolvla_libero"))
    a = p.parse_args()

    import lerobot
    print("lerobot", getattr(lerobot, "__version__", "?"), Path(lerobot.__file__).parent)
    try:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
        print("dataset CODEBASE_VERSION", CODEBASE_VERSION)
        import inspect
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        for fn in ("create", "add_frame", "save_episode", "finalize"):
            f = getattr(LeRobotDataset, fn, None)
            print(f"  LeRobotDataset.{fn}{inspect.signature(f) if f else ': MISSING'}")
    except ImportError as exc:  # the dataset extra is not installed in this env
        print("dataset module NOT importable:", str(exc).splitlines()[0])
    for mod in ("lerobot.scripts.lerobot_train", "lerobot.scripts.train"):
        print(f"train entry {mod}: {'yes' if importlib.util.find_spec(mod) else 'no'}")

    cfg = json.loads((Path(a.ckpt) / "config.json").read_text())
    print("policy type", cfg.get("type"), " chunk_size", cfg.get("chunk_size"), " n_action_steps", cfg.get("n_action_steps"))
    print("input_features", json.dumps(cfg.get("input_features"), indent=None))
    print("output_features", json.dumps(cfg.get("output_features"), indent=None))
    print("normalization_mapping", cfg.get("normalization_mapping"))
    print("dtype", cfg.get("dtype"), " num_inference_steps", cfg.get("num_inference_steps"),
          " tokenizer", cfg.get("text_tokenizer_name"))
    print("files", sorted(x.name for x in Path(a.ckpt).iterdir()))

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    pcfg = PreTrainedConfig.from_pretrained(a.ckpt)
    pre, post = make_pre_post_processors(pcfg, pretrained_path=a.ckpt)
    for name, pipe in (("pre", pre), ("post", post)):
        for step in getattr(pipe, "steps", []):
            stats = getattr(step, "stats", None)
            if not stats:
                continue
            print(f"[{name}] {type(step).__name__}")
            for k, v in stats.items():
                d = {sk: np.round(np.asarray(sv, dtype=float).reshape(-1), 3).tolist()[:8] for sk, sv in v.items()
                     if sk in ("mean", "std", "min", "max")}
                print(f"   {k}: {d}")


if __name__ == "__main__":
    main()
