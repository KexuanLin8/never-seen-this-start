"""Prepare an Evo-1 fine-tuning run (FT-orig / FT-mid / FT-start) from the released LIBERO checkpoint.

1. Writes the RELEASED checkpoint's normalization statistics into meta/stats.json of every dataset used
   (an existing stats.json is kept as stats.json.orig), so fine-tuning keeps exactly the action/state
   scaling the checkpoint was trained with.
2. Writes an Evo-1 dataset config (data_groups -> franka_ee_pose_delta) and prints the training command.

  python prepare_evo1_finetune.py --variant start \
      --dataset libero_spatial_no_noops_lerobot=$W/data/libero_lerobot_full/libero_spatial_no_noops_1.0.0_lerobot \
      --dataset synth_start_libero_spatial=$W/data/synth_lerobot/start_libero_spatial  [...]

Run in the Evo1 env.
"""

import os
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def stats_from_checkpoint(ckpt):
    raw = json.loads((Path(ckpt) / "norm_stats.json").read_text())
    if "libero_robot" in raw:  # released (old) format: {arm: {"observation.state": {...}, "action": {...}}}
        src = raw["libero_robot"]
    else:  # new format: {arm: {dataset: {...}}} -> take the first dataset
        arm = next(iter(raw.values()))
        src = next(iter(arm.values()))
    out = {}
    for key in ("observation.state", "action"):
        lo, hi = np.array(src[key]["min"], float), np.array(src[key]["max"], float)
        # only min/max are used with normalization_type=bounds; the rest is filled for completeness
        out[key] = {"min": lo.tolist(), "max": hi.tolist(), "mean": ((lo + hi) / 2).tolist(),
                    "std": ((hi - lo) / 4).tolist(), "q01": lo.tolist(), "q99": hi.tolist()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="orig | mid | start (used in names)")
    p.add_argument("--ckpt", default=os.path.join(os.environ.get("W", "work"), "ckpt/Evo1_LIBERO"))
    p.add_argument("--dataset", action="append", required=True, help="name=path, repeat for every dataset")
    p.add_argument("--evo1-root", default=os.path.join(os.environ.get("W", "work"), "code/Evo-1/Evo_1"))
    p.add_argument("--work", default=os.path.join(os.environ.get("W", "work"), "ft"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ckpt-interval", type=int, default=0,
                   help="0 = only the final checkpoint (each DeepSpeed checkpoint with optimizer state is ~11 GB)")
    a = p.parse_args()
    ckpt_interval = a.ckpt_interval or a.steps + 1

    stats = stats_from_checkpoint(a.ckpt)
    groups = {}
    for spec in a.dataset:
        name, path = spec.split("=", 1)
        meta = Path(path) / "meta"
        if not (meta / "tasks.jsonl").exists():
            raise SystemExit(f"{path} is not a LeRobot v2.1 dataset (meta/tasks.jsonl missing)")
        target = meta / "stats.json"
        if target.exists() and not (meta / "stats.json.orig").exists():
            shutil.copy2(target, meta / "stats.json.orig")
        target.write_text(json.dumps(stats, indent=2))
        groups[name] = {"path": path,
                        "view_map": {"image_1": "observation.images.image", "image_2": "observation.images.wrist_image"},
                        "use_delta_action": False, "process_suite": "franka_ee_pose"}
        print(f"stats.json <- checkpoint stats: {path}")

    work = Path(a.work) / f"evo1_ft_{a.variant}"
    work.mkdir(parents=True, exist_ok=True)
    cfg = {"max_action_dim": 24, "max_state_dim": 24, "max_views": 3, "normalization_type": "bounds",
           "datasets_manifest": f"manifest_evo1_ft_{a.variant}.pkl", "decoder_cache_size": 16,
           "cache_shard_size_bytes": 268435456, "data_groups": {"franka_ee_pose_delta": groups}}
    cfg_path = work / "dataset_config.yaml"
    try:
        import yaml
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    except ImportError:  # JSON is valid YAML
        cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {cfg_path}")

    ckpt = Path(a.ckpt)
    # Evo-1's train.py always calls swanlab.init (interactive login in online mode) -> disable it
    cmd = (f"export SWANLAB_MODE=disabled WANDB_MODE=disabled && "
           f"cd {a.evo1_root} && accelerate launch --use_deepspeed --num_processes 1 --num_machines 1 "
           f"--deepspeed_config_file ds_config.json scripts/train.py --run_name evo1_ft_{a.variant} "
           f"--action_head flowmatching --use_augmentation --lr {a.lr} --dropout 0.2 --weight_decay 1e-3 "
           f"--batch_size {a.batch_size} --image_size 448 --max_steps {a.steps} --log_interval 10 "
           f"--ckpt_interval {ckpt_interval} --warmup_steps 500 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 "
           f"--finetune_vlm --finetune_action_head --disable_wandb --prefetch_factor 2 --video_backend av "
           f"--cache_dir {work}/cache --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path {cfg_path} "
           f"--per_action_dim 24 --state_dim 24 --save_dir {work}/checkpoints "
           f"--resume --resume_pretrain --resume_path {ckpt}")
    (work / "train_cmd.sh").write_text(cmd + "\n")
    print(f"\ntraining command (also saved to {work}/train_cmd.sh):\n{cmd}")


if __name__ == "__main__":
    main()
