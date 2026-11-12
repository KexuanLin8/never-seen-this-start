#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on the real-robot datasets; same steps and batch size for every variant.
#   bash train_real.sh orig | startchain                 (dataset $W/data/real/real_<variant>)
#   DS=orig bash train_real.sh orig_s2                   (second run on an existing dataset)
#   STEPS=50 BS=4 DS=orig bash train_real.sh smoke       (short test of the whole pipeline)
# Our dataset has two cameras (camera1 desk, camera2 wrist); the base model's third camera input stays unused.
# Run with the lerobot env on PATH. Only the final checkpoint is kept.
set -uo pipefail
W=${W:?set W to your work directory, see README}
v=${1:?variant: orig | startchain | ...}
STEPS=${STEPS:-20000}
BS=${BS:-32}
SEED=${SEED:-1000}  # LeRobot training is deterministic: a repeated run needs a different seed
DS=${DS:-$v}
OUT=$W/ft_real/smol_$v
if [ -d "$OUT/checkpoints/last/pretrained_model" ]; then echo "skip $v (already trained)"; exit 0; fi
mkdir -p "$W/ft_real"
echo "$(date '+%F %T') START real $v (dataset real_$DS, $STEPS steps, batch $BS)"
lerobot-train --policy.path="$W/ckpt/smolvla_base" --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id="local/real_$DS" --dataset.root="$W/data/real/real_$DS" --dataset.video_backend=pyav \
  --batch_size="$BS" --steps="$STEPS" --save_freq="$STEPS" --log_freq=100 --num_workers=8 \
  --output_dir="$OUT" --job_name="real_$v" --wandb.enable=false --seed="$SEED"
rc=$?
find "$OUT/checkpoints" -mindepth 1 -maxdepth 1 -type d ! -name last ! -name "$(printf '%06d' "$STEPS")" -exec rm -rf {} + 2>/dev/null
echo "$(date '+%F %T') END real $v (exit $rc)  $(du -sh "$OUT" 2>/dev/null | cut -f1)"
exit $rc
