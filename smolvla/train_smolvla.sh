#!/usr/bin/env bash
# Fine-tune the released SmolVLA LIBERO checkpoint on one of the datasets built by build_smolvla_dataset.py.
# Same steps / batch size for every variant (equal compute); only the final checkpoint is kept.
#   bash train_smolvla.sh orig  > $W/ft/smolvla_ft_orig.log 2>&1       (dataset $W/data/smolvla/smol_orig)
#   bash train_smolvla.sh start > $W/ft/smolvla_ft_start.log 2>&1      (dataset $W/data/smolvla/smol_start)
#   DS=startchain bash train_smolvla.sh startchain_s2   (second run on an existing dataset)
# STEPS / BS / DS (dataset suffix, default: the variant) can be overridden from the environment. Run in the lerobot env (project_env.sh sourced).
set -uo pipefail
W=${W:?set W to your work directory, see README}
v=${1:?variant: orig | start | ...}
STEPS=${STEPS:-20000}
BS=${BS:-32}
SEED=${SEED:-1000}  # LeRobot training is deterministic: a repeated run needs a different seed
DS=${DS:-$v}
OUT=$W/ft/smolvla_ft_$v
if [ -d "$OUT/checkpoints/last/pretrained_model" ]; then echo "skip $v (already trained)"; exit 0; fi
if command -v lerobot-train >/dev/null; then TRAIN=(lerobot-train); else TRAIN=(python -m lerobot.scripts.lerobot_train); fi
echo "$(date '+%F %T') START smolvla $v ($STEPS steps, batch $BS)"
"${TRAIN[@]}" --policy.path="$W/ckpt/smolvla_libero" --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id="local/smol_$DS" --dataset.root="$W/data/smolvla/smol_$DS" --dataset.video_backend=pyav \
  --batch_size="$BS" --steps="$STEPS" --save_freq="$STEPS" --log_freq=100 --num_workers=8 \
  --output_dir="$OUT" --job_name="smolvla_ft_$v" --wandb.enable=false --seed="$SEED"
rc=$?
find "$OUT/checkpoints" -mindepth 1 -maxdepth 1 -type d ! -name last ! -name "$(printf '%06d' "$STEPS")" -exec rm -rf {} + 2>/dev/null
echo "$(date '+%F %T') END smolvla $v (exit $rc)  $(du -sh "$OUT" 2>/dev/null | cut -f1)"
exit $rc
