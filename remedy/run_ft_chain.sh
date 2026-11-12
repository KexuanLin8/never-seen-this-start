#!/usr/bin/env bash
# Train the Evo-1 fine-tuning variants one after another (start the chain from a shell where the Evo1 env
# is activated). After each run only the final model weights are kept: the DeepSpeed optimizer state
# (~9 GB), the noisy single-batch "step_best" checkpoint and the decoded-video cache are deleted.
#   nohup bash run_ft_chain.sh start orig mid > $W/ft/ft_chain.log 2>&1 &
set -uo pipefail
W=${W:?set W to your work directory, see README}
export SWANLAB_MODE=disabled WANDB_MODE=disabled
for v in "$@"; do
  d=$W/ft/evo1_ft_$v
  if [ -f "$d/checkpoints/step_final/mp_rank_00_model_states.pt" ]; then
    echo "$(date '+%F %T') skip $v (already trained)"
    continue
  fi
  echo "$(date '+%F %T') START $v"
  bash "$d/train_cmd.sh" > "$d/train.log" 2>&1
  rc=$?
  find "$d/checkpoints" -name "*optim_states.pt" -delete 2>/dev/null
  rm -rf "$d/checkpoints/step_best" "$d/checkpoints/wandb" "$d/cache"  # decoded-video cache: 1-5 GB, rebuilt if needed
  echo "$(date '+%F %T') END $v (exit $rc)  $(du -sh "$d/checkpoints" 2>/dev/null | cut -f1)"
done
echo "$(date '+%F %T') FT CHAIN DONE"
