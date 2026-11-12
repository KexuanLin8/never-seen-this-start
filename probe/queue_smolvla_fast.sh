#!/usr/bin/env bash
# Faster SmolVLA queue: FOUR probe servers (9401-9404) and EIGHT clients per round.
# Resumable: finished episodes are skipped, so it can replace queue_smolvla.sh + queue_disturb.sh smolvla
# at any point. Covers every SmolVLA condition (diagnosis + mid-episode pushes).
#   nohup bash queue_smolvla_fast.sh > $W/outputs/probe/logs/queue_smolvla_fast.out 2>&1 &
source "$(dirname "$0")/queue_lib.sh"
M=smolvla
SV="--render-res 256 --dummy-gripper -1 --horizon 0"
REF=$W/outputs/probe/homing_ref.json
N=8

# eight <bench> <condition> [client args...]: 8 shards over the 4 servers
eight() {
  local bench=$1 cond=$2
  shift 2
  for i in $(seq 0 $((N - 1))); do
    launch "$bench" $M "$cond" $((9401 + i % 4)) "$i" $N $SV "$@"
  done
}

round_start "C0 clean reference";      eight libero C0_full --branches no_lang no_state no_agent no_wrist --branch-every 10; round_end
round_start "C1 language cf";          eight libero C1_lang_cf --prompt-mode counterfactual; round_end
round_start "C2 language empty";       eight libero C2_lang_empty --prompt-mode empty; round_end
round_start "C3 state zero";           eight libero C3_state_zero --state-mode zero; round_end
round_start "C6 mask agentview";       eight libero C6_mask_agent --mask-views 0; round_end
round_start "C7 mask wrist";           eight libero C7_mask_wrist --mask-views 1; round_end
round_start "H0 homing test (no homing)"; eight libero_plus H0_full --categories robot --frac 0.1; round_end
round_start "H1 homing test (homing)"; eight libero_plus H1_homing --categories robot --frac 0.1 --homing-ref "$REF"; round_end
round_start "P0 LIBERO-plus reference"; eight libero_plus P0_full --frac 0.05; round_end
round_start "X1 push early mild";      eight libero X1_push_early_mild --disturb-at 40 --disturb-scale 0.2; round_end
round_start "X2 push early strong";    eight libero X2_push_early_strong --disturb-at 40 --disturb-scale 0.4; round_end
round_start "X3 push late strong";     eight libero X3_push_late_strong --disturb-at 120 --disturb-scale 0.4; round_end

echo "$(date '+%F %T') QUEUE smolvla_fast DONE"
