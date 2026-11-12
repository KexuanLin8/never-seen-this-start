#!/usr/bin/env bash
# SmolVLA probe queue (LeRobot checkpoint HuggingFaceVLA/smolvla_libero; probe servers --model lerobot on 9401 and 9402).
# The checkpoint replans every step (n_action_steps=1, --horizon 0 = server default), so sensitivity
# branches are computed only on every 10th query.
#   nohup bash queue_smolvla.sh > $W/outputs/probe/logs/queue_smolvla.out 2>&1 &
source "$(dirname "$0")/queue_lib.sh"
M=smolvla
SV="--render-res 256 --dummy-gripper -1 --horizon 0"
REF=$W/outputs/probe/homing_ref.json

round_start "A clean reference with sensitivity branches"
for i in 0 1 2 3; do
  launch libero $M C0_full $((9401 + i % 2)) $i 4 $SV --branches no_lang no_state no_agent no_wrist --branch-every 10
done
round_end

round_start "B language"
for i in 0 1; do launch libero $M C1_lang_cf    $((9401 + i)) $i 2 $SV --prompt-mode counterfactual; done
for i in 0 1; do launch libero $M C2_lang_empty $((9401 + i)) $i 2 $SV --prompt-mode empty; done
round_end

round_start "C state + agentview"
for i in 0 1; do launch libero $M C3_state_zero $((9401 + i)) $i 2 $SV --state-mode zero; done
for i in 0 1; do launch libero $M C6_mask_agent $((9401 + i)) $i 2 $SV --mask-views 0; done
round_end

round_start "D wrist"
for i in 0 1 2 3; do launch libero $M C7_mask_wrist $((9401 + i % 2)) $i 4 $SV --mask-views 1; done
round_end

round_start "E homing test (LIBERO-plus robot-init, paired)"
for i in 0 1; do launch libero_plus $M H0_full   $((9401 + i)) $i 2 $SV --categories robot --frac 0.1; done
for i in 0 1; do launch libero_plus $M H1_homing $((9401 + i)) $i 2 $SV --categories robot --frac 0.1 --homing-ref "$REF"; done
round_end

round_start "F LIBERO-plus reference"
for i in 0 1 2 3; do launch libero_plus $M P0_full $((9401 + i % 2)) $i 4 $SV --frac 0.05; done
round_end

echo "$(date '+%F %T') QUEUE smolvla DONE"
