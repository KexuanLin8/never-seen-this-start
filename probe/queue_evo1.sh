#!/usr/bin/env bash
# Evo-1 probe queue (needs probe servers --model evo1 on ports 9201 and 9202, and homing_ref.json).
#   nohup bash queue_evo1.sh > $W/outputs/probe/logs/queue_evo1.out 2>&1 &
source "$(dirname "$0")/queue_lib.sh"
M=evo1
EVO="--render-res 448 --dummy-gripper 0 --horizon 15"
REF=$W/outputs/probe/homing_ref.json

round_start "A homing test (LIBERO-plus robot-init, paired)"
launch libero_plus $M H0_full   9201 0 2 $EVO --categories robot --frac 0.1
launch libero_plus $M H0_full   9202 1 2 $EVO --categories robot --frac 0.1
launch libero_plus $M H1_homing 9201 0 2 $EVO --categories robot --frac 0.1 --homing-ref "$REF"
launch libero_plus $M H1_homing 9202 1 2 $EVO --categories robot --frac 0.1 --homing-ref "$REF"
round_end

round_start "B clean reference with sensitivity branches"
for i in 0 1 2 3; do
  launch libero $M C0_full $((9201 + i % 2)) $i 4 $EVO --branches no_lang no_state no_agent no_wrist
done
round_end

round_start "C language"
for i in 0 1; do launch libero $M C1_lang_cf    $((9201 + i)) $i 2 $EVO --prompt-mode counterfactual; done
for i in 0 1; do launch libero $M C2_lang_empty $((9201 + i)) $i 2 $EVO --prompt-mode empty; done
round_end

round_start "D state + agentview"
for i in 0 1; do launch libero $M C3_state_zero $((9201 + i)) $i 2 $EVO --state-mode zero; done
for i in 0 1; do launch libero $M C6_mask_agent $((9201 + i)) $i 2 $EVO --mask-views 0; done
round_end

round_start "E wrist + LIBERO-plus reference"
for i in 0 1; do launch libero $M C7_mask_wrist $((9201 + i)) $i 2 $EVO --mask-views 1; done
for i in 0 1; do launch libero_plus $M P0_full $((9201 + i)) $i 2 $EVO --frac 0.05; done
round_end

echo "$(date '+%F %T') QUEUE evo1 DONE"
