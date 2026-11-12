#!/usr/bin/env bash
# VLA-Adapter probe queue. VLA-Adapter has one checkpoint per suite, so it needs FOUR probe servers:
#   9301 libero_spatial | 9302 libero_object | 9303 libero_goal | 9304 libero_10
# Every round runs one client per suite (4 clients), each against its own suite's server.
#   nohup bash queue_vla_adapter.sh > $W/outputs/probe/logs/queue_vla_adapter.out 2>&1 &
source "$(dirname "$0")/queue_lib.sh"
M=vla_adapter
VA="--render-res 256 --dummy-gripper -1 --horizon 8"
SUITES=(libero_spatial libero_object libero_goal libero_10)
REF=$W/outputs/probe/homing_ref.json

# per_suite <bench> <condition> [client args...]: one client per suite, each writing results_<suite>.jsonl
per_suite() {
  local bench=$1 cond=$2
  shift 2
  for i in 0 1 2 3; do
    launch "$bench" $M "$cond" $((9301 + i)) 0 1 $VA --suites "${SUITES[$i]}" --file-tag "${SUITES[$i]}" "$@"
  done
}

round_start "A clean reference with sensitivity branches"; per_suite libero C0_full --branches no_lang no_state no_agent no_wrist; round_end
round_start "B counterfactual language";  per_suite libero C1_lang_cf --prompt-mode counterfactual; round_end
round_start "C empty language";           per_suite libero C2_lang_empty --prompt-mode empty; round_end
round_start "D state zero";               per_suite libero C3_state_zero --state-mode zero; round_end
round_start "E mask agentview";           per_suite libero C6_mask_agent --mask-views 0; round_end
round_start "F mask wrist";               per_suite libero C7_mask_wrist --mask-views 1; round_end
round_start "G homing test";              per_suite libero_plus H0_full --categories robot --frac 0.1; round_end
round_start "H homing test";              per_suite libero_plus H1_homing --categories robot --frac 0.1 --homing-ref "$REF"; round_end
round_start "I LIBERO-plus reference";    per_suite libero_plus P0_full --frac 0.05; round_end

echo "$(date '+%F %T') QUEUE vla_adapter DONE"
