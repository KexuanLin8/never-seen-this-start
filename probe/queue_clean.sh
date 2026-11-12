#!/usr/bin/env bash
# LIBERO-plus re-run with CLEANED instructions (perturbation-parameter suffixes stripped, --prompt-mode clean).
# Same episodes as P0_full / H0_full / H1_homing, so raw vs clean prompts can be compared pairwise.
#   nohup bash queue_clean.sh evo1        > .../logs/queue_clean_evo1.out 2>&1 &        (servers 9201 9202)
#   nohup bash queue_clean.sh vla_adapter > .../logs/queue_clean_vla_adapter.out 2>&1 & (servers 9301-9304)
#   nohup bash queue_clean.sh smolvla     > .../logs/queue_clean_smolvla.out 2>&1 &     (servers 9401-9404)
source "$(dirname "$0")/queue_lib.sh"
M=${1:?model: evo1 | vla_adapter | smolvla}
REF=$W/outputs/probe/homing_ref.json

run_cond() {  # run_cond <condition> [client args...]   (always LIBERO-plus, cleaned prompts)
  local cond=$1
  shift
  case $M in
    evo1)
      for i in 0 1 2 3; do
        launch libero_plus $M "$cond" $((9201 + i % 2)) $i 4 --render-res 448 --dummy-gripper 0 --horizon 15 \
          --prompt-mode clean "$@"
      done ;;
    smolvla)
      for i in $(seq 0 7); do
        launch libero_plus $M "$cond" $((9401 + i % 4)) $i 8 --render-res 256 --dummy-gripper -1 --horizon 0 \
          --prompt-mode clean "$@"
      done ;;
    vla_adapter)
      local suites=(libero_spatial libero_object libero_goal libero_10)
      for i in 0 1 2 3; do
        launch libero_plus $M "$cond" $((9301 + i)) 0 1 --render-res 256 --dummy-gripper -1 --horizon 8 \
          --prompt-mode clean --suites "${suites[$i]}" --file-tag "${suites[$i]}" "$@"
      done ;;
    *) echo "unknown model $M"; exit 1 ;;
  esac
}

round_start "H0c robot-init, clean prompt, no homing"; run_cond H0c_full --categories robot --frac 0.1; round_end
round_start "H1c robot-init, clean prompt, homing";    run_cond H1c_homing --categories robot --frac 0.1 --homing-ref "$REF"; round_end
round_start "P0c LIBERO-plus, clean prompt";           run_cond P0c_full --frac 0.05; round_end
echo "$(date '+%F %T') QUEUE clean $M DONE"
