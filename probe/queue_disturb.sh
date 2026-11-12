#!/usr/bin/env bash
# Mid-episode disturbance queue (clean LIBERO, 200 episodes per condition) for one model.
# A push during execution cannot be undone by homing, so this tests whether the configuration-OOD
# failures seen at the episode start also appear mid-task.
#   nohup bash queue_disturb.sh evo1        > .../logs/queue_disturb_evo1.out 2>&1 &   (servers 9201 9202)
#   nohup bash queue_disturb.sh vla_adapter > .../logs/queue_disturb_vla_adapter.out 2>&1 &  (servers 9301-9304)
#   nohup bash queue_disturb.sh smolvla     > .../logs/queue_disturb_smolvla.out 2>&1 &   (servers 9401 9402)
source "$(dirname "$0")/queue_lib.sh"
M=${1:?model: evo1 | vla_adapter | smolvla}

# condition name -> push parameters (push after N env steps; scale = normalized OSC position action)
CONDS=(
  "X1_push_early_mild   --disturb-at 40  --disturb-scale 0.2"
  "X2_push_early_strong --disturb-at 40  --disturb-scale 0.4"
  "X3_push_late_strong  --disturb-at 120 --disturb-scale 0.4"
)

run_cond() {  # run_cond <condition> [push args...]
  local cond=$1
  shift
  case $M in
    evo1)
      for i in 0 1 2 3; do
        launch libero $M "$cond" $((9201 + i % 2)) $i 4 --render-res 448 --dummy-gripper 0 --horizon 15 "$@"
      done ;;
    smolvla)
      for i in 0 1 2 3; do
        launch libero $M "$cond" $((9401 + i % 2)) $i 4 --render-res 256 --dummy-gripper -1 --horizon 0 "$@"
      done ;;
    vla_adapter)
      local suites=(libero_spatial libero_object libero_goal libero_10)
      for i in 0 1 2 3; do
        launch libero $M "$cond" $((9301 + i)) 0 1 --render-res 256 --dummy-gripper -1 --horizon 8 \
          --suites "${suites[$i]}" --file-tag "${suites[$i]}" "$@"
      done ;;
    *) echo "unknown model $M"; exit 1 ;;
  esac
}

for spec in "${CONDS[@]}"; do
  read -r -a parts <<< "$spec"
  round_start "${parts[0]}"
  run_cond "${parts[@]}"
  round_end
done
echo "$(date '+%F %T') QUEUE disturb $M DONE"
