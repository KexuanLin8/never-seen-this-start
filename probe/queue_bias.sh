#!/usr/bin/env bash
# State-bias dose response (clean LIBERO, 200 episodes per level): a constant error of 1 / 2 / 4 cm is
# added to the end-effector position in the state (random direction per episode, like a calibration error).
#   nohup bash queue_bias.sh evo1        > .../logs/queue_bias_evo1.out 2>&1 &        (servers 9201 9202)
#   nohup bash queue_bias.sh vla_adapter > .../logs/queue_bias_vla_adapter.out 2>&1 & (servers 9301-9304)
#   nohup bash queue_bias.sh smolvla     > .../logs/queue_bias_smolvla.out 2>&1 &     (servers 9401-9404)
source "$(dirname "$0")/queue_lib.sh"
M=${1:?model: evo1 | vla_adapter | smolvla}

run_cond() {  # run_cond <condition> [client args...]   (clean LIBERO)
  local cond=$1
  shift
  case $M in
    evo1)
      for i in 0 1 2 3; do
        launch libero $M "$cond" $((9201 + i % 2)) $i 4 --render-res 448 --dummy-gripper 0 --horizon 15 "$@"
      done ;;
    smolvla)
      for i in $(seq 0 7); do
        launch libero $M "$cond" $((9401 + i % 4)) $i 8 --render-res 256 --dummy-gripper -1 --horizon 0 "$@"
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

round_start "S1 state bias 1 cm"; run_cond S1_bias_1cm --state-mode bias --state-bias 0.01; round_end
round_start "S2 state bias 2 cm"; run_cond S2_bias_2cm --state-mode bias --state-bias 0.02; round_end
round_start "S3 state bias 4 cm"; run_cond S3_bias_4cm --state-mode bias --state-bias 0.04; round_end
echo "$(date '+%F %T') QUEUE bias $M DONE"
