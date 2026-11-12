#!/usr/bin/env bash
# Remedy evaluation: released Evo-1 vs the fine-tuned variants (FT-orig / FT-mid / FT-start).
# Starts and stops its own four Evo-1 probe servers (ports 9201-9204) for every model, so nothing else may
# use the GPU or these ports while it runs. Clients resume finished episodes, so re-running is safe.
#
#   phase A (core: does the remedy work?)  C0, start perturbation 0.1/0.2/0.3/0.5 rad, LIBERO-plus robot-init
#   phase B (side effects / baselines)     mid-episode pushes X1-X3, state zero, state bias 1/2/4 cm, LIBERO-plus all
#   phase C (task chaining)                start where another task of the suite ended (needs chain_starts.json)
#   phase H (held-out source tasks)        chaining from end poses of tasks seen / not seen in synthesis (split_chain_pool.py)
#
#   setsid nohup bash queue_remedy.sh A orig mid start evo1 > $W/outputs/probe/logs/queue_remedy_A.out 2>&1 < /dev/null &
#   setsid nohup bash queue_remedy.sh B orig mid start evo1 > $W/outputs/probe/logs/queue_remedy_B.out 2>&1 < /dev/null &
source "$(dirname "$0")/queue_lib.sh"
PHASE=${1:?phase: A | B | C | H}
shift
[ $# -gt 0 ] || { echo "usage: queue_remedy.sh <A|B> <orig|mid|start|evo1>..."; exit 1; }
PY_EVO=$W/conda/envs/Evo1/bin/python
EVO="--render-res 448 --dummy-gripper 0 --horizon 15"
FT_KEYS="--arm-key franka_ee_pose_delta --dataset-key libero_spatial_no_noops_lerobot"
SERVERS=()
trap 'stop_servers' EXIT

stop_servers() {
  for pid in "${SERVERS[@]}"; do kill "$pid" 2>/dev/null; done
  for pid in "${SERVERS[@]}"; do wait "$pid" 2>/dev/null; done
  SERVERS=()
}

# start_servers <model_tag> <ckpt> [server args...]: four servers, returns once all report ready
start_servers() {
  local tag=$1 ckpt=$2
  shift 2
  for port in 9201 9202 9203 9204; do
    local slog="$LOG/server_${tag}_${port}.out"
    "$PY_EVO" "$W/code/probe/probe_server.py" --model evo1 --ckpt "$ckpt" --port "$port" "$@" > "$slog" 2>&1 &
    SERVERS+=("$!")
    for _ in $(seq 120); do  # loading takes ~1 min; give up after 10 min
      grep -q "ready on" "$slog" && break
      kill -0 "${SERVERS[-1]}" 2>/dev/null || { echo "  !! server $port died:"; tail -n 5 "$slog"; return 1; }
      sleep 5
    done
    grep -q "ready on" "$slog" || { echo "  !! server $port not ready after 10 min"; return 1; }
  done
  echo "$(date '+%F %T') servers ready for $tag ($ckpt)"
}

cond() {  # cond <bench> <model_tag> <condition> [client args...]: 8 shards over the four servers
  local bench=$1 m=$2 c=$3
  shift 3
  round_start "$m $c"
  for i in $(seq 0 7); do launch "$bench" "$m" "$c" $((9201 + i % 4)) $i 8 $EVO "$@"; done
  round_end
}

for v in "$@"; do
  if [ "$v" = evo1 ]; then
    tag=evo1; ckpt=$W/ckpt/Evo1_LIBERO; sargs=()
  else
    tag=evo1_ft_$v; ckpt=$W/ft/evo1_ft_$v/checkpoints/step_final; sargs=($FT_KEYS)
    if [ ! -f "$ckpt/mp_rank_00_model_states.pt" ]; then
      echo "$(date '+%F %T') SKIP $tag: $ckpt/mp_rank_00_model_states.pt missing (not trained yet?)"
      continue
    fi
    # the normalizer keys passed to the server must exist in the checkpoint's norm_stats.json
    if ! "$PY_EVO" -c "import json,sys; s=json.load(open(sys.argv[1])); assert sys.argv[3] in s[sys.argv[2]], list(s)" \
         "$ckpt/norm_stats.json" franka_ee_pose_delta libero_spatial_no_noops_lerobot; then
      echo "$(date '+%F %T') SKIP $tag: normalizer keys not found in $ckpt/norm_stats.json"
      continue
    fi
  fi
  start_servers "$tag" "$ckpt" "${sargs[@]}" || { stop_servers; continue; }

  if [ "$PHASE" = A ]; then
    cond libero "$tag" C0_full
    cond libero "$tag" T1_start_010 --start-perturb 0.1
    cond libero "$tag" T2_start_020 --start-perturb 0.2
    cond libero "$tag" T3_start_030 --start-perturb 0.3
    cond libero "$tag" T4_start_050 --start-perturb 0.5
    cond libero_plus "$tag" H0c_full --prompt-mode clean --categories robot --frac 0.1
  elif [ "$PHASE" = H ]; then
    cond libero "$tag" K3_chain_unseen --chain-starts "$W/outputs/probe/chain_starts_odd.json"
    cond libero "$tag" K4_chain_seen --chain-starts "$W/outputs/probe/chain_starts_even.json"
  elif [ "$PHASE" = C ]; then
    cond libero "$tag" K2_chain_half --chain-starts "$W/outputs/probe/chain_starts.json" --chain-frac 0.5
    cond libero "$tag" K1_chain --chain-starts "$W/outputs/probe/chain_starts.json"
  else
    cond libero "$tag" X1_push_early_mild   --disturb-at 40  --disturb-scale 0.2
    cond libero "$tag" X2_push_early_strong --disturb-at 40  --disturb-scale 0.4
    cond libero "$tag" X3_push_late_strong  --disturb-at 120 --disturb-scale 0.4
    cond libero "$tag" C3_state_zero --state-mode zero
    cond libero "$tag" S1_bias_1cm --state-mode bias --state-bias 0.01
    cond libero "$tag" S2_bias_2cm --state-mode bias --state-bias 0.02
    cond libero "$tag" S3_bias_4cm --state-mode bias --state-bias 0.04
    cond libero_plus "$tag" P0c_full --prompt-mode clean --frac 0.05
  fi
  stop_servers
done
echo "$(date '+%F %T') QUEUE remedy $PHASE DONE"
