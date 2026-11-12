#!/usr/bin/env bash
# Remedy evaluation for SmolVLA (same conditions as queue_remedy.sh for Evo-1). Starts and stops its own four
# LeRobot probe servers (ports 9401-9404) per model; clients resume finished episodes.
#   setsid nohup bash queue_remedy_smol.sh L smolvla orig startchain > $W/outputs/probe/logs/queue_smol_A.out 2>&1 < /dev/null &
# model tags: smolvla (released, already has C0/H0c/... from the diagnostics) and smolvla_ft_<v>
source "$(dirname "$0")/queue_lib.sh"
PHASE=${1:?phase: L | A | B | C}
shift
PY_LR=$W/conda/envs/lerobot/bin/python
SMOL="--render-res 256 --dummy-gripper -1 --horizon 0"
SERVERS=()
trap 'for p in "${SERVERS[@]}"; do kill "$p" 2>/dev/null; done' EXIT

stop_servers() {
  for p in "${SERVERS[@]}"; do kill "$p" 2>/dev/null; done
  for p in "${SERVERS[@]}"; do wait "$p" 2>/dev/null; done
  SERVERS=()
}

start_servers() {  # start_servers <tag> <ckpt>
  for port in 9401 9402 9403 9404; do
    local slog="$LOG/server_${1}_${port}.out"
    "$PY_LR" "$W/code/probe/probe_server.py" --model lerobot --ckpt "$2" --port "$port" > "$slog" 2>&1 &
    SERVERS+=("$!")
    for _ in $(seq 120); do
      grep -q "ready on" "$slog" && break
      kill -0 "${SERVERS[-1]}" 2>/dev/null || { echo "  !! server $port died:"; tail -n 5 "$slog"; return 1; }
      sleep 5
    done
    grep -q "ready on" "$slog" || { echo "  !! server $port not ready"; return 1; }
  done
  echo "$(date '+%F %T') servers ready for $1 ($2)"
}

cond() {  # cond <bench> <tag> <condition> [client args...]: 8 shards over the four servers
  local bench=$1 m=$2 c=$3
  shift 3
  round_start "$m $c"
  for i in $(seq 0 7); do launch "$bench" "$m" "$c" $((9401 + i % 4)) $i 8 $SMOL "$@"; done
  round_end
}

for v in "$@"; do
  if [ "$v" = smolvla ]; then
    tag=smolvla; ckpt=$W/ckpt/smolvla_libero
  else
    tag=smolvla_ft_$v; ckpt=$W/ft/smolvla_ft_$v/checkpoints/last/pretrained_model
    [ -f "$ckpt/config.json" ] || { echo "$(date '+%F %T') SKIP $tag: $ckpt missing"; continue; }
  fi
  start_servers "$tag" "$ckpt" || { stop_servers; continue; }
  if [ "$PHASE" = L ]; then  # lite set (SmolVLA re-plans every step: ~5x slower than Evo-1)
    cond libero "$tag" C0_full
    cond libero "$tag" T3_start_030 --start-perturb 0.3
    cond libero_plus "$tag" H0c_full --prompt-mode clean --categories robot --frac 0.1
    cond libero "$tag" K1_chain --chain-starts "$W/outputs/probe/chain_starts.json"
  elif [ "$PHASE" = A ]; then
    cond libero "$tag" C0_full
    cond libero "$tag" T1_start_010 --start-perturb 0.1
    cond libero "$tag" T2_start_020 --start-perturb 0.2
    cond libero "$tag" T3_start_030 --start-perturb 0.3
    cond libero "$tag" T4_start_050 --start-perturb 0.5
    cond libero_plus "$tag" H0c_full --prompt-mode clean --categories robot --frac 0.1
  elif [ "$PHASE" = C ]; then
    cond libero "$tag" K2_chain_half --chain-starts "$W/outputs/probe/chain_starts.json" --chain-frac 0.5
    cond libero "$tag" K1_chain --chain-starts "$W/outputs/probe/chain_starts.json"
  else
    cond libero "$tag" X1_push_early_mild   --disturb-at 40  --disturb-scale 0.2
    cond libero "$tag" X2_push_early_strong --disturb-at 40  --disturb-scale 0.4
    cond libero "$tag" X3_push_late_strong  --disturb-at 120 --disturb-scale 0.4
    cond libero "$tag" C3_state_zero --state-mode zero
  fi
  stop_servers
done
echo "$(date '+%F %T') QUEUE remedy-smol $PHASE DONE"
