#!/usr/bin/env bash
# Shared helpers for the probe queues. Each round launches client processes in the background
# and `wait`s for all of them, so rounds run strictly one after another.
W=${W:?set W to your work directory, see README}
PY_LIBERO=$W/conda/envs/libero/bin/python
PY_PLUS=$W/conda/envs/libero_plus/bin/python
CLIENT=$W/code/probe/probe_client.py
LOG=$W/outputs/probe/logs
mkdir -p "$LOG"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# launch <libero|libero_plus> <model_tag> <condition> <port> <shard> <num_shards> [client args...]
PIDS=()
launch() {
  local bench=$1 model=$2 cond=$3 port=$4 shard=$5 n=$6
  shift 6
  local logf="$LOG/${bench}_${model}_${cond}_s${shard}_p${port}.out"  # port in the name: unique per suite client
  if [ "$bench" = "libero_plus" ]; then
    env LIBERO_CONFIG_PATH="$HOME/.libero-plus" "$PY_PLUS" "$CLIENT" --bench "$bench" --model-tag "$model" \
      --condition "$cond" --shard "$shard" --num-shards "$n" --server-url "ws://127.0.0.1:$port" "$@" \
      > "$logf" 2>&1 &
  else
    env -u LIBERO_CONFIG_PATH "$PY_LIBERO" "$CLIENT" --bench "$bench" --model-tag "$model" \
      --condition "$cond" --shard "$shard" --num-shards "$n" --server-url "ws://127.0.0.1:$port" "$@" \
      > "$logf" 2>&1 &
  fi
  PIDS+=("$!:$logf")
}

round_start() { PIDS=(); echo "$(date '+%F %T') START $*"; }
# wait for every client of the round and report the ones that exited with an error
round_end() {
  local failed=0 item pid logf
  for item in "${PIDS[@]}"; do
    pid=${item%%:*}; logf=${item#*:}
    if ! wait "$pid"; then
      failed=$((failed + 1))
      echo "  !! client failed: $logf"; tail -n 3 "$logf" | sed 's/^/     /'
    fi
  done
  echo "$(date '+%F %T') END (failed clients: $failed)"
}
