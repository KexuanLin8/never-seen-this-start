#!/usr/bin/env bash
# Missing diagnostics of the paper: VLA-Adapter with cleaned LIBERO-plus instructions (H0c/H1c/P0c) and the
# state-bias sweep (S1-S3), SmolVLA state-bias sweep. Starts and stops its own probe servers (ports 9301-9304
# VLA-Adapter, one per suite; 9401-9404 SmolVLA), so nothing else may use the GPU while it runs.
#   setsid nohup bash run_diag_fill.sh > $W/outputs/probe/logs/diag_fill.out 2>&1 < /dev/null &
set -uo pipefail
W=${W:?set W to your work directory, see README}
CODE=$W/code/probe
LOG=$W/outputs/probe/logs
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
SERVERS=()
trap 'for p in "${SERVERS[@]}"; do kill "$p" 2>/dev/null; done' EXIT

wait_ready() {  # wait_ready <log> <pid>
  for _ in $(seq 180); do
    grep -q "ready on" "$1" && return 0
    kill -0 "$2" 2>/dev/null || { echo "!! server died: $1"; tail -n 5 "$1"; return 1; }
    sleep 5
  done
  echo "!! server not ready after 15 min: $1"; return 1
}

stop_servers() {
  for p in "${SERVERS[@]}"; do kill "$p" 2>/dev/null; done
  for p in "${SERVERS[@]}"; do wait "$p" 2>/dev/null; done
  SERVERS=()
}

# ---- VLA-Adapter: one server per suite (per-suite Pro checkpoints) ----
declare -A VA_CKPT=([libero_spatial]=Spatial [libero_object]=Object [libero_goal]=Goal [libero_10]=Long)
port=9301
for suite in libero_spatial libero_object libero_goal libero_10; do
  ckpt=$(ls -d "$W/ckpt/vla_adapter/"*"${VA_CKPT[$suite]}"* 2>/dev/null | head -n 1)
  [ -n "$ckpt" ] || { echo "!! no VLA-Adapter checkpoint matching ${VA_CKPT[$suite]} in $W/ckpt/vla_adapter"; exit 1; }
  slog=$LOG/server_va_$port.out
  "$W/conda/envs/vla_adapter/bin/python" "$CODE/probe_server.py" --model vla_adapter --ckpt "$ckpt" --suite "$suite" \
    --port $port > "$slog" 2>&1 &
  SERVERS+=("$!")
  wait_ready "$slog" "${SERVERS[-1]}" || exit 1
  echo "$(date '+%F %T') VLA-Adapter $suite ready on $port ($ckpt)"
  port=$((port + 1))
done
bash "$CODE/queue_clean.sh" vla_adapter
bash "$CODE/queue_bias.sh" vla_adapter
stop_servers

# ---- SmolVLA: four servers ----
for port in 9401 9402 9403 9404; do
  slog=$LOG/server_smolvla_$port.out
  "$W/conda/envs/lerobot/bin/python" "$CODE/probe_server.py" --model lerobot --ckpt "$W/ckpt/smolvla_libero" \
    --port $port > "$slog" 2>&1 &
  SERVERS+=("$!")
  wait_ready "$slog" "${SERVERS[-1]}" || exit 1
done
echo "$(date '+%F %T') SmolVLA servers ready"
bash "$CODE/queue_bias.sh" smolvla
stop_servers
echo "$(date '+%F %T') DIAG FILL DONE"
