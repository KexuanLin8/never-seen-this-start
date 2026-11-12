#!/usr/bin/env bash
# Launch one probe condition as N background shards, round-robin over the probe servers in PROBE_PORTS.
# Activate the bench's conda env first (libero or libero_plus).
#
# usage: PROBE_PORTS="9201 9202" bash run_probe.sh <libero|libero_plus> <model_tag> <condition> <num_shards> [probe_client args...]
set -euo pipefail
BENCH=$1; MODEL=$2; COND=$3; N=$4; shift 4
W=${W:?set W to your work directory, see README}
read -r -a PORTS <<< "${PROBE_PORTS:?set PROBE_PORTS, e.g. PROBE_PORTS=\"9201 9202\"}"
LOG=$W/outputs/probe/logs
mkdir -p "$LOG"
for i in $(seq 0 $((N - 1))); do
  PORT=${PORTS[$((i % ${#PORTS[@]}))]}
  nohup python "$W/code/probe/probe_client.py" --bench "$BENCH" --model-tag "$MODEL" --condition "$COND" \
    --shard "$i" --num-shards "$N" --server-url "ws://127.0.0.1:$PORT" "$@" \
    > "$LOG/${BENCH}_${MODEL}_${COND}_s$i.out" 2>&1 &
done
sleep 8
echo "launched $N shards of $BENCH/$MODEL/$COND on ports ${PORTS[*]}"
tail -n 1 "$LOG/${BENCH}_${MODEL}_${COND}"_s*.out
