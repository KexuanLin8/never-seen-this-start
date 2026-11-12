#!/usr/bin/env bash
# Launch one P1 condition as N background shards, spread round-robin over the P1 servers.
# Activate the bench's conda env first (libero or libero_plus).
#
# usage: bash run_p1.sh <libero|libero_plus> <condition> <num_shards> [extra p1_client.py args...]
#   bash run_p1.sh libero C0_full 4 --branches no_lang no_state no_agent no_wrist --attn-first
#   P1_PORTS="9100" bash run_p1.sh libero C1_lang_cf 2 --prompt-mode counterfactual
set -euo pipefail
BENCH=$1; COND=$2; N=$3; shift 3
W=${W:?set W to your work directory, see README}
read -r -a PORTS <<< "${P1_PORTS:-9100 9101}"
LOG=$W/outputs/p1/logs
mkdir -p "$LOG"
for i in $(seq 0 $((N - 1))); do
  PORT=${PORTS[$((i % ${#PORTS[@]}))]}
  nohup python "$W/code/pilot/p1_client.py" --bench "$BENCH" --condition "$COND" \
    --shard "$i" --num-shards "$N" --server-url "ws://127.0.0.1:$PORT" "$@" \
    > "$LOG/${BENCH}_${COND}_s$i.out" 2>&1 &
done
sleep 5
echo "launched $N shards of $BENCH/$COND"
tail -n 1 "$LOG/${BENCH}_${COND}"_s*.out
