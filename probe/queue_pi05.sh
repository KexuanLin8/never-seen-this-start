#!/usr/bin/env bash
# pi0.5 diagnosis with the same protocol as the lightweight models. Starts and stops its own probe servers
# (ports 9501.., PI_SERVERS of them, default 2; two clients per server); clients resume finished episodes.
#   bash queue_pi05.sh smoke                     10 episodes in a separate output folder: load, speed, sanity
#   setsid nohup bash queue_pi05.sh full > $W/outputs/probe/logs/queue_pi05.out 2>&1 < /dev/null &
# Rounds are ordered by what the paper needs most: the state-reliance prediction first.
source "$(dirname "$0")/queue_lib.sh"
MODE=${1:?mode: smoke | full}
NS=${PI_SERVERS:-2}
CKPT=${PI_CKPT:-$W/ckpt/pi05_libero}
PY_LR=$W/conda/envs/lerobot/bin/python
ARGS="--render-res 256 --dummy-gripper -1 --horizon 10"  # n_action_steps=10 as in LeRobot's and OpenPI's LIBERO evaluation
REF=$W/outputs/probe/homing_ref.json
SERVERS=()
trap 'for p in "${SERVERS[@]}"; do kill "$p" 2>/dev/null; done' EXIT

for s in $(seq 0 $((NS - 1))); do
  port=$((9501 + s))
  slog=$LOG/server_pi05_$port.out
  "$PY_LR" "$W/code/probe/probe_server.py" --model lerobot --ckpt "$CKPT" --port "$port" > "$slog" 2>&1 &
  SERVERS+=("$!")
  for _ in $(seq 180); do
    grep -q "ready on" "$slog" && break
    kill -0 "${SERVERS[-1]}" 2>/dev/null || { echo "  server $port died:"; tail -n 8 "$slog"; exit 1; }
    sleep 5
  done
  grep -q "ready on" "$slog" || { echo "  server $port not ready after 15 min"; exit 1; }
  grep -E "default_horizon|type=" "$slog" | tail -n 2
done
echo "$(date '+%F %T') $NS pi0.5 servers ready; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use"

cond() {  # cond <bench> <condition> [client args...]: two clients per server
  local bench=$1 c=$2 n=$((NS * 2))
  shift 2
  round_start "pi05 $c"
  for i in $(seq 0 $((n - 1))); do launch "$bench" pi05 "$c" $((9501 + i % NS)) "$i" "$n" $ARGS "$@"; done
  round_end
}

if [ "$MODE" = smoke ]; then
  round_start "pi05 smoke (10 episodes)"
  for i in $(seq 0 $((NS * 2 - 1))); do
    launch libero pi05_smoke C0_smoke $((9501 + i % NS)) "$i" $((NS * 2)) $ARGS --suites libero_spatial --inits-per-task 1 \
      --out-root "$W/outputs/probe_smoke"
  done
  round_end
  cat "$W"/outputs/probe_smoke/libero/pi05_smoke/C0_smoke/results_*.jsonl | python3 -c "import sys,json; r=[json.loads(l) for l in sys.stdin if l.strip()]; print(len(r),'episodes, success',round(100*sum(x['success'] for x in r)/len(r),1),'%, mean',round(sum(x['seconds'] for x in r)/len(r),1),'s per episode')"
  exit 0
fi

# 1. does the state matter? (prediction: yes, the state is serialised into the prompt tokens)
cond libero C0_full --branches no_lang no_state no_agent no_wrist
cond libero C3_state_zero --state-mode zero
cond libero S1_bias_1cm --state-mode bias --state-bias 0.01
cond libero S2_bias_2cm --state-mode bias --state-bias 0.02
cond libero S3_bias_4cm --state-mode bias --state-bias 0.04
# 2. the start: dose-response, homing, chaining
cond libero T1_start_010 --start-perturb 0.1
cond libero T2_start_020 --start-perturb 0.2
cond libero T3_start_030 --start-perturb 0.3
cond libero T4_start_050 --start-perturb 0.5
cond libero_plus H0c_full --prompt-mode clean --categories robot --frac 0.1
cond libero_plus H1c_homing --prompt-mode clean --categories robot --frac 0.1 --homing-ref "$REF"
cond libero K1_chain --chain-starts "$W/outputs/probe/chain_starts.json"
# 3. matched-displacement pushes
cond libero X1_push_early_mild   --disturb-at 40  --disturb-scale 0.2
cond libero X2_push_early_strong --disturb-at 40  --disturb-scale 0.4
cond libero X3_push_late_strong  --disturb-at 120 --disturb-scale 0.4
# 4. language, views, LIBERO-plus overall
cond libero C1_lang_cf --prompt-mode counterfactual
cond libero C2_lang_empty --prompt-mode empty
cond libero C6_mask_agent --mask-views 0
cond libero C7_mask_wrist --mask-views 1
cond libero_plus P0c_full --prompt-mode clean --frac 0.05
echo "$(date '+%F %T') QUEUE pi05 DONE"
