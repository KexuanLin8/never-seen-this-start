#!/usr/bin/env bash
# Full-scale synthesis for all four suites, then packing into LeRobot v2.1 datasets.
#   nohup bash run_synth.sh > $W/outputs/synth_full.log 2>&1 &
# MODES / PER_DEMO / SHARDS / SYNTH_ARGS can be overridden from the environment, e.g. SHARDS=4 bash run_synth.sh
#   MODES=chain SYNTH_ARGS="--recovery-steps 400" bash run_synth.sh
# TAG names the staging folder and the dataset (default: the mode), e.g. TAG=chainho -> synth_lerobot/chainho_<suite>
set -uo pipefail
W=${W:?set W to your work directory, see README}
PY_LIBERO=$W/conda/envs/libero/bin/python
PY_EVO=$W/conda/envs/Evo1/bin/python
CODE=$W/code/remedy
MODES=${MODES:-"start mid"}
PER_DEMO=${PER_DEMO:-2}
SHARDS=${SHARDS:-6}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
unset LIBERO_CONFIG_PATH  # vanilla LIBERO (~/.libero), not LIBERO-plus

for mode in $MODES; do
  tag=${TAG:-$mode}
  for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "$(date '+%F %T') synth $mode $suite"
    for i in $(seq 0 $((SHARDS - 1))); do
      "$PY_LIBERO" "$CODE/synth_libero.py" --suite "$suite" --mode "$mode" --per-demo "$PER_DEMO" \
        --tag "$tag" --shard "$i" --num-shards "$SHARDS" ${SYNTH_ARGS:-} > "$W/outputs/synth_${tag}_${suite}_s$i.log" 2>&1 &
    done
    wait
  done
done

for mode in $MODES; do
  tag=${TAG:-$mode}
  for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "$(date '+%F %T') build $tag $suite"
    "$PY_EVO" "$CODE/build_lerobot_v21.py" --staging "$W/data/synth_staging/$tag/$suite" \
      --template "$W/data/libero_lerobot_parquet/${suite}_no_noops_1.0.0_lerobot" \
      --out "$W/data/synth_lerobot/${tag}_${suite}"
  done
done
echo "$(date '+%F %T') SYNTH DONE"
