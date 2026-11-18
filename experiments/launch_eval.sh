#!/usr/bin/env bash
# Evaluate ONE checkpoint on the whole frozen suite: one process per terrain (Isaac Gym allows one sim
# per process), pass 1 -> pool AD stats -> pass 2 (compute-matched controls).
#
# Usage: ./launch_eval.sh <adapter.py> <checkpoint> <method> <train_seed> <out_dir> [episodes]
# Env overrides:
#   PY        python executable                 (default: python)
#   TERRAINS  space-separated terrain list      (default: full suite)
#   PASS1     modes for pass 1                  (default below; add ORIG for the checkpoint's own sampler)
#   PASS2     compute-matched modes             (default: AD-mean AD-shuffle AD-random)
#   PASS1_ADD modes appended to pass 1          (e.g. "SP-retest": SP again as the last mode, run-to-run variation)
#   EXTRA     extra args for run_e0.py          (e.g. "--episode-seconds 60 --device cuda:1")
set -euo pipefail
ADAPTER=$1; CKPT=$2; METHOD=$3; SEED=$4; OUT=$5; EPS=${6:-200}
PY=${PY:-python}
TERRAINS=${TERRAINS:-"flat rough slope10 slope18 stair_up stair_down discrete heldout_slope heldout_stairs heldout_gaps"}   # transition course: not implemented for terrain.py
PASS1=${PASS1:-"SP D1 D2 D3 D5 D7 D8 D10 AD D7@0 AD@0 ES"}
PASS2=${PASS2-"AD-mean AD-shuffle AD-random"}   # PASS2="" skips pass 2
PASS1="$PASS1 ${PASS1_ADD:-}"
EXTRA=${EXTRA:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
for T in $TERRAINS; do
  $PY "$HERE/run_e0.py" --adapter "$ADAPTER" --checkpoint "$CKPT" --method "$METHOD" --train-seed "$SEED" \
      --out "$OUT" --terrains "$T" --modes $PASS1 --episodes "$EPS" $EXTRA
done
if [ -n "$PASS2" ]; then
  $PY "$HERE/pool_ad_stats.py" --out "$OUT" --method "$METHOD" --train-seed "$SEED" --token AD
  for T in $TERRAINS; do
    $PY "$HERE/run_e0.py" --adapter "$ADAPTER" --checkpoint "$CKPT" --method "$METHOD" --train-seed "$SEED" \
        --out "$OUT" --terrains "$T" --modes $PASS2 --episodes "$EPS" \
        --ad-stats "$OUT/adstats_${METHOD}_seed${SEED}_AD.json" $EXTRA
  done
fi
echo "done: $METHOD seed $SEED -> $OUT"
