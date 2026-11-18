#!/usr/bin/env bash
# Run the training jobs in runs.txt, PARALLEL at a time on one GPU (check memory with nvidia-smi first).
# Each line "variant seed" becomes:  cd $REPO/envs && VARIANT=variant SEED=seed bash run.sh onepass
# (the variants are defined in $REPO/envs/exp_onepass.sh). Activate the Isaac Gym conda env first.
#
# Usage: REPO=/path/to/OnePassDiff PARALLEL=2 GPU=0 ./launch_train.sh [runs.txt]
#        DRY=1 ./launch_train.sh        # print the commands only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS=$(realpath "${1:-$HERE/runs.txt}")
REPO=$(realpath "${REPO:-$(dirname "$HERE")}")
PARALLEL=${PARALLEL:-1}
GPU=${GPU:-0}
LOGDIR=$(realpath -m "${LOGDIR:-$HERE/train_logs}")
DRY=${DRY:-}
[ -f "$REPO/envs/exp_onepass.sh" ] || { echo "REPO=$REPO has no envs/exp_onepass.sh" >&2; exit 1; }
[ -f "$REPO/experiments/onepass_kit.py" ] || { echo "put experiments/ in the repo root ($REPO/experiments)" >&2; exit 1; }
mkdir -p "$LOGDIR"
export REPO GPU LOGDIR DRY

grep -v '^\s*#' "$RUNS" | grep -v '^\s*$' | \
  xargs -P "$PARALLEL" -L 1 bash -c '
    v=$0; s=$1
    if [ -n "$DRY" ]; then (cd "$REPO/envs" && VARIANT=$v SEED=$s bash run.sh onepass -r | tail -1); exit 0; fi
    echo "[start] $v seed $s"
    (cd "$REPO/envs" && CUDA_VISIBLE_DEVICES=$GPU VARIANT=$v SEED=$s bash run.sh onepass) \
      > "$LOGDIR/${v}_seed${s}.log" 2>&1 && echo "[done]  $v seed $s" || echo "[FAIL]  $v seed $s (see $LOGDIR)"'
