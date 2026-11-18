#!/usr/bin/env bash
# BiRoDiff-R and DiffuseLoco-R. Run after ppo_mlp seed 1 has finished training.
#   1. collect demonstrations from PPO-MLP seed 1 in its training environment (Isaac Gym, ~10 min)
#   2. train 3 seeds of each baseline by behaviour cloning (torch only)
# Usage, from OnePassDiff/experiments with the Isaac Gym env active:
#   ./run_baselines.sh                    # everything
#   STEP=train ./run_baselines.sh         # skip collection (demos already there)
#   SEEDS="1" ARCHS="diffuseloco" ./run_baselines.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY=${PY:-python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3"}
ARCHS=${ARCHS:-"birodiff diffuseloco"}
OUTROOT=${OUTROOT:-../outputs/Biped/train}
DEMOS=${DEMOS:-demos/ppo_mlp_seed1.pt}
STEP=${STEP:-all}
mkdir -p demos train_logs

if [ "$STEP" = "all" ] || [ "$STEP" = "collect" ]; then
  CK=$($PY -c "from my_adapter import Adapter as A; print(A.find_checkpoint('$OUTROOT/onepass_ppo_mlp_seed1', ${EXPERT_EPOCH:-5000}))" | tail -n 1)
  [ -f "$CK" ] || { echo "[baselines] no PPO-MLP seed 1 checkpoint (got: $CK)" >&2; exit 1; }
  echo "[baselines] expert: $CK"
  CUDA_VISIBLE_DEVICES=$GPU $PY collect_demos.py --expert "$CK" --out "$DEMOS" --device cuda:0 \
      2>&1 | tee train_logs/collect_demos.log
fi
if [ "$STEP" = "all" ] || [ "$STEP" = "train" ]; then
  for a in $ARCHS; do for s in $SEEDS; do
    echo "[baselines] train $a seed $s"
    CUDA_VISIBLE_DEVICES=$GPU $PY train_bc.py --demos "$DEMOS" --arch "$a" --seed "$s" \
        --out "$OUTROOT/bc_${a}_seed${s}" --device cuda:0 > "train_logs/bc_${a}_seed${s}.log" 2>&1 \
      && echo "[done]  $a seed $s" || echo "[FAIL]  $a seed $s (train_logs/bc_${a}_seed${s}.log)"
  done; done
fi
