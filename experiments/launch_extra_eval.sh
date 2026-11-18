#!/usr/bin/env bash
# Further evaluations. Run after the main evaluation (launch_eval.sh), from experiments/.
#   PART=curve   learning curves on a fixed evaluation set: checkpoints every CURVE_EVERY iterations of
#                full / ppo_only / ppo_mlp, 5 seeds, training terrains, CURVE_EPS episodes of CURVE_SEC s
#                -> results_curve/       (plot_extra.py curve)
#   PART=speed   commanded speeds SPEEDS for the main controllers (0.5 m/s is the main evaluation), training terrains
#                -> results_speed/v<speed>/   (plot_extra.py speed --results results/ results_speed/v*)
#   PART=sweep   slope-angle and stair-rise sweeps for the main controllers, SWEEP_EPS episodes of SWEEP_SEC s
#                -> results_sweep/       (plot_extra.py sweep)
#   PART=all     the three, in this order
#   PART=grid    OPT-IN, not part of "all": the slope and stair sweeps
#                repeated at the commanded speeds GRID_SPEEDS -> results_grid/v<speed>/; with results_sweep/ (0.5 m/s)
#                they form a speed x difficulty grid (plot_ral.py grid, a 3D-bar figure in the style of REFINE-DP Fig. 10)
# Usage: PART=all ./launch_extra_eval.sh my_adapter.py
# Every run_e0 call evaluates one checkpoint (or one run's curve checkpoints) on all its terrains, one process per
# terrain; a finished part is not skipped, so re-run a single part with PART=<part>.
set -euo pipefail
ADAPTER=$1
PART=${PART:-all}
PY=${PY:-python}
TRAIN=${TRAIN:-../outputs/Biped/train}
SEEDS5=${SEEDS5:-"1 2 3 4 5"}
SEEDS3=${SEEDS3:-"1 2 3"}
TRAIN_TERRAINS=${TRAIN_TERRAINS:-"flat rough slope18 stair_up stair_down"}
CURVE_EVERY=${CURVE_EVERY:-500}; CURVE_EPS=${CURVE_EPS:-50}; CURVE_SEC=${CURVE_SEC:-20}
SPEEDS=${SPEEDS:-"0.3 0.8"}; SPEED_EPS=${SPEED_EPS:-100}; SPEED_SEC=${SPEED_SEC:-60}
SWEEP_EPS=${SWEEP_EPS:-100}; SWEEP_SEC=${SWEEP_SEC:-20}
GRID_SPEEDS=${GRID_SPEEDS:-"0.3 0.8"}
HERE="$(cd "$(dirname "$0")" && pwd)"
# Read the sweep levels from my_adapter.py WITHOUT importing it (a bare import of it can segfault outside
# an Isaac Gym process), so nothing here depends on that import.
SWEEP_TERRAINS=${SWEEP_TERRAINS:-"$($PY - "$HERE/my_adapter.py" <<'PYEOF'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
v = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
     if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
     and n.targets[0].id in ("SWEEP_SLOPES_DEG", "SWEEP_STAIRS_MM")}
print(" ".join([f"sweep_slope_{d}" for d in v["SWEEP_SLOPES_DEG"]] + [f"sweep_stair_{m}" for m in v["SWEEP_STAIRS_MM"]]))
PYEOF
)"}
EXTRA=${EXTRA:-}   # extra run_e0 arguments for every call (e.g. "--device cuda:1")
# shellcheck disable=SC2086
e0() { $PY "$HERE/run_e0.py" --adapter "$ADAPTER" "$@" $EXTRA; }

ck() {   # ck <variant> <seed> <epoch...> -> checkpoint paths; same rule as Adapter.find_checkpoint, without the import:
  # at max_epochs rl_games writes the periodic ..._rew_<r>.pth and the final ..._rew__<r>_.pth (same weights): prefer the final
  local v=$1 s=$2 e hits pick out=(); shift 2
  for e in "$@"; do
    hits=$(find "$TRAIN/onepass_${v}_seed${s}" -name "*_ep_${e}_rew*.pth" 2>/dev/null | sort)
    [ -n "$hits" ] || { echo "no checkpoint for epoch $e in $TRAIN/onepass_${v}_seed${s}" >&2; return 1; }
    pick=$(printf '%s\n' "$hits" | grep '_rew__' | head -n 1); [ -n "$pick" ] || pick=$(printf '%s\n' "$hits" | head -n 1)
    out+=("$pick")
  done
  echo "${out[*]}"
}
modes_of() { case $1 in full) echo "SP AD";; ppo_only) echo "SP";; *) echo "ORIG";; esac; }
nosp_of() { case $1 in full|ppo_only) echo "";; *) echo "--no-sp-deviation";; esac; }
bc_ck() { echo "$TRAIN/bc_$1_seed$2/bc.pt"; }

part_curve() {
  local eps; eps=$(seq "$CURVE_EVERY" "$CURVE_EVERY" 5000 | tr '\n' ' ')
  for v in full ppo_only ppo_mlp; do
    for s in $SEEDS5; do
      # shellcheck disable=SC2086
      local cks; cks=($(ck "$v" "$s" $eps))
      echo "[curve] $v seed $s: ${#cks[@]} checkpoints"
      # shellcheck disable=SC2046,SC2086
      e0 --checkpoint "${cks[0]}" --curve-checkpoints "${cks[@]:1}" --method "$v" --train-seed "$s" \
          --out results_curve --terrains $TRAIN_TERRAINS --modes $(modes_of $v) $(nosp_of $v) \
          --episodes "$CURVE_EPS" --episode-seconds "$CURVE_SEC" --latency-iters 10
    done
  done
}

part_speed() {
  for v_cmd in $SPEEDS; do
    local out="results_speed/v$v_cmd"
    for v in full ppo_mlp; do
      for s in $SEEDS5; do
        # shellcheck disable=SC2046,SC2086
        ONEPASS_EVAL_VX=$v_cmd e0 --checkpoint "$(ck "$v" "$s" 5000)" --method "$v" --train-seed "$s" --out "$out" \
            --terrains $TRAIN_TERRAINS --modes $(modes_of $v) $(nosp_of $v) --episodes "$SPEED_EPS" \
            --episode-seconds "$SPEED_SEC" --latency-iters 10
      done
    done
    for a in birodiff diffuseloco; do
      for s in $SEEDS3; do
        # shellcheck disable=SC2086
        ONEPASS_EVAL_VX=$v_cmd e0 --checkpoint "$(bc_ck $a $s)" --method "$a" --train-seed "$s" --out "$out" \
            --terrains $TRAIN_TERRAINS --modes ORIG --no-sp-deviation --episodes "$SPEED_EPS" \
            --episode-seconds "$SPEED_SEC" --latency-iters 10
      done
    done
  done
}

part_sweep() {   # SWEEP_OUT: output directory (default results_sweep; part_grid sets one per speed)
  local out=${SWEEP_OUT:-results_sweep}
  for v in full ppo_mlp; do
    for s in $SEEDS5; do
      # shellcheck disable=SC2046,SC2086
      e0 --checkpoint "$(ck "$v" "$s" 5000)" --method "$v" --train-seed "$s" --out "$out" \
          --terrains $SWEEP_TERRAINS --modes $(modes_of $v) $(nosp_of $v) \
          --episodes "$SWEEP_EPS" --episode-seconds "$SWEEP_SEC" --latency-iters 10
    done
  done
  for a in birodiff diffuseloco; do
    for s in $SEEDS3; do
      # shellcheck disable=SC2086
      e0 --checkpoint "$(bc_ck $a $s)" --method "$a" --train-seed "$s" --out "$out" \
          --terrains $SWEEP_TERRAINS --modes ORIG --no-sp-deviation \
          --episodes "$SWEEP_EPS" --episode-seconds "$SWEEP_SEC" --latency-iters 10
    done
  done
}

part_grid() {
  for v_cmd in $GRID_SPEEDS; do
    ( export ONEPASS_EVAL_VX=$v_cmd; SWEEP_OUT="results_grid/v$v_cmd" part_sweep )
  done
}

[ -n "$SWEEP_TERRAINS" ] || { echo "could not read the sweep levels from my_adapter.py" >&2; exit 1; }
case $PART in
  curve) part_curve ;;
  speed) part_speed ;;
  sweep) part_sweep ;;
  grid) part_grid ;;
  all) part_curve; part_speed; part_sweep ;;
  *) echo "PART must be curve, speed, sweep, all or grid" >&2; exit 1 ;;
esac
echo "extra evaluation done: $PART"
