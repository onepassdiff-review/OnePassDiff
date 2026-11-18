#!/usr/bin/env python3
"""Statistics: seeds are the unit of analysis.

Reads results/episodes_*.csv written by run_e0.py and produces
  summary.csv / summary.md : every (method, mode, terrain) cell, mean over seeds with a 95% CI from a
                             two-level bootstrap (resample training seeds, then episodes within seed)
  comparisons_<metric>.md  : requested comparisons with bootstrap p-values, Holm-corrected across terrains
  table_<name>.tex         : LaTeX rows "mean [lo, hi]" (LaTeX table cells)
  table_<name>_perseed.tex : the same cells, one row per training seed (per-seed table)

Pairing: two cells with the SAME method share checkpoints and evaluation episodes (run_e0 replays the
same episodes for every mode), so their difference is bootstrapped pairwise (same seeds, same episodes).
Different methods are independent training runs and are bootstrapped independently.

Examples
  python analyze_results.py --results results/ --out analysis/
  python analyze_results.py --results results/ --out analysis/ \\
      --latex inference "ppo_only:SP,ppo_only:D5,ppo_only:D7,ppo_only:D10,ppo_only:AD,ppo_only:AD-mean" \\
      --terrains flat rough slope18 stair_up stair_down \\
      --compare full:AD full:AD-mean --compare full:AD full:SP --compare ppo_only:SP ppo_only:ORIG
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

METRICS_PCT = {"success": 100.0, "progress": 100.0, "fall": 100.0, "collision": 100.0}
# decimals in tables/summaries (default 1): small-valued metrics need more
DECIMALS = {"cot": 3, "vel_rmse": 3, "mean_eta": 3, "dev_from_sp": 3, "mean_n": 2, "jerk_rms": 0, "max_exc_rad": 4,
            "spl": 3, "footsteps": 0, "stumble_pct": 2}
# Locomotion outcomes derived from run_e0's raw fields (older CSVs may lack them):
#   fall / collision  share of episodes ended by a fall (term_cause 1) / a leg collision (term_cause 2); with success
#                     they sum to 1 (term_cause 3, unclassified, is reported in the load() check)
#   spl               SPL (Anderson et al., 2018) adapted to fixed-duration velocity tracking: the goal is the point
#                     l = v T ahead, d the distance travelled along the command, p the executed path length:
#                     SPL = S d / max(p, l) = S progress / max(p / l, 1), in [0, 1]; l = cmd_speed * seconds_alive for
#                     successful episodes (= v T), and failed episodes score 0
DERIVED = ("fall", "collision", "spl")
METRIC_CHOICES = ["success", "progress", "vel_rmse", "mean_n", "mean_eta", "dev_from_sp", "seconds_alive",
                  "cot", "joint_viol_per_min", "limit_time_pct", "max_exc_rad", "target_viol_pct", "jerk_rms",
                  "path_len", "footsteps", "stumble_pct", *DERIVED]
D = 1   # set from --metric in main()


def load(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "episodes_*.csv")))
    if not files:
        raise SystemExit(f"no episodes_*.csv in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["cell"] = df["method"] + ":" + df["mode"]
    # one evaluation condition per directory: learning-curve and speed-sweep results go to their own
    # directories and are summarised by plot_extra.py (otherwise the de-duplication below would mix them)
    multi_ck = df.groupby(["method", "train_seed"])["checkpoint"].nunique()
    if (multi_ck > 1).any():
        raise SystemExit(f"{results_dir} holds several checkpoints of one run ({list(multi_ck[multi_ck > 1].index)}): "
                         "learning-curve results? use plot_extra.py curve")
    if "cmd_speed" in df and df["cmd_speed"].round(3).nunique() > 1:
        raise SystemExit(f"{results_dir} mixes commanded speeds {sorted(df['cmd_speed'].round(3).unique())}: "
                         "keep one speed per directory (results_speed/v<speed>)")
    # keep the last evaluation if a cell was re-run
    df = df.drop_duplicates(subset=["method", "mode", "terrain", "train_seed", "episode"], keep="last")
    if "term_cause" in df:
        tc = df["term_cause"]
        df["fall"] = (tc == 1).astype(float).where(tc.notna())
        df["collision"] = (tc == 2).astype(float).where(tc.notna())
        n3 = int((tc == 3).sum())
        if n3:
            print(f"[analyze] {n3} terminated episodes with an unclassified cause (term_cause 3)")
    if "path_len" in df:
        ell = (df["cmd_speed"] * df["seconds_alive"]).clip(lower=1e-6)
        ratio = (df["path_len"] / ell).clip(lower=1.0)
        df["spl"] = df["success"].astype(float) * df["progress"].clip(lower=0.0) / ratio
        df.loc[df["success"].astype(float) == 0, "spl"] = 0.0
        df.loc[df["path_len"].isna(), "spl"] = np.nan
    return df


def matrix(df, cell, terrain, metric):
    """[n_seeds, n_eps] array (episodes truncated to the minimum count across seeds) and seed ids."""
    sub = df[(df.cell == cell) & (df.terrain == terrain)]
    if sub.empty:
        return None, []
    groups = {s: g.sort_values("episode")[metric].to_numpy(float) for s, g in sub.groupby("train_seed")}
    n = min(len(v) for v in groups.values())
    seeds = sorted(groups)
    return np.stack([groups[s][:n] for s in seeds]), seeds


def boot_means(M, rng, n_boot, seed_idx=None, ep_idx=None):
    S, E = M.shape
    if seed_idx is None:
        seed_idx = rng.integers(0, S, (n_boot, S))
        ep_idx = rng.integers(0, E, (n_boot, S, E))
    vals = M[seed_idx[..., None], ep_idx]  # [n_boot, S, E]
    return np.nanmean(np.nanmean(vals, axis=2), axis=1), seed_idx, ep_idx


def ci(M, rng, n_boot):
    point = float(np.nanmean(np.nanmean(M, axis=1)))
    if M.shape[0] < 2:
        return point, np.nan, np.nan
    b, _, _ = boot_means(M, rng, n_boot)
    return point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def compare(df, a, b, terrain, metric, rng, n_boot):
    Ma, sa = matrix(df, a, terrain, metric)
    Mb, sb = matrix(df, b, terrain, metric)
    if Ma is None or Mb is None:
        return None
    same_method = a.split(":")[0] == b.split(":")[0]
    if same_method:
        common = sorted(set(sa) & set(sb))
        Ma = Ma[[sa.index(s) for s in common]]
        Mb = Mb[[sb.index(s) for s in common]]
        E = min(Ma.shape[1], Mb.shape[1])
        D = Ma[:, :E] - Mb[:, :E]
        point = float(np.nanmean(np.nanmean(D, axis=1)))
        bd, _, _ = boot_means(D, rng, n_boot)
        n_seeds = len(common)
    else:
        point = float(np.nanmean(np.nanmean(Ma, axis=1)) - np.nanmean(np.nanmean(Mb, axis=1)))
        ba, _, _ = boot_means(Ma, rng, n_boot)
        bb, _, _ = boot_means(Mb, rng, n_boot)
        bd = ba - bb
        n_seeds = (Ma.shape[0], Mb.shape[0])
    p = 2 * min(np.mean(bd <= 0), np.mean(bd >= 0))
    p = float(min(1.0, max(p, 1.0 / n_boot)))
    return {"diff": point, "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
            "p": p, "paired": same_method, "n_seeds": n_seeds}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fmt(point, lo, hi, scale):
    if np.isnan(lo):
        return f"{point * scale:.{D}f}"
    return f"{point * scale:.{D}f} [{lo * scale:.{D}f}, {hi * scale:.{D}f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="success", choices=METRIC_CHOICES)
    ap.add_argument("--terrains", nargs="*", default=None)
    ap.add_argument("--latex", nargs=2, action="append", default=[], metavar=("NAME", "CELLS"),
                    help="table name and comma-separated method:mode cells (rows)")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    scale = METRICS_PCT.get(args.metric, 1.0)
    global D
    D = DECIMALS.get(args.metric, 1)

    df = load(args.results)
    if args.metric not in df:
        raise SystemExit(f"{args.metric} is not in {args.results} (results written by run_e0 before the locomotion "
                         "outcome fields were added?)")
    terrains = args.terrains or sorted(df.terrain.unique())

    rows = []
    for (cell, terrain), _ in df.groupby(["cell", "terrain"]):
        M, seeds = matrix(df, cell, terrain, args.metric)
        point, lo, hi = ci(M, rng, args.n_boot)
        rows.append({"cell": cell, "terrain": terrain, "n_seeds": len(seeds), "episodes_per_seed": M.shape[1],
                     "mean": point * scale, "ci_lo": lo * scale, "ci_hi": hi * scale,
                     "per_seed": ";".join(f"{v * scale:.{D}f}" for v in np.nanmean(M, axis=1))})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out, f"summary_{args.metric}.csv"), index=False)
    with open(os.path.join(args.out, f"summary_{args.metric}.md"), "w") as fh:
        fh.write(f"# {args.metric}: mean over training seeds, 95% two-level bootstrap CI\n\n")
        try:
            fh.write(summ.to_markdown(index=False, floatfmt=".1f"))
        except ImportError:  # tabulate not installed
            fh.write(summ.to_string(index=False, float_format=lambda v: f"{v:.{D}f}"))
        few = summ[summ.n_seeds < 3]
        if len(few):
            fh.write(f"\n\nWARNING: {len(few)} cells have fewer than 3 training seeds; their CIs are not meaningful.\n")
    print(f"summary: {len(summ)} cells -> {args.out}")

    if args.compare:
        lines = [f"# Comparisons on {args.metric} (difference A - B, scale x{scale:g})\n",
                 "Holm correction across terrains within each comparison. 'paired' = same checkpoints and episodes.\n"]
        for a, b in args.compare:
            res = {t: compare(df, a, b, t, args.metric, rng, args.n_boot) for t in terrains}
            res = {t: r for t, r in res.items() if r is not None}
            if not res:
                lines.append(f"\n## {a} vs {b}: no common cells\n")
                continue
            adj = holm(np.array([r["p"] for r in res.values()]))
            lines.append(f"\n## {a} vs {b}\n\n| terrain | diff | 95% CI | p | p (Holm) | paired | seeds | verdict |\n"
                         "|---|---|---|---|---|---|---|---|\n")
            for (t, r), pa in zip(res.items(), adj):
                verdict = ("A > B" if r["diff"] > 0 else "A < B") if pa < 0.05 else "comparable"
                lines.append(f"| {t} | {r['diff'] * scale:+.{D}f} | [{r['lo'] * scale:+.{D}f}, {r['hi'] * scale:+.{D}f}] | "
                             f"{r['p']:.3f} | {pa:.3f} | {r['paired']} | {r['n_seeds']} | {verdict} |\n")
        with open(os.path.join(args.out, f"comparisons_{args.metric}.md"), "w") as fh:
            fh.writelines(lines)
        print("".join(lines))

    for name, cells in args.latex:
        out = []
        for cell in cells.split(","):
            vals = []
            for t in terrains:
                M, _ = matrix(df, cell, t, args.metric)
                vals.append("--" if M is None else fmt(*ci(M, rng, args.n_boot), scale))
            out.append(f"{cell} & " + " & ".join(vals) + r" \\")
        path = os.path.join(args.out, f"table_{name}_{args.metric}.tex")
        with open(path, "w") as fh:
            fh.write(f"% columns: {' | '.join(terrains)}\n" + "\n".join(out) + "\n")
        print(f"latex rows -> {path}")
        # per-seed table (supplementary): one row per training seed, mean over that seed's episodes
        ps = []
        for cell in cells.split(","):
            per_t = {}
            for t in terrains:
                M, seeds = matrix(df, cell, t, args.metric)
                if M is not None:
                    per_t[t] = dict(zip(seeds, np.nanmean(M, axis=1)))
            for sd in sorted({sd for d in per_t.values() for sd in d}):
                vals = [f"{per_t[t][sd] * scale:.{D}f}" if t in per_t and sd in per_t[t] else "--" for t in terrains]
                ps.append(f"{cell} & {sd} & " + " & ".join(vals) + r" \\")
        path = os.path.join(args.out, f"table_{name}_{args.metric}_perseed.tex")
        with open(path, "w") as fh:
            fh.write(f"% per training seed; columns: cell | seed | {' | '.join(terrains)}\n" + "\n".join(ps) + "\n")
        print(f"per-seed rows -> {path}")


if __name__ == "__main__":
    main()
