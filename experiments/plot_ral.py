#!/usr/bin/env python3
"""Heatmaps and charts for the RA-L letter. Reads the files run_e0.py writes; adds no new statistics.

Seeds are the unit of analysis, as in analyze_results.py: means are over training seeds, intervals are 95%
two-level bootstrap CIs (seeds, then episodes), and comparisons within one method are paired bootstraps
(analyze_results.compare) with Holm correction across the terrains of a row.

Subcommands
  modes   --results results/ --methods ppo_only full [--ref SP]
          heatmap, one panel per method: rows = inference modes, columns = terrains. The reference row (SP)
          shows success (%); every other row shows its paired difference from SP (percentage points), bold
          with "*" where the Holm-corrected p < 0.05. A right-hand column gives the network evaluations (NFE)
          per control step. Replaces the inference-mode table and the success-vs-N figure.
  steps   --results results/ --methods full full_tnppriv full_tnpuntrained [--token AD]
          heatmap, one panel per step-predictor head: rows = terrains, columns = N, cell = share of logged
          control steps (all seeds pooled with equal weight per seed); mean N and mean eta per terrain at the
          right. Also writes steps_separation.csv: per seed, the share of the variance of eta (and of N)
          explained by the terrain (R^2 of a one-way decomposition), mean and t-interval over seeds.
  sweep   --results results_sweep/ --cells ppo_mlp:ORIG full:SP full:AD ...
          heatmap, rows = controllers, columns = difficulty levels (slope and stair panels), cell = success (%);
          a dashed line marks the training maximum, the right-hand column the maximum traversable level
          (mean +- s.d. over seeds; rule of plot_extra.py sweep).
  pareto  --results results/ --cells full:SP full:D5 full:D10 full:AD ppo_mlp:ORIG ... [--x-csv latency.csv]
          success averaged over --terrains (two-level bootstrap CI) against computation per control step: NFE
          by default (D-N: N; AD-type modes: logged mean N; overrides with --nfe cell=value), or a measured
          latency from --x-csv (columns cell,ms). The non-dominated points are joined by a step line.
  grid    --results unused --grid 0.3=results_grid/v0.3 0.5=results_sweep 0.8=results_grid/v0.8 [--style bars3d|heatmaps]
          success over a commanded-speed x difficulty grid (launch_extra_eval.sh PART=grid, opt-in), slope and stair
          panels, one bar colour per controller, dashed line = training maximum. bars3d follows the 3D-bar style of
          REFINE-DP Fig. 10(a); heatmaps gives the same values as annotated small multiples (no occlusion).
  seedtable --results results/ --cells full:AD full:SP ppo_only:SP ppo_mlp:ORIG birodiff:ORIG diffuseloco:ORIG
          LaTeX rows of the per-seed table: success of every training run averaged over the terrains, and the mean
          over seeds [95% two-level bootstrap CI]
  seeds   --results results/ --cells full:SP full:AD ppo_mlp:ORIG
          heatmap, rows = controller x training seed, columns = terrains: the spread between independent runs.

Every subcommand writes <name>.pdf/.png and a CSV of the plotted values to --out.
Mock or synthetic inputs (checkpoint name containing "toy" or "MOCK") are stamped "MOCK DATA - NOT A RESULT".

Colours: the documented default categorical order (first five slots), one hue per controller and never per rank,
plus a marker and a line style per controller so that print in greyscale and colour-vision deficiency do not rely
on hue; heatmaps use a single-hue blue ramp (magnitude) and a red-grey-blue diverging ramp (difference).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_results as AR  # noqa: E402
import plot_extra as PE  # noqa: E402

TERRAIN_NAMES = {"flat": "Flat", "rough": "Rough", "slope10": "Slope 10°", "slope18": "Slope 18°",
                 "stair_up": "Stair up", "stair_down": "Stair down", "discrete": "Discrete",
                 "heldout_slope": "H/O slope", "heldout_stairs": "H/O stairs", "heldout_gaps": "H/O stones"}
DEFAULT_TERRAINS = list(TERRAIN_NAMES)
DEFAULT_MODES = ["SP", "RP", "D1", "D2", "D3", "D5", "D7", "D8", "D10", "D7@0", "AD", "AD@0",
                 "AD-mean", "AD-shuffle", "AD-random", "ES"]
CELL_NAMES = {"full": "OnePassDiff", "ppo_only": "OnePassDiff-PPO-only", "ppo_mlp": "PPO-MLP", "birodiff": "BiRoDiff-R",
              "diffuseloco": "DiffuseLoco-R", "full_tnppriv": "Privileged predictor",
              "full_tnpuntrained": "Untrained predictor"}
# fixed identity -> style (colour follows the controller, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STYLE = {"full:AD": (SERIES[0], "o", "-"), "full:SP": (SERIES[1], "s", "--"), "ppo_mlp:ORIG": (SERIES[2], "^", "-."),
         "birodiff:ORIG": (SERIES[3], "D", ":"), "diffuseloco:ORIG": (SERIES[4], "v", (0, (5, 1, 1, 1)))}
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GREY = ["#f0efec", "#c3c2b7", "#8e8d86", "#52514e"]
DIVERGING = ["#8f1d1c", "#c43a39", "#e98f8e", "#f0efec", "#86b6ef", "#2a78d6", "#104281"]
INK, MUTED = "#0b0b0b", "#52514e"
NFE_DEFAULT = {"birodiff:ORIG": 60.0, "diffuseloco:ORIG": 10.0, "ppo_mlp:ORIG": 1.0}


# ----------------------------------------------------------------------------- helpers
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK})
    return plt


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


def _ink_on(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def _is_mock(df):
    return df["checkpoint"].astype(str).str.contains("toy|MOCK", regex=True).any()


def _stamp(fig, mock):
    if mock:
        fig.text(0.5, 0.5, "MOCK DATA - NOT A RESULT", ha="center", va="center", rotation=20, fontsize=16,
                 color="#d03b3b", alpha=0.35, fontweight="bold", zorder=100)


def _label(cell):
    m, _, mode = cell.partition(":")
    base = CELL_NAMES.get(m, m)
    return base if mode in ("ORIG", "") else f"{base} ({mode})"


def _nfe(df, cell, overrides):
    if cell in overrides:
        return overrides[cell]
    if cell in NFE_DEFAULT:
        return NFE_DEFAULT[cell]
    mode = cell.partition(":")[2].removesuffix("-retest")
    if mode in ("SP", "ORIG"):
        return 1.0
    m = re.fullmatch(r"D(\d+)(@0)?", mode)
    if m:
        return float(m.group(1))
    v = df.loc[df.cell == cell, "mean_n"].to_numpy(float)   # adaptive / early-stopping modes: logged mean N
    return float(np.nanmean(v)) if v.size and not np.isnan(v).all() else np.nan


def _terrains(df, wanted):
    return [t for t in wanted if t in set(df.terrain)]


# ----------------------------------------------------------------------------- modes heatmap
def cmd_modes(args):
    plt = _plt()
    rng = np.random.default_rng(args.seed)
    df = PE._load(args.results)
    terrains = _terrains(df, args.terrains)
    panels, rows_csv, vmax = [], [], 0.0
    for method in args.methods:
        modes = [m for m in args.modes if f"{method}:{m}" in set(df.cell)]
        if args.ref not in modes:
            raise SystemExit(f"{method}:{args.ref} missing in {args.results}")
        val = np.full((len(modes), len(terrains)), np.nan)
        sig = np.zeros_like(val, bool)
        for i, mode in enumerate(modes):
            cell = f"{method}:{mode}"
            if mode == args.ref:
                for j, t in enumerate(terrains):
                    M, _ = AR.matrix(df, cell, t, args.metric)
                    if M is not None:
                        p, lo, hi = AR.ci(M, rng, args.n_boot)
                        val[i, j] = 100 * p
                        rows_csv.append({"method": method, "mode": mode, "terrain": t, "value": 100 * p,
                                         "lo": 100 * lo, "hi": 100 * hi, "kind": "level"})
                continue
            res = [AR.compare(df, cell, f"{method}:{args.ref}", t, args.metric, rng, args.n_boot) for t in terrains]
            ok = [k for k, r in enumerate(res) if r is not None]
            adj = AR.holm(np.array([res[k]["p"] for k in ok])) if ok else []
            for k, pa in zip(ok, adj):
                r = res[k]
                val[i, k] = 100 * r["diff"]
                sig[i, k] = pa < args.alpha
                rows_csv.append({"method": method, "mode": mode, "terrain": terrains[k], "value": 100 * r["diff"],
                                 "lo": 100 * r["lo"], "hi": 100 * r["hi"], "p": r["p"], "p_holm": pa,
                                 "kind": f"diff_vs_{args.ref}"})
            if ok:
                vmax = max(vmax, float(np.nanmax(np.abs(val[i]))))
        nfe = [_nfe(df, f"{method}:{m}", {}) for m in modes]
        panels.append((method, modes, val, sig, nfe))
    vmax = args.vmax or max(5.0, 5.0 * math.ceil(vmax / 5.0))
    seq, div = _cmap(GREY, "grey"), _cmap(DIVERGING, "div")
    from matplotlib.colors import Normalize
    nseq, ndiv = Normalize(0, 100), Normalize(-vmax, vmax)
    nrow = max(len(p[1]) for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(args.width, 0.155 * nrow + 0.95), squeeze=False,
                             gridspec_kw={"wspace": 0.42})
    for ax, (method, modes, val, sig, nfe) in zip(axes[0], panels):
        for i, mode in enumerate(modes):
            for j in range(len(terrains)):
                v = val[i, j]
                if np.isnan(v):
                    c = (1, 1, 1, 1)
                elif mode == args.ref:
                    c = seq(nseq(v))
                else:
                    c = div(ndiv(np.clip(v, -vmax, vmax)))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="white", lw=1.0))
                if not np.isnan(v):
                    txt = f"{v:.0f}" if mode == args.ref else f"{v:+.0f}" + ("*" if sig[i, j] else "")
                    ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=5, color=_ink_on(c),
                            fontweight="bold" if sig[i, j] else "normal")
            ax.text(len(terrains) + 0.15, i + 0.5, "—" if np.isnan(nfe[i]) else f"{nfe[i]:.1f}".rstrip("0").rstrip("."),
                    ha="left", va="center", fontsize=5.5, color=MUTED)
        ax.axhline(1, color=INK, lw=0.8)   # reference row above the line
        ax.set_xlim(0, len(terrains))
        ax.set_ylim(len(modes), 0)
        ax.set_xticks(np.arange(len(terrains)) + 0.5)
        ax.set_xticklabels([TERRAIN_NAMES.get(t, t) for t in terrains], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(modes)) + 0.5)
        ax.set_yticklabels([m.replace("@0", r" ($x_N{=}0$)") for m in modes])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(len(terrains) + 0.15, -0.3, "NFE", ha="left", va="bottom", fontsize=5.5, color=MUTED)
        ax.set_title(CELL_NAMES.get(method, method), pad=10)
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(ndiv, div), ax=axes[0].tolist(), fraction=0.025, pad=0.06)
    cb.set_label(f"Difference from {args.ref} (pp)")
    cb.outline.set_visible(False)
    _stamp(fig, _is_mock(df))
    PE._save(fig, args.out, args.name)
    pd.DataFrame(rows_csv).to_csv(os.path.join(args.out, f"{args.name}.csv"), index=False)


# ----------------------------------------------------------------------------- step-count heatmap
def _load_steplogs(dirs, method, token, terrain):
    import torch
    per_seed = {}
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, f"steplog_{method}_seed*_{token}_{terrain}.pt")))
    for f in files:
        m = re.search(rf"steplog_{re.escape(method)}_seed(\d+)_", os.path.basename(f))
        log = torch.load(f, map_location="cpu", weights_only=True)
        n = torch.cat([e["n"][e["alive"]] for e in log]).numpy()
        etas = [e["eta"][e["alive"]] for e in log if e.get("eta") is not None]
        eta = torch.cat(etas).numpy() if etas else np.full(n.shape, np.nan)
        per_seed[int(m.group(1))] = (n, eta)
    return per_seed


def _r2(groups):
    """share of the total variance explained by the group means (one-way decomposition)."""
    allv = np.concatenate(groups)
    tot = np.var(allv)
    if not np.isfinite(tot) or tot == 0:
        return np.nan
    gm = allv.mean()
    between = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / len(allv)
    return float(between / tot)


def cmd_steps(args):
    plt = _plt()
    if not any(glob.glob(os.path.join(d, "steplog_*.pt")) for d in args.results):
        raise SystemExit(f"no steplog_*.pt in {args.results}")
    ns = list(range(args.n_min, args.n_max + 1))
    panels, rows_csv, sep_rows, mock = [], [], [], False
    for method in args.methods:
        terrains, share, mean_n, mean_eta, by_seed = [], [], [], [], {}
        for t in args.terrains:
            ps = _load_steplogs(args.results, method, args.token, t)
            if not ps:
                continue
            terrains.append(t)
            # equal weight per seed: average the per-seed shares
            sh = np.mean([[np.mean(n == k) for k in ns] for n, _ in ps.values()], axis=0)
            share.append(100 * sh)
            mean_n.append(np.mean([n.mean() for n, _ in ps.values()]))
            mean_eta.append(np.nanmean([np.nanmean(e) for _, e in ps.values()]))
            for s, (n, e) in ps.items():
                by_seed.setdefault(s, {})[t] = (n.astype(float), e.astype(float))
            for k, v in zip(ns, sh):
                rows_csv.append({"method": method, "terrain": t, "N": k, "share_pct": 100 * v})
        if not terrains:
            print(f"no step logs for {method}:{args.token}; skipped")
            continue
        for qty, idx in (("eta", 1), ("N", 0)):
            r2 = [_r2([d[t][idx] for t in terrains if t in d]) for d in by_seed.values()]
            r2 = np.array([x for x in r2 if np.isfinite(x)])
            if not r2.size:   # zero variance: the head gives one value everywhere
                sep_rows.append({"method": method, "quantity": qty, "r2_mean": np.nan, "r2_ci_halfwidth": np.nan,
                                 "n_seeds": len(by_seed), "per_seed": "constant output"})
            if r2.size:
                m, h = PE.t_band(r2[:, None])
                sep_rows.append({"method": method, "quantity": qty, "r2_mean": float(m[0]),
                                 "r2_ci_halfwidth": float(h[0]), "n_seeds": int(r2.size),
                                 "per_seed": ";".join(f"{x:.3f}" for x in r2)})
        panels.append((method, terrains, np.array(share), mean_n, mean_eta))
    if not panels:
        raise SystemExit("no step logs for any method")
    seq = _cmap(["#ffffff"] + BLUE, "seq0")
    from matplotlib.colors import Normalize
    norm = Normalize(0, 100)
    nt = max(len(p[1]) for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(args.width, 0.16 * nt + 0.75), squeeze=False,
                             gridspec_kw={"wspace": 0.3})
    for ax, (method, terrains, share, mean_n, mean_eta) in zip(axes[0], panels):
        for i in range(len(terrains)):
            for j in range(len(ns)):
                c = seq(norm(share[i, j]))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="white", lw=1.0))
                if share[i, j] >= 0.5:
                    ax.text(j + 0.5, i + 0.5, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=5.2,
                            color=_ink_on(c))
            ax.text(len(ns) + 0.15, i + 0.5, f"{mean_n[i]:.1f}", ha="left", va="center", fontsize=5.2, color=MUTED)
        ax.text(len(ns) + 0.15, -0.25, r"$\bar N$", ha="left", va="bottom", fontsize=5.2, color=MUTED)
        ax.set_xlim(0, len(ns))
        ax.set_ylim(len(terrains), 0)
        ax.set_xticks(np.arange(len(ns)) + 0.5)
        ax.set_xticklabels(ns)
        ax.set_xlabel("Refinement steps $N$")
        ax.set_yticks(np.arange(len(terrains)) + 0.5)
        ax.set_yticklabels([TERRAIN_NAMES.get(t, t) for t in terrains] if ax is axes[0][0] else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title({"full": "Self-supervised"}.get(method, CELL_NAMES.get(method, method)), pad=8)
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm, seq), ax=axes[0].tolist(), fraction=0.03, pad=0.12)
    cb.set_label("Share of control steps (%)")
    cb.outline.set_visible(False)
    ck = [f for d in args.results for f in glob.glob(os.path.join(d, "episodes_*.csv"))]
    if ck:
        mock = _is_mock(pd.concat([pd.read_csv(f, usecols=["checkpoint"]) for f in ck]))
    _stamp(fig, mock or args.mock)
    PE._save(fig, args.out, args.name)
    pd.DataFrame(rows_csv).to_csv(os.path.join(args.out, f"{args.name}.csv"), index=False)
    pd.DataFrame(sep_rows).to_csv(os.path.join(args.out, f"{args.name}_separation.csv"), index=False)
    print(pd.DataFrame(sep_rows).to_string(index=False))


# ----------------------------------------------------------------------------- difficulty heatmap
def cmd_sweep(args):
    plt = _plt()
    df = PE._load(args.results)
    cells = args.cells or sorted(df.cell.unique())
    kinds = [k for k in ("slope", "stair") if PE._sweep_levels(df, k)]
    if not kinds:
        raise SystemExit("no sweep_slope_* / sweep_stair_* terrains in the results")
    train_max = {"slope": 21.8, "stair": 15.0}
    seq = _cmap(["#ffffff"] + BLUE, "seq0")
    from matplotlib.colors import Normalize
    norm = Normalize(0, 100)
    widths = [len(PE._sweep_levels(df, k)) for k in kinds]
    fig, axes = plt.subplots(1, len(kinds), figsize=(args.width, 0.17 * len(cells) + 0.8),
                             squeeze=False, gridspec_kw={"width_ratios": widths, "wspace": 0.45})
    rows_csv = []
    for ax, kind in zip(axes[0], kinds):
        levels = PE._sweep_levels(df, kind)
        xlabel, unit = PE.SWEEP_UNIT[kind]
        vals = [v * unit for v in levels.values()]
        for i, cell in enumerate(cells):
            for j, (t, lv) in enumerate(levels.items()):
                M, _ = AR.matrix(df, cell, t, args.metric)
                v = np.nan if M is None else 100 * float(np.nanmean(np.nanmean(M, axis=1)))
                c = (1, 1, 1, 1) if np.isnan(v) else seq(norm(v))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="white", lw=1.0))
                if not np.isnan(v):
                    ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=5.2, color=_ink_on(c))
                rows_csv.append({"cell": cell, "kind": kind, "level": lv * unit, "success_pct": v})
            mt = PE.max_traversable(df[df.cell == cell], levels, args.metric, args.threshold)
            if mt:
                v = np.array(list(mt.values()), float) * unit
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                ax.text(len(levels) + 0.15, i + 0.5, f"{v.mean():.1f}±{sd:.1f}", ha="left", va="center",
                        fontsize=5.2, color=MUTED)
        # training maximum: boundary between the last trained level and the first beyond it
        k = sum(1 for x in vals if x <= train_max[kind] + 1e-9)
        ax.axvline(k, color=INK, lw=0.8, ls="--")
        ax.text(len(levels) + 0.15, -0.25, "max", ha="left", va="bottom", fontsize=5.2, color=MUTED)
        ax.set_xlim(0, len(levels))
        ax.set_ylim(len(cells), 0)
        ax.set_xticks(np.arange(len(levels)) + 0.5)
        ax.set_xticklabels([f"{x:g}" for x in vals])
        ax.set_xlabel(xlabel)
        ax.set_yticks(np.arange(len(cells)) + 0.5)
        ax.set_yticklabels([_label(c) for c in cells] if ax is axes[0][0] else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm, seq), ax=axes[0].tolist(), fraction=0.025, pad=0.1)
    cb.set_label("Success (%)")
    cb.outline.set_visible(False)
    _stamp(fig, _is_mock(df))
    PE._save(fig, args.out, args.name)
    pd.DataFrame(rows_csv).to_csv(os.path.join(args.out, f"{args.name}.csv"), index=False)


# ----------------------------------------------------------------------------- compute-performance chart
def cmd_pareto(args):
    plt = _plt()
    rng = np.random.default_rng(args.seed)
    df = PE._load(args.results)
    terrains = _terrains(df, args.terrains)
    df = df[df.terrain.isin(terrains)]
    over = {}
    for item in args.nfe or []:
        k, _, v = item.partition("=")
        over[k] = float(v)
    xmeas = {}
    if args.x_csv:
        xc = pd.read_csv(args.x_csv)
        xmeas = dict(zip(xc["cell"], xc["ms"]))
    pts = []
    for cell in args.cells:
        sub = df[df.cell == cell]
        if sub.empty:
            print(f"{cell}: no episodes; skipped")
            continue
        p, lo, hi, n = PE.boot_ci(sub, args.metric, rng, args.n_boot)
        x = xmeas.get(cell, np.nan) if args.x_csv else _nfe(df, cell, over)
        pts.append({"cell": cell, "x": x, "mean": 100 * p, "lo": 100 * lo, "hi": 100 * hi, "n_seeds": n})
    P = pd.DataFrame(pts)
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    fam_color = {"full": SERIES[0], "ppo_only": "#7a7974", "ppo_mlp": SERIES[2], "birodiff": SERIES[3],
                 "diffuseloco": SERIES[4]}
    # fixed-N DDIM modes of one method form a curve, labelled once, with N next to each point
    is_dn = P.cell.str.fullmatch(r"[a-z_]+:D\d+")
    for method, g in P[is_dn].groupby(P[is_dn].cell.str.split(":").str[0]):
        g = g.sort_values("x")
        col = fam_color.get(method, MUTED)
        ax.errorbar(g.x, g["mean"], yerr=[g["mean"] - g.lo, g.hi - g["mean"]], fmt="-", marker=".", ms=4,
                    color=col, lw=0.8, elinewidth=0.6, capsize=0, alpha=0.8)
        for _, r in g.iterrows():
            ax.annotate(f"{int(r.x)}", (r.x, r.lo), xytext=(0, -6), textcoords="offset points", fontsize=5,
                        ha="center", color=MUTED)
        r = g.iloc[0]
        ax.annotate(f"{CELL_NAMES.get(method, method)} D-$N$", (r.x, r["mean"]), xytext=(-4, -1),
                    textcoords="offset points", fontsize=5.5, ha="right", color=INK)
    # every other point: marker + direct label; labels pushed apart vertically where they would collide
    rest = P[~is_dn].sort_values("mean")
    placed = []   # (log10 x, y in points-equivalent units)
    for _, r in rest.iterrows():
        col, mk, _ = STYLE.get(r.cell, (fam_color.get(r.cell.split(":")[0], MUTED), "o", "-"))
        ax.errorbar(r.x, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt=mk, ms=4.5, color=col,
                    mec="white", mew=0.6, elinewidth=0.8, capsize=0, zorder=3)
        lx, y = math.log10(max(r.x, 1e-3)), r["mean"]
        for px, py in placed:
            if abs(px - lx) < 0.35 and abs(py - y) < args.label_gap:
                y = py + args.label_gap
        placed.append((lx, y))
        ax.annotate(_label(r.cell), (r.x, r["mean"]), xytext=(r.x * 1.12, y), textcoords="data", fontsize=5.5,
                    va="center", color=INK, arrowprops=None)
    ok = P.dropna(subset=["x", "mean"]).sort_values(["x", "mean"], ascending=[True, False])
    front, best = [], -np.inf
    for _, r in ok.iterrows():
        if r["mean"] > best:
            front.append((r.x, r["mean"]))
            best = r["mean"]
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color=MUTED, lw=0.7, ls=":", zorder=0)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 60, 100) if P.x.min() * 0.9 <= t <= P.x.max() * 1.1] if not args.x_csv else None
    if ticks:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(P.x.min() * 0.8, P.x.max() * 2.2)
    ax.set_xlabel("Latency per control step (ms)" if args.x_csv else "Network evaluations per control step (NFE)")
    ax.set_ylabel(f"Mean {args.metric} over terrains (%)")
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    _stamp(fig, _is_mock(df))
    PE._save(fig, args.out, args.name)
    P.to_csv(os.path.join(args.out, f"{args.name}.csv"), index=False)
    print(P.to_string(index=False))


# ----------------------------------------------------------------------------- per-seed heatmap
def cmd_seeds(args):
    plt = _plt()
    df = PE._load(args.results)
    terrains = _terrains(df, args.terrains)
    rows, labels, groups = [], [], []
    for cell in args.cells:
        sub = df[df.cell == cell]
        for s in sorted(sub.train_seed.unique()):
            g = sub[sub.train_seed == s]
            rows.append([100 * g[g.terrain == t][args.metric].mean() if (g.terrain == t).any() else np.nan
                         for t in terrains])
            labels.append(f"{_label(cell)} · s{s}")
        groups.append(len(rows))
    V = np.array(rows)
    seq = _cmap(["#ffffff"] + BLUE, "seq0")
    from matplotlib.colors import Normalize
    norm = Normalize(0, 100)
    fig, ax = plt.subplots(figsize=(0.36 * len(terrains) + 1.6, 0.16 * len(rows) + 0.8))
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            c = (1, 1, 1, 1) if np.isnan(V[i, j]) else seq(norm(V[i, j]))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="white", lw=0.8))
            if not np.isnan(V[i, j]):
                ax.text(j + 0.5, i + 0.5, f"{V[i, j]:.0f}", ha="center", va="center", fontsize=5, color=_ink_on(c))
    for gEnd in groups[:-1]:
        ax.axhline(gEnd, color=INK, lw=0.8)
    ax.set_xlim(0, len(terrains))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(terrains)) + 0.5)
    ax.set_xticklabels([TERRAIN_NAMES.get(t, t) for t in terrains], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _stamp(fig, _is_mock(df))
    PE._save(fig, args.out, args.name)
    pd.DataFrame(V, index=labels, columns=terrains).to_csv(os.path.join(args.out, f"{args.name}.csv"))


# ----------------------------------------------------------------------------- speed x difficulty grid
def _grid_values(args):
    """{kind: (levels (display units), speeds, {cell: Z[speed, level] in %})} from --grid speed=dir pairs."""
    pairs = sorted(((float(k), v) for k, _, v in (g.partition("=") for g in args.grid)), key=lambda kv: kv[0])
    frames = {sp: PE._load([d]) for sp, d in pairs}
    out, rows, mock = {}, [], False
    for kind in args.kinds:
        lv = None
        for sp, df in frames.items():
            mock |= _is_mock(df)
            l = PE._sweep_levels(df, kind)
            lv = l if lv is None else {t: v for t, v in lv.items() if t in l}
        if not lv:
            continue
        unit = PE.SWEEP_UNIT[kind][1]
        Z = {}
        for cell in args.cells:
            M = np.full((len(frames), len(lv)), np.nan)
            for i, (sp, df) in enumerate(frames.items()):
                for j, t in enumerate(lv):
                    m, _ = AR.matrix(df, cell, t, args.metric)
                    if m is not None:
                        M[i, j] = 100 * float(np.nanmean(np.nanmean(m, axis=1)))
                        rows.append({"kind": kind, "cell": cell, "speed": sp, "level": lv[t] * unit,
                                     "success_pct": M[i, j], "n_seeds": m.shape[0]})
            Z[cell] = M
        out[kind] = ([v * unit for v in lv.values()], list(frames), Z)
    return out, rows, mock


def cmd_grid(args):
    plt = _plt()
    data, rows, mock = _grid_values(args)
    if not data:
        raise SystemExit("no sweep levels common to all --grid directories")
    train_max = {"slope": 21.8, "stair": 15.0}
    names = {"slope": "Slope angle (deg)", "stair": "Stair rise (cm)"}
    fam_color = {"full:AD": SERIES[0], "full:SP": SERIES[1], "ppo_mlp:ORIG": SERIES[2], "birodiff:ORIG": SERIES[3],
                 "diffuseloco:ORIG": SERIES[4]}
    if args.style == "bars3d":
        from matplotlib.patches import Patch
        fig = plt.figure(figsize=(args.width, 2.5))
        k = len(args.cells)
        for a, (kind, (levels, speeds, Z)) in enumerate(data.items()):
            ax = fig.add_subplot(1, len(data), a + 1, projection="3d")
            ax.computed_zorder = False   # painter's order set below: bars far from the camera are drawn first
            dx = 0.8 / k
            cam = np.array([math.cos(math.radians(args.azim)), math.sin(math.radians(args.azim))])
            bars = []
            for c, cell in enumerate(args.cells):
                for i in range(len(speeds)):
                    for j in range(len(levels)):
                        if not np.isnan(Z[cell][i, j]):
                            x0 = j - 0.4 + c * dx
                            bars.append((float(np.dot(cam, [x0 + dx / 2, i])), x0, i, Z[cell][i, j], cell))
            for _, x0, i, z, cell in sorted(bars):
                ax.bar3d(x0, i - 0.25, 0, dx * 0.9, 0.5, z, color=fam_color.get(cell, MUTED), shade=True,
                         edgecolor="none")
            kb = sum(1 for x in levels if x <= train_max[kind] + 1e-9) - 0.5
            ax.plot([kb, kb], [-0.5, len(speeds) - 0.5], [0, 0], color=INK, lw=0.9, ls="--")
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([f"{x:g}" for x in levels], fontsize=5.5)
            ax.set_yticks(range(len(speeds)))
            ax.set_yticklabels([f"{v:g}" for v in speeds], fontsize=5.5)
            ax.set_zlim(0, 100)
            ax.set_zticks([0, 50, 100])
            ax.tick_params(axis="z", labelsize=5.5, pad=0)
            ax.tick_params(axis="x", pad=-3)
            ax.tick_params(axis="y", pad=-3)
            ax.set_xlabel(names[kind], fontsize=6, labelpad=-6)
            ax.set_ylabel("Command (m/s)", fontsize=6, labelpad=-6)
            ax.set_zlabel("Success (%)", fontsize=6, labelpad=-7)
            ax.view_init(elev=args.elev, azim=args.azim)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_facecolor((0.96, 0.96, 0.95, 1.0))
                pane.set_edgecolor("#e4e3df")
        fig.legend(handles=[Patch(color=fam_color.get(c, MUTED), label=_label(c)) for c in args.cells],
                   loc="upper center", ncol=len(args.cells), frameon=False, fontsize=6)
        fig.subplots_adjust(left=0.0, right=0.98, bottom=0.05, top=0.9, wspace=0.05)
    else:   # small-multiple heatmaps: exact values, no occlusion
        seq = _cmap(["#ffffff"] + BLUE, "seq0")
        from matplotlib.colors import Normalize
        norm = Normalize(0, 100)
        kinds = list(data)
        widths = [len(data[kd][0]) for kd in kinds]
        fig, axes = plt.subplots(len(args.cells), len(kinds), figsize=(args.width, 0.55 * len(args.cells) + 0.6),
                                 squeeze=False, gridspec_kw={"width_ratios": widths, "wspace": 0.15, "hspace": 0.35})
        for r, cell in enumerate(args.cells):
            for cidx, kind in enumerate(kinds):
                levels, speeds, Z = data[kind]
                ax, M = axes[r][cidx], Z[cell]
                for i in range(len(speeds)):
                    for j in range(len(levels)):
                        v = M[i, j]
                        col = (1, 1, 1, 1) if np.isnan(v) else seq(norm(v))
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col, edgecolor="white", lw=0.8))
                        if not np.isnan(v):
                            ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=5,
                                    color=_ink_on(col))
                kb = sum(1 for x in levels if x <= train_max[kind] + 1e-9)
                ax.axvline(kb, color=INK, lw=0.8, ls="--")
                ax.set_xlim(0, len(levels)); ax.set_ylim(0, len(speeds))
                ax.set_xticks(np.arange(len(levels)) + 0.5)
                ax.set_xticklabels([f"{x:g}" for x in levels] if r == len(args.cells) - 1 else [])
                ax.set_yticks(np.arange(len(speeds)) + 0.5)
                ax.set_yticklabels([f"{v:g}" for v in speeds] if cidx == 0 else [])
                ax.tick_params(length=0)
                for sp_ in ax.spines.values():
                    sp_.set_visible(False)
                if cidx == 0:
                    ax.set_ylabel(_label(cell), fontsize=6)
                if r == len(args.cells) - 1:
                    ax.set_xlabel(names[kind])
    _stamp(fig, mock)
    PE._save(fig, args.out, f"{args.name}_{args.style}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{args.name}.csv"), index=False)


# ----------------------------------------------------------------------------- per-seed table
def cmd_seedtable(args):
    """LaTeX rows: controller & s1 ... sK & mean [95% CI]; suite mean of every training run (terrains weighted equally)."""
    rng = np.random.default_rng(args.seed)
    df = PE._load(args.results)
    terrains = _terrains(df, args.terrains)
    df = df[df.terrain.isin(terrains)]
    seeds = sorted(df.train_seed.unique())
    lines, rows = [], []
    for cell in args.cells:
        sub = df[df.cell == cell]
        if sub.empty:
            print(f"{cell}: no episodes; skipped")
            continue
        per = {}
        for sd, g in sub.groupby("train_seed"):
            if g.terrain.nunique() == len(terrains):   # a run counts only with every terrain evaluated
                per[sd] = 100 * g.groupby("terrain")[args.metric].mean().mean()
        p, lo, hi, n = PE.boot_ci(sub, args.metric, rng, args.n_boot)
        vals = [f"{per[sd]:.1f}" if sd in per else "--" for sd in seeds]
        ci = f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]" if n > 1 else f"{100 * p:.1f}"
        lines.append(f"{_label(cell)} & " + " & ".join(vals) + f" & {ci} \\\\")
        rows.append({"cell": cell, **{f"s{sd}": per.get(sd, np.nan) for sd in seeds}, "mean": 100 * p,
                     "ci_lo": 100 * lo, "ci_hi": 100 * hi, "n_seeds": n})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.tex"), "w") as fh:
        fh.write(f"% {args.metric} (%) per training seed, mean over {len(terrains)} terrains; last column: mean over "
                 "seeds [95% two-level bootstrap CI]\n" + "\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{args.name}.csv"), index=False)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, name, metric="success"):
        p.add_argument("--results", nargs="+", required=True)
        p.add_argument("--out", default="figs_ral")
        p.add_argument("--name", default=name)
        p.add_argument("--metric", default=metric)
        p.add_argument("--terrains", nargs="+", default=DEFAULT_TERRAINS)
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--seed", type=int, default=0)

    p = sp.add_parser("modes"); common(p, "heatmap_modes")
    p.add_argument("--methods", nargs="+", default=["ppo_only", "full"])
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    p.add_argument("--ref", default="SP")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--vmax", type=float, default=None, help="colour limit of the difference (pp)")
    p.add_argument("--width", type=float, default=7.16, help="figure width in inches (7.16 = IEEE two columns)")
    p = sp.add_parser("steps"); common(p, "heatmap_steps")
    p.add_argument("--methods", nargs="+", default=["full", "full_tnppriv", "full_tnpuntrained"])
    p.add_argument("--token", default="AD")
    p.add_argument("--n-min", type=int, default=5)
    p.add_argument("--n-max", type=int, default=10)
    p.add_argument("--mock", action="store_true", help="stamp the figure as mock data")
    p.add_argument("--width", type=float, default=3.45, help="figure width in inches (3.45 = one IEEE column)")
    p = sp.add_parser("sweep"); common(p, "heatmap_difficulty")
    p.add_argument("--cells", nargs="+", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--width", type=float, default=3.45)
    p = sp.add_parser("pareto"); common(p, "compute_success")
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--nfe", nargs="*", help="cell=value overrides of the network evaluations per step")
    p.add_argument("--x-csv", default=None, help="CSV with columns cell,ms: plot measured latency instead of NFE")
    p.add_argument("--label-gap", type=float, default=2.2, help="minimum vertical gap between labels (y units)")
    p = sp.add_parser("grid"); common(p, "grid_speed_difficulty")
    p.add_argument("--grid", nargs="+", required=True, help="speed=results_dir pairs, e.g. 0.3=results_grid/v0.3 "
                   "0.5=results_sweep 0.8=results_grid/v0.8")
    p.add_argument("--cells", nargs="+", default=["ppo_mlp:ORIG", "full:SP", "full:AD"])
    p.add_argument("--kinds", nargs="+", default=["slope", "stair"])
    p.add_argument("--style", choices=["bars3d", "heatmaps"], default="bars3d")
    p.add_argument("--width", type=float, default=7.16)
    p.add_argument("--elev", type=float, default=32)
    p.add_argument("--azim", type=float, default=-58)
    p = sp.add_parser("seedtable"); common(p, "table_seeds")
    p.add_argument("--cells", nargs="+", required=True)
    p = sp.add_parser("seeds"); common(p, "heatmap_seeds")
    p.add_argument("--cells", nargs="+", required=True)
    args = ap.parse_args()
    {"modes": cmd_modes, "steps": cmd_steps, "sweep": cmd_sweep, "pareto": cmd_pareto, "seeds": cmd_seeds,
     "grid": cmd_grid, "seedtable": cmd_seedtable}[args.cmd](args)


if __name__ == "__main__":
    main()
