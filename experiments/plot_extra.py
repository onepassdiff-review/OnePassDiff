#!/usr/bin/env python3
"""Figures and tables for the further experiments.

Seeds are the unit of analysis, as in analyze_results.py: intervals are 95% two-level bootstrap CIs
(training seeds, then episodes within a seed) from analyze_results.ci, except for the training curves,
which have one value per seed and iteration and use a t-interval over seeds.

Subcommands
  training  --scalars <export_scalars out>/scalars [--variants full ppo_only ppo_mlp]
            training reward and curriculum level vs iteration: every seed (thin) and the mean with a
            95% t-interval over seeds (band), moving average over --smooth iterations
  curve     --results results_curve/
            success on a fixed evaluation set vs training iteration (run_e0 --curve-checkpoints)
  speed     --results results/ results_speed/v0.3 results_speed/v0.8 ...
            success and velocity-tracking RMSE vs commanded speed (one speed per directory)
  sweep     --results results_sweep/ [--threshold 0.5]
            success vs slope angle and stair rise; the largest sweep value a seed traverses, i.e. the
            hardest level up to which every level has success >= threshold
  traj      --results results/ --cells full:SP full:AD ppo_mlp:ORIG --seed 1 --terrain stair_up [--robot 0]
            base path (top and side view) and forward velocity vs time, velocity RMSE in the legend
  rmse      --results results/ --cells full:AD full:SP ppo_mlp:ORIG ...
            velocity-tracking RMSE per terrain and controller (bar chart with CIs)

Every subcommand writes a PDF and PNG figure and a CSV of the plotted values to --out; speed and sweep also
write LaTeX rows (mean [lo, hi]). Display names: --names full:AD="Ours (AD)" ...
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

VARIANT_NAMES = {"full": "OnePassDiff", "ppo_only": "OnePassDiff-PPO-only", "ppo_mlp": "PPO-MLP",
                 "birodiff": "BiRoDiff-R", "diffuseloco": "DiffuseLoco-R"}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
SWEEP_RE = {"slope": re.compile(r"sweep_slope_(\d+)$"), "stair": re.compile(r"sweep_stair_(\d+)$")}
SWEEP_UNIT = {"slope": ("Slope angle (deg)", 1.0), "stair": ("Stair rise (cm)", 0.1)}   # stair: mm -> cm


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    print(f"figure -> {os.path.join(out, name)}.pdf")


def _names(args):
    d = {}
    for item in args.names or []:
        k, _, v = item.partition("=")
        d[k] = v
    return d


def _label(cell, names):
    if cell in names:
        return names[cell]
    m, _, mode = cell.partition(":")
    base = VARIANT_NAMES.get(m, m)
    return base if mode in ("ORIG", "") else f"{base} ({mode})"


def _load(dirs, allow_multi_ck=False):
    frames = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "episodes_*.csv")))
        if not files:
            raise SystemExit(f"no episodes_*.csv in {d}")
        frames += [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["cell"] = df["method"] + ":" + df["mode"]
    key = ["method", "mode", "terrain", "train_seed", "episode"]
    if allow_multi_ck:
        key.append("checkpoint")
    if "cmd_speed" in df:
        df["speed"] = df["cmd_speed"].round(2)
        key.append("speed")
    return df.drop_duplicates(subset=key, keep="last")


def seed_matrix(sub, metric):
    """[n_seeds, n_eps] for one cell, episodes of all terrains in `sub` pooled per seed (terrain order fixed,
    equal episode counts per terrain give equal terrain weights), truncated to the shortest seed."""
    rows, seeds = [], []
    for s, g in sub.groupby("train_seed"):
        g = g.sort_values(["terrain", "episode"])
        rows.append(g[metric].to_numpy(float))
        seeds.append(s)
    if not rows:
        return None, []
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows]), seeds


def boot_ci(sub, metric, rng, n_boot):
    M, seeds = seed_matrix(sub, metric)
    if M is None:
        return np.nan, np.nan, np.nan, 0
    p, lo, hi = AR.ci(M, rng, n_boot)
    return p, lo, hi, len(seeds)


def t_band(Y):
    """Y [n_seeds, T] -> mean, half-width of the 95% t-interval over seeds."""
    n = Y.shape[0]
    m = np.nanmean(Y, axis=0)
    if n < 2:
        return m, np.zeros_like(m)
    return m, T95.get(n - 1, 1.96) * np.nanstd(Y, axis=0, ddof=1) / math.sqrt(n)


# ----------------------------------------------------------------------------- training curves
def cmd_training(args):
    plt = _plt()
    runs = {}
    for f in sorted(glob.glob(os.path.join(args.scalars, "*.csv"))):
        m = re.match(r"onepass_(.+)_seed(\d+)$", os.path.splitext(os.path.basename(f))[0])
        if m and (not args.variants or m.group(1) in args.variants):
            runs.setdefault(m.group(1), {})[int(m.group(2))] = pd.read_csv(f)
    if not runs:
        raise SystemExit(f"no onepass_<variant>_seed<s>.csv under {args.scalars}")
    variants = [v for v in (args.variants or sorted(runs)) if v in runs]
    tags = [("rewards/iter", "Training reward"), ("Episode/terrain_level", "Curriculum level")]
    grid = np.arange(1, args.iters + 1)
    fig, axes = plt.subplots(1, len(tags), figsize=(7.0, 2.3))
    rows = []
    for ax, (tag, ylabel) in zip(axes, tags):
        for i, v in enumerate(variants):
            Y = []
            for s, d in sorted(runs[v].items()):
                d = d[d.tag == tag].sort_values("step")
                if d.empty:
                    continue
                y = np.interp(grid, d.step.to_numpy(float), d.value.to_numpy(float), left=np.nan, right=np.nan)
                if args.smooth > 1:
                    y = pd.Series(y).rolling(args.smooth, min_periods=1).mean().to_numpy()
                Y.append(y)
                ax.plot(grid, y, color=f"C{i}", lw=0.4, alpha=0.35)
                rows += [{"tag": tag, "variant": v, "seed": s, "iteration": int(it), "value": float(val)}
                         for it, val in zip(grid[::50], y[::50])]
            if not Y:
                continue
            Y = np.stack(Y)
            m, h = t_band(Y)
            ax.plot(grid, m, color=f"C{i}", lw=1.2, label=f"{VARIANT_NAMES.get(v, v)} ({Y.shape[0]} seeds)")
            ax.fill_between(grid, m - h, m + h, color=f"C{i}", alpha=0.2, lw=0)
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, loc="lower right")
    _save(fig, args.out, "training_curves")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "training_curves.csv"), index=False)
    # final values per seed (last 50 iterations): for the per-seed table
    fin = []
    for v in variants:
        for s, d in sorted(runs[v].items()):
            rec = {"variant": v, "seed": s}
            for tag, _ in tags:
                dd = d[(d.tag == tag) & (d.step > args.iters - 50)]
                rec[tag] = float(dd.value.mean()) if len(dd) else np.nan
            fin.append(rec)
    pd.DataFrame(fin).to_csv(os.path.join(args.out, "training_final_per_seed.csv"), index=False)
    print(pd.DataFrame(fin).to_string(index=False))


# ----------------------------------------------------------------------------- learning curve (fixed eval set)
def cmd_curve(args):
    plt = _plt()
    names = _names(args)
    rng = np.random.default_rng(args.seed)
    df = _load(args.results, allow_multi_ck=True)
    it = df["checkpoint"].str.extract(r"_ep_(\d+)_rew")[0]
    df = df[it.notna()].copy()
    df["iteration"] = it.dropna().astype(int)
    if args.terrains:
        df = df[df.terrain.isin(args.terrains)]
    cells = args.cells or sorted(df.cell.unique())
    rows = []
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, cell in enumerate(cells):
        sub = df[df.cell == cell]
        pts = []
        for k, g in sub.groupby("iteration"):
            p, lo, hi, n = boot_ci(g, args.metric, rng, args.n_boot)
            pts.append((k, p, lo, hi, n))
            rows.append({"cell": cell, "iteration": k, "mean": p, "ci_lo": lo, "ci_hi": hi, "n_seeds": n})
        if not pts:
            continue
        k, p, lo, hi, n = map(np.array, zip(*pts))
        sc = AR.METRICS_PCT.get(args.metric, 1.0)
        ax.plot(k, p * sc, "-o", ms=2, color=f"C{i}", label=_label(cell, names))
        ax.fill_between(k, lo * sc, hi * sc, color=f"C{i}", alpha=0.2, lw=0)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel({"success": "Success on fixed set (%)", "progress": "Progress (%)"}.get(args.metric, args.metric))
    ax.legend(frameon=False)
    _save(fig, args.out, f"curve_{args.metric}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"curve_{args.metric}.csv"), index=False)


# ----------------------------------------------------------------------------- speed sweep
def cmd_speed(args):
    plt = _plt()
    names = _names(args)
    rng = np.random.default_rng(args.seed)
    df = _load(args.results)
    if "speed" not in df:
        raise SystemExit("no cmd_speed column: re-run with the current run_e0.py")
    if args.terrains:
        df = df[df.terrain.isin(args.terrains)]
    cells = args.cells or sorted(df.cell.unique())
    speeds = sorted(df.speed.unique())
    metrics = [("success", "Success (%)", 100.0), ("vel_rmse", "Velocity RMSE (m/s)", 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    rows, tex = [], []
    for i, cell in enumerate(cells):
        cells_tex = []
        for ax, (met, ylabel, sc) in zip(axes, metrics):
            pts = []
            for v in speeds:
                p, lo, hi, n = boot_ci(df[(df.cell == cell) & (df.speed == v)], met, rng, args.n_boot)
                pts.append((v, p, lo, hi))
                rows.append({"cell": cell, "metric": met, "speed": v, "mean": p * sc, "ci_lo": lo * sc,
                             "ci_hi": hi * sc, "n_seeds": n})
            v_, p, lo, hi = map(np.array, zip(*pts))
            ok = ~np.isnan(p)
            ax.errorbar(v_[ok] + 0.006 * (i - len(cells) / 2), p[ok] * sc,
                        yerr=[(p - lo)[ok] * sc, (hi - p)[ok] * sc], fmt="-o", ms=2.5, capsize=2, lw=1,
                        color=f"C{i}", label=_label(cell, names))
            ax.set_xlabel("Commanded speed (m/s)")
            ax.set_ylabel(ylabel)
        for v in speeds:
            for met, _, sc in metrics:
                r = [x for x in rows if x["cell"] == cell and x["metric"] == met and x["speed"] == v][0]
                d = 1 if met == "success" else 3
                cells_tex.append("--" if np.isnan(r["mean"]) else f"{r['mean']:.{d}f}")
        tex.append(f"{_label(cell, names)} & " + " & ".join(cells_tex) + r" \\")
    axes[0].legend(frameon=False)
    _save(fig, args.out, "speed_sweep")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "speed_sweep.csv"), index=False)
    with open(os.path.join(args.out, "table_speed.tex"), "w") as fh:
        fh.write(f"% columns per speed {speeds}: success (%) & vel. RMSE (m/s); CIs in speed_sweep.csv\n")
        fh.write("\n".join(tex) + "\n")


# ----------------------------------------------------------------------------- difficulty sweep
def _sweep_levels(df, kind):
    lv = {}
    for t in df.terrain.unique():
        m = SWEEP_RE[kind].match(t)
        if m:
            lv[t] = int(m.group(1))
    return dict(sorted(lv.items(), key=lambda kv: kv[1]))


def max_traversable(sub, levels, metric, threshold):
    """per seed: the hardest level up to which every level reaches `threshold` (0 if the easiest fails)."""
    out = {}
    for s, g in sub.groupby("train_seed"):
        best = 0
        for t, val in levels.items():
            gt = g[g.terrain == t]
            if gt.empty or gt[metric].mean() < threshold:
                break
            best = val
        out[s] = best
    return out


def cmd_sweep(args):
    plt = _plt()
    names = _names(args)
    rng = np.random.default_rng(args.seed)
    df = _load(args.results)
    cells = args.cells or sorted(df.cell.unique())
    kinds = [k for k in ("slope", "stair") if _sweep_levels(df, k)]
    if not kinds:
        raise SystemExit("no sweep_slope_* / sweep_stair_* terrains in the results")
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.5 * len(kinds), 2.3), squeeze=False)
    rows, mt_rows = [], []
    for ax, kind in zip(axes[0], kinds):
        levels = _sweep_levels(df, kind)
        xlabel, unit = SWEEP_UNIT[kind]
        for i, cell in enumerate(cells):
            pts = []
            for t, val in levels.items():
                p, lo, hi, n = boot_ci(df[(df.cell == cell) & (df.terrain == t)], args.metric, rng, args.n_boot)
                pts.append((val * unit, p, lo, hi))
                rows.append({"cell": cell, "kind": kind, "level": val, "mean": p, "ci_lo": lo, "ci_hi": hi,
                             "n_seeds": n})
            x, p, lo, hi = map(np.array, zip(*pts))
            ok = ~np.isnan(p)
            if ok.any():
                ax.plot(x[ok], 100 * p[ok], "-o", ms=2.5, lw=1, color=f"C{i}", label=_label(cell, names))
                ax.fill_between(x[ok], 100 * lo[ok], 100 * hi[ok], color=f"C{i}", alpha=0.15, lw=0)
            mt = max_traversable(df[df.cell == cell], levels, args.metric, args.threshold)
            if mt:
                vals = np.array(list(mt.values()), float) * unit
                mt_rows.append({"cell": cell, "kind": kind, "mean": vals.mean(),
                                "sd": vals.std(ddof=1) if len(vals) > 1 else 0.0, "n_seeds": len(vals),
                                "per_seed": ";".join(f"{v:g}" for v in vals)})
        if kind == "slope":
            ax.axvline(21.8, color="0.5", lw=0.6, ls="--")
        else:
            ax.axvline(15.0, color="0.5", lw=0.6, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Success (%)")
        ax.set_ylim(-3, 103)
    axes[0][0].legend(frameon=False, loc="lower left")
    _save(fig, args.out, "difficulty_sweep")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "difficulty_sweep.csv"), index=False)
    mt = pd.DataFrame(mt_rows)
    mt.to_csv(os.path.join(args.out, "max_traversable.csv"), index=False)
    tex = []
    for cell in cells:
        vals = []
        for kind in kinds:
            r = mt[(mt.cell == cell) & (mt.kind == kind)] if len(mt) else mt
            vals.append("--" if r.empty else f"{r['mean'].iloc[0]:.1f} $\\pm$ {r['sd'].iloc[0]:.1f}")
        tex.append(f"{_label(cell, names)} & " + " & ".join(vals) + r" \\")
    with open(os.path.join(args.out, "table_max_traversable.tex"), "w") as fh:
        fh.write(f"% columns: {' | '.join(kinds)} (slope in deg, stair rise in cm); mean +- sd over seeds; "
                 f"threshold {args.metric} >= {args.threshold}\n" + "\n".join(tex) + "\n")
    print(mt.to_string(index=False))


# ----------------------------------------------------------------------------- trajectories
def cmd_traj(args):
    import torch
    plt = _plt()
    names = _names(args)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    rows = []
    for i, cell in enumerate(args.cells):
        method, _, mode = cell.partition(":")
        f = os.path.join(args.results[0], f"traj_{method}_seed{args.seed}_{mode}_{args.terrain}.pt")
        if not os.path.exists(f):
            print(f"missing {f}")
            continue
        d = torch.load(f, map_location="cpu")
        r = args.robot
        alive = d["alive"][:, r].numpy().astype(bool)
        T = int(alive.sum()) if not alive.all() else len(alive)
        t = np.arange(T) * float(d["dt"])
        pos = d["pos"][:T, r].numpy()
        vx = d["vel"][:T, r, 0].numpy()
        cmd = d["cmd"][:T, r, 0].numpy()
        rmse = float(np.sqrt(np.mean(((d["vel"][:T, r] - d["cmd"][:T, r]) ** 2).sum(-1).numpy())))
        fell = not alive.all()
        lab = f"{_label(cell, names)}: RMSE {rmse:.3f}" + (" (fell)" if fell else "")
        axes[0].plot(pos[:, 0] - pos[0, 0], pos[:, 1] - pos[0, 1], color=f"C{i}", lw=0.9, label=lab)
        axes[1].plot(pos[:, 0] - pos[0, 0], pos[:, 2], color=f"C{i}", lw=0.9)
        k = max(1, int(round(args.smooth_s / float(d["dt"]))))
        axes[2].plot(t, pd.Series(vx).rolling(k, min_periods=1).mean(), color=f"C{i}", lw=0.8)
        if i == 0:
            axes[2].plot(t, cmd, color="k", lw=0.8, ls="--", label="command")
        rows.append({"cell": cell, "terrain": args.terrain, "seed": args.seed, "robot": r, "seconds": T * float(d["dt"]),
                     "fell": fell, "vel_rmse": rmse, "distance_x": float(pos[-1, 0] - pos[0, 0])})
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("Base height (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Forward velocity (m/s)")   # moving average over --smooth-s (state it in the caption)
    fig.subplots_adjust(wspace=0.42)
    h, lab = [], []
    for ax in (axes[0], axes[2]):
        hh, ll = ax.get_legend_handles_labels()
        h += hh
        lab += ll
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=min(3, len(lab)), frameon=False, fontsize=6)
    _save(fig, args.out, f"traj_{args.terrain}_seed{args.seed}_robot{args.robot}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"traj_{args.terrain}_seed{args.seed}.csv"), index=False)


# ----------------------------------------------------------------------------- velocity RMSE bars
def cmd_rmse(args):
    plt = _plt()
    names = _names(args)
    rng = np.random.default_rng(args.seed)
    df = _load(args.results)
    terrains = args.terrains or sorted(df.terrain.unique())
    cells = args.cells or sorted(df.cell.unique())
    w = 0.8 / max(1, len(cells))
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    rows = []
    for i, cell in enumerate(cells):
        for j, t in enumerate(terrains):
            p, lo, hi, n = boot_ci(df[(df.cell == cell) & (df.terrain == t)], args.metric, rng, args.n_boot)
            rows.append({"cell": cell, "terrain": t, "mean": p, "ci_lo": lo, "ci_hi": hi, "n_seeds": n})
            if not np.isnan(p):
                yerr = None if np.isnan(lo) else [[p - lo], [hi - p]]
                ax.bar(j + (i - (len(cells) - 1) / 2) * w, p, w, yerr=yerr, capsize=1.5, color=f"C{i}",
                       label=_label(cell, names) if j == 0 else None, error_kw={"lw": 0.6})
    ax.set_xticks(range(len(terrains)))
    ax.set_xticklabels(terrains, rotation=20)
    ax.set_ylabel({"vel_rmse": "Velocity RMSE (m/s)"}.get(args.metric, args.metric))
    ax.legend(frameon=False, ncol=3, fontsize=6)
    _save(fig, args.out, f"bars_{args.metric}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"bars_{args.metric}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, results=True):
        if results:
            p.add_argument("--results", nargs="+", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--cells", nargs="*", default=None, help="method:mode cells (default: all)")
        p.add_argument("--names", nargs="*", default=None, help='display names, e.g. full:AD="Ours (AD)"')
        p.add_argument("--terrains", nargs="*", default=None)
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--seed", type=int, default=0, help="bootstrap seed (traj: training seed)")

    p = sp.add_parser("training")
    common(p, results=False)
    p.add_argument("--scalars", required=True)
    p.add_argument("--variants", nargs="*", default=["full", "ppo_only", "ppo_mlp"])
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--smooth", type=int, default=50)
    p = sp.add_parser("curve")
    common(p)
    p.add_argument("--metric", default="success")
    p = sp.add_parser("speed")
    common(p)
    p = sp.add_parser("sweep")
    common(p)
    p.add_argument("--metric", default="success")
    p.add_argument("--threshold", type=float, default=0.5)
    p = sp.add_parser("traj")
    common(p)
    p.add_argument("--terrain", required=True)
    p.add_argument("--robot", type=int, default=0)
    p.add_argument("--smooth-s", type=float, default=0.8, help="moving average of the velocity plot [s] (gait period)")
    p = sp.add_parser("rmse")
    common(p)
    p.add_argument("--metric", default="vel_rmse")
    args = ap.parse_args()
    {"training": cmd_training, "curve": cmd_curve, "speed": cmd_speed, "sweep": cmd_sweep,
     "traj": cmd_traj, "rmse": cmd_rmse}[args.cmd](args)


if __name__ == "__main__":
    main()
