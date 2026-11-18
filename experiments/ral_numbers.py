#!/usr/bin/env python3
"""Every number of the Results section, from the files run_e0.py and export_scalars.py write.

Statistics as in analyze_results.py / Sec. III of the paper: the training seed is the unit of analysis; a mean over
the suite weights the ten terrains equally (per seed: mean over terrains of the per-terrain episode means, then mean
over seeds); 95% CIs from a two-level bootstrap (resample seeds, then episodes within seed and terrain); cells that share
checkpoints and episodes (inference modes of one method, the step-predictor heads evaluated on the full checkpoints)
are compared pairwise episode by episode, other controllers independently; Holm correction across the terrains of
one comparison. A seed counts for a metric only if that metric was logged on every terrain of the comparison (the
counts used are written next to every value).

Usage (from a directory holding analyze_results.py and plot_extra.py, CPU only):
  python ral_numbers.py --results RESULTS --curve RESULTS_CURVE --sweep RESULTS_SWEEP --scalars SCALARS_DIR --out OUT
Writes OUT/ral_numbers.json and the figures OUT/fig_train.{pdf,png}, OUT/fig_traj.{pdf,png} (+ CSVs of plotted values).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_results as AR  # noqa: E402
import plot_extra as PE  # noqa: E402

T10 = ["flat", "rough", "slope10", "slope18", "stair_up", "stair_down", "discrete",
       "heldout_slope", "heldout_stairs", "heldout_gaps"]
PCT = {"success", "spl", "fall", "collision", "progress"}
B = 2000
RNG = np.random.default_rng(0)
NAMES = {"full": "OnePassDiff", "ppo_only": "OnePassDiff-PPO-only", "ppo_mlp": "PPO-MLP", "birodiff": "BiRoDiff-R",
         "diffuseloco": "DiffuseLoco-R"}


# ----------------------------------------------------------------------------- bootstrap core
def _groups_from(frame, col, terrains, require_all=True):
    g = {}
    for (s, t), x in frame.groupby(["train_seed", "terrain"]):
        v = x[col].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v):
            g.setdefault(int(s), {})[t] = v
    if require_all:
        g = {s: d for s, d in g.items() if all(t in d for t in terrains)}
    return g


def _boot(groups):
    """point estimate, bootstrap distribution [B] (None if < 2 seeds), per-seed means."""
    seeds = sorted(groups)
    if not seeds:
        return np.nan, None, {}
    seedb, per = [], {}
    for s in seeds:
        tb, tp = [], []
        for t, v in groups[s].items():
            idx = RNG.integers(0, len(v), (B, len(v)))
            tb.append(v[idx].mean(1))
            tp.append(v.mean())
        seedb.append(np.mean(tb, 0))
        per[s] = float(np.mean(tp))
    seedb = np.stack(seedb)                       # [S, B]
    point = float(np.mean(list(per.values())))
    if len(seeds) < 2:
        return point, None, per
    sidx = RNG.integers(0, len(seeds), (B, len(seeds)))
    boot = seedb[sidx, np.arange(B)[:, None]].mean(1)
    return point, boot, per


def _scale(metric):
    return 100.0 if metric in PCT else 1.0


def stat(df, cell, terrains, metric, require_all=True):
    sub = df[(df.cell == cell) & df.terrain.isin(terrains)]
    if sub.empty or metric not in sub:
        return None
    g = _groups_from(sub, metric, terrains, require_all)
    p, boot, per = _boot(g)
    if not per:
        return None
    k = _scale(metric)
    lo, hi = (np.nan, np.nan) if boot is None else (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    return {"mean": k * p, "lo": k * lo, "hi": k * hi, "n_seeds": len(per),
            "per_seed": {str(s): k * v for s, v in per.items()}, "_boot": None if boot is None else k * boot}


def _pval(bd):
    p = 2 * min(np.mean(bd <= 0), np.mean(bd >= 0))
    return float(min(1.0, max(p, 1.0 / B)))


def paired(df, a, b, terrains, metric):
    """a - b on the same seeds, terrains and episodes."""
    key = ["train_seed", "terrain", "episode"]
    A = df[(df.cell == a) & df.terrain.isin(terrains)][key + [metric]]
    Bf = df[(df.cell == b) & df.terrain.isin(terrains)][key + [metric]]
    m = A.merge(Bf, on=key, suffixes=("_a", "_b")).dropna()
    if m.empty:
        return None
    m["d"] = m[f"{metric}_a"] - m[f"{metric}_b"]
    g = _groups_from(m, "d", terrains)
    p, boot, per = _boot(g)
    if not per or boot is None:
        return None
    k = _scale(metric)
    return {"diff": k * p, "lo": k * np.percentile(boot, 2.5), "hi": k * np.percentile(boot, 97.5),
            "p": _pval(boot), "n_seeds": len(per), "paired": True}


def independent(df, a, b, terrains, metric):
    sa, sb = stat(df, a, terrains, metric), stat(df, b, terrains, metric)
    if sa is None or sb is None or sa["_boot"] is None or sb["_boot"] is None:
        return None
    bd = sa["_boot"] - sb["_boot"]
    return {"diff": sa["mean"] - sb["mean"], "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
            "p": _pval(bd), "n_seeds": [sa["n_seeds"], sb["n_seeds"]], "paired": False}


def per_terrain(df, a, b, metric, pair):
    fn = paired if pair else independent
    res = {t: fn(df, a, b, [t], metric) for t in T10}
    ok = [t for t in T10 if res[t] is not None]
    if ok:
        for t, pa in zip(ok, AR.holm(np.array([res[t]["p"] for t in ok]))):
            res[t]["p_holm"] = float(pa)
    return {t: r for t, r in res.items() if r is not None}


def clean(x):
    """drop bootstrap arrays, make JSON-safe."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items() if k != "_boot"}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


COL = {"full": "#2a78d6", "ppo_only": "#eb6834", "ppo_mlp": "#1baf7a", "birodiff": "#eda100", "diffuseloco": "#e87ba4"}
LS = {"full": "-", "ppo_only": "--", "ppo_mlp": "-.", "birodiff": ":", "diffuseloco": (0, (5, 1, 1, 1))}


def fig_train(scalars, curve_df, out, iters=5000, smooth=50):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.45), gridspec_kw={"wspace": 0.42})
    grid = np.arange(1, iters + 1)
    rows, final = [], []
    for v in ("ppo_mlp", "ppo_only", "full"):
        Y = []
        for f in sorted(glob.glob(os.path.join(scalars, f"onepass_{v}_seed*.csv"))):
            s = int(re.search(r"_seed(\d+)\.csv$", f).group(1))
            d = pd.read_csv(f)
            d = d[d.tag == "rewards/iter"].sort_values("step")
            if d.empty:
                continue
            y = np.interp(grid, d.step.to_numpy(float), d.value.to_numpy(float), left=np.nan, right=np.nan)
            y = pd.Series(y).rolling(smooth, min_periods=1).mean().to_numpy()
            Y.append(y)
            a1.plot(grid, y, color=COL[v], lw=0.35, alpha=0.45, ls="-")
            last = d[d.step > iters - 50].value
            final.append({"variant": v, "seed": s, "reward_last50": float(last.mean()) if len(last) else None,
                          "last_step": int(d.step.max())})
            rows += [{"variant": v, "seed": s, "iteration": int(i), "reward": float(r)}
                     for i, r in zip(grid[::100], y[::100])]
        if Y:
            m = np.nanmean(np.stack(Y), 0)
            a1.plot(grid, m, color=COL[v], lw=1.1, ls=LS[v], label=f"{NAMES[v]} ({len(Y)})")
    a1.set_xlabel("PPO iteration")
    a1.set_ylabel("Training reward")
    a1.set_xticks([0, 2500, 5000])
    a1.legend(frameon=False, loc="lower right", fontsize=5, handlelength=2.2)
    a1.set_title("(a)", loc="left", fontsize=7)
    # (b) fixed-set success of saved checkpoints (seeds with every checkpoint and terrain evaluated)
    crow = []
    if curve_df is not None and len(curve_df):
        it = curve_df["checkpoint"].str.extract(r"_ep_(\d+)_rew")[0].astype(float)
        c = curve_df.assign(iteration=it).dropna(subset=["iteration"])
        rows_per_seed = c.groupby("train_seed").size()      # a seed still being evaluated has fewer rows
        full_seeds = [int(s) for s in rows_per_seed.index if rows_per_seed[s] == rows_per_seed.max()]
        c = c[c.train_seed.isin(full_seeds)]
        terr = sorted(c.terrain.unique())
        for mode, ls, mk in (("SP", "-", "o"), ("AD", "--", "s")):
            pts = []
            for k, g in c[c["mode"] == mode].groupby("iteration"):
                g = g.assign(cell="x")
                st = stat(g, "x", terr, "success")
                if st:
                    pts.append((k, st["mean"], st["lo"], st["hi"], st["n_seeds"]))
                    crow.append({"mode": mode, "iteration": int(k), **{q: st[q] for q in ("mean", "lo", "hi", "n_seeds")}})
            if pts:
                k, m, lo, hi, n = map(np.array, zip(*pts))
                a2.plot(k, m, ls=ls, marker=mk, ms=2, lw=0.9, color=COL["full"] if mode == "SP" else "#184f95",
                        label=f"{'SP' if mode == 'SP' else 'AD'}")
                a2.fill_between(k, lo, hi, color=COL["full"], alpha=0.15, lw=0)
        a2.set_xlabel("PPO iteration")
        a2.set_ylabel("Fixed-set success (%)")
        a2.set_xticks([0, 2500, 5000])
        a2.set_ylim(0, 100)
        a2.legend(frameon=False, loc="lower right", fontsize=5.5)
        a2.set_title(f"(b) {NAMES['full']}, {len(full_seeds)} seeds", loc="left", fontsize=7)
    PE._save(fig, out, "fig_train")
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig_train_reward.csv"), index=False)
    pd.DataFrame(crow).to_csv(os.path.join(out, "fig_train_curve.csv"), index=False)
    return {"final_reward": final, "curve": crow, "curve_terrains": terr if crow else [],
            "curve_seeds": [int(s) for s in full_seeds] if crow else []}


def fig_traj(results, out, terrain="stair_up", seed=1, robot=0, smooth_s=0.8):
    import torch
    plt = _plt()
    cells = ["ppo_mlp:ORIG", "birodiff:ORIG", "diffuseloco:ORIG", "ppo_only:SP", "full:SP"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.5), gridspec_kw={"wspace": 0.38})
    rows = []
    for cell in cells:
        method, _, mode = cell.partition(":")
        f = os.path.join(results, f"traj_{method}_seed{seed}_{mode}_{terrain}.pt")
        if not os.path.exists(f):
            print(f"missing {f}")
            continue
        d = torch.load(f, map_location="cpu")
        alive = d["alive"][:, robot].numpy().astype(bool)
        T = len(alive) if alive.all() else max(1, int(np.argmax(~alive)))
        dt = float(d["dt"])
        pos = d["pos"][:T, robot].numpy()
        vx = d["vel"][:T, robot, 0].numpy()
        cmd = d["cmd"][:T, robot, 0].numpy()
        rmse = float(np.sqrt(np.mean(((d["vel"][:T, robot] - d["cmd"][:T, robot]) ** 2).sum(-1).numpy())))
        fell = not alive.all()
        x = pos[:, 0] - pos[0, 0]
        lab = f"{NAMES[method]}: {rmse:.2f}" + (" (fell)" if fell else "")
        a1.plot(x, pos[:, 2], color=COL[method], ls=LS[method], lw=0.8, label=lab)
        if fell:
            a1.plot(x[-1], pos[-1, 2], "x", color=COL[method], ms=3.5, mew=0.9)
        k = max(1, int(round(smooth_s / dt)))
        t = np.arange(T) * dt
        a2.plot(t, pd.Series(vx).rolling(k, min_periods=1).mean(), color=COL[method], ls=LS[method], lw=0.7)
        rows.append({"cell": cell, "seconds": T * dt, "fell": fell, "vel_rmse": rmse, "distance_x": float(x[-1]),
                     "dt": dt, "steps": T})
    a2.axhline(0.5, color="k", lw=0.6, ls=(0, (2, 2)))
    a1.set_xlabel("Forward position (m)")
    a1.set_ylabel("Base height (m)")
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("Forward velocity (m/s)")
    a1.set_title("(a)", loc="left", fontsize=7)
    a2.set_title("(b)", loc="left", fontsize=7)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=5.3,
               title="velocity RMSE (m/s)", title_fontsize=5.3)
    PE._save(fig, out, "fig_traj")
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig_traj.csv"), index=False)
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--curve", default=None)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--scalars", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = AR.load(args.results)
    J = {"note": __doc__.split("\n\n")[1], "cells": {}}
    # coverage: seeds and terrains with each field logged
    cov = {}
    for cell, g in df.groupby("cell"):
        cov[cell] = {m: sorted(int(s) for s in g.dropna(subset=[m]).train_seed.unique())
                     for m in ("success", "spl", "fall", "footsteps", "path_len", "cot", "vel_rmse", "dev_from_sp")
                     if m in g}
    J["coverage"] = cov
    print("[ral] coverage:", json.dumps(cov)[:2000])

    base = ["ppo_mlp:ORIG", "birodiff:ORIG", "diffuseloco:ORIG", "ppo_only:SP", "full:SP"]
    ref = "full:SP"
    # 1. Table III: per terrain + suite, success / SPL / RMSE / progress; marks vs full:SP
    bench = {}
    for cell in base:
        bench[cell] = {}
        for m in ("success", "spl", "vel_rmse", "progress"):
            bench[cell][m] = {t: stat(df, cell, [t], m) for t in T10}
            bench[cell][m]["suite"] = stat(df, cell, T10, m)
    J["bench"] = bench
    J["bench_vs_ref"] = {c: {m: per_terrain(df, c, ref, m, pair=False) for m in ("success", "spl", "vel_rmse", "progress")}
                         for c in base if c != ref}
    J["bench_vs_ref_suite"] = {c: {m: independent(df, c, ref, T10, m) for m in ("success", "spl", "vel_rmse", "progress")}
                               for c in base if c != ref}
    print("[ral] bench done")
    # 2. Table IV: suite outcomes
    out_cells = base + ["full:AD", "full:D10"]
    om = ["success", "spl", "fall", "collision", "cot", "footsteps", "path_len", "joint_viol_per_min", "jerk_rms",
          "mean_n", "target_viol_pct", "limit_time_pct", "max_exc_rad", "progress", "vel_rmse"]
    J["outcomes"] = {c: {m: stat(df, c, T10, m, require_all=(m != "cot")) for m in om} for c in out_cells}
    J["outcomes_terrain"] = {c: {m: {t: stat(df, c, [t], m) for t in T10}
                                 for m in ("fall", "collision", "joint_viol_per_min", "cot", "footsteps")}
                             for c in out_cells}
    J["viol_vs_ppo_mlp"] = per_terrain(df, ref, "ppo_mlp:ORIG", "joint_viol_per_min", pair=False)
    J["viol_vs_ppo_mlp_suite"] = independent(df, ref, "ppo_mlp:ORIG", T10, "joint_viol_per_min")
    J["outcomes_vs_ref_suite"] = {c: {m: independent(df, c, ref, T10, m)
                                      for m in ("fall", "collision", "cot", "footsteps", "path_len", "joint_viol_per_min",
                                                "jerk_rms", "target_viol_pct")}
                                  for c in base if c != ref}
    print("[ral] outcomes done")
    # 3. Table II: per seed, suite success and SPL
    J["seeds"] = {c: {m: stat(df, c, T10, m) for m in ("success", "spl")} for c in base}
    # 4. inference modes vs SP (paired), per terrain (Holm) and suite
    J["modes"] = {}
    for method in ("ppo_only", "full"):
        modes = sorted(set(df[df.method == method]["mode"]) - {"SP"})
        J["modes"][method] = {}
        for mode in modes:
            a = f"{method}:{mode}"
            J["modes"][method][mode] = {
                "suite": {m: paired(df, a, f"{method}:SP", T10, m) for m in ("success", "spl", "progress", "vel_rmse")},
                "terrain_success": per_terrain(df, a, f"{method}:SP", "success", pair=True),
                "terrain_spl": per_terrain(df, a, f"{method}:SP", "spl", pair=True),
                "level": {m: stat(df, a, T10, m) for m in ("success", "spl", "mean_n", "dev_from_sp", "mean_eta")},
                "mean_n_terrain": {t: stat(df, a, [t], "mean_n") for t in T10},
            }
        J["modes"][method]["SP"] = {"level": {m: stat(df, f"{method}:SP", T10, m)
                                              for m in ("success", "spl", "dev_from_sp")}}
        print(f"[ral] modes {method} done")
    # 5. learned step count vs compute-matched controls; deterministic init; heads on the full checkpoints
    ctr = {}
    for c in ("AD-mean", "AD-shuffle", "AD-random", "ES", "D8", "D10", "D7"):
        if f"full:{c}" in set(df.cell):
            ctr[c] = {"suite": {m: paired(df, "full:AD", f"full:{c}", T10, m) for m in ("success", "spl")},
                      "terrain_success": per_terrain(df, "full:AD", f"full:{c}", "success", pair=True)}
    J["ad_vs"] = ctr
    J["init0"] = {"full": {"D7@0_vs_D7": paired(df, "full:D7@0", "full:D7", T10, "success"),
                           "AD@0_vs_AD": paired(df, "full:AD@0", "full:AD", T10, "success"),
                           "D7@0_vs_SP": paired(df, "full:D7@0", "full:SP", T10, "success")},
                  "ppo_only": {"D7@0_vs_D7": paired(df, "ppo_only:D7@0", "ppo_only:D7", T10, "success"),
                               "D7@0_vs_SP": paired(df, "ppo_only:D7@0", "ppo_only:SP", T10, "success")}}
    J["retest"] = {"suite": paired(df, "full:SP-retest", "full:SP", T10, "success"),
                   "terrain": per_terrain(df, "full:SP-retest", "full:SP", "success", pair=True)}
    heads = {}
    for h in ("full_tnppriv", "full_tnpuntrained"):
        if f"{h}:AD" in set(df.cell):
            heads[h] = {"vs_SP": paired(df, f"{h}:AD", "full:SP", T10, "success"),
                        "vs_AD": paired(df, f"{h}:AD", "full:AD", T10, "success"),
                        "level": {m: stat(df, f"{h}:AD", T10, m) for m in ("success", "mean_n")},
                        "mean_n_terrain": {t: stat(df, f"{h}:AD", [t], "mean_n") for t in T10}}
    J["heads"] = heads
    # 6. ablation rows available now: full (5 seeds) and PPO-only (5 seeds)
    J["ablation"] = {}
    for c, adc in (("full:SP", "full:AD"), ("ppo_only:SP", "ppo_only:AD")):
        J["ablation"][c] = {"success": stat(df, c, T10, "success"), "spl": stat(df, c, T10, "spl"),
                            "stair_up": stat(df, c, ["stair_up"], "success"),
                            "viol": stat(df, c, T10, "joint_viol_per_min"), "tgt": stat(df, c, T10, "target_viol_pct"),
                            "ad_minus_sp": paired(df, adc, c, T10, "success")}
    print("[ral] modes/controls/heads done")
    # 7. difficulty sweep at 0.5 m/s
    if args.sweep:
        sd = PE._load([args.sweep])
        J["sweep"] = {}
        for cell in sorted(sd.cell.unique()):
            J["sweep"][cell] = {}
            for kind in ("slope", "stair"):
                levels = PE._sweep_levels(sd, kind)
                if not levels:
                    continue
                unit = PE.SWEEP_UNIT[kind][1]
                mt = PE.max_traversable(sd[sd.cell == cell], levels, "success", 0.5)
                vals = np.array(list(mt.values()), float) * unit
                J["sweep"][cell][kind] = {
                    "max_mean": float(vals.mean()) if len(vals) else None,
                    "max_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                    "per_seed": {str(k): float(v) * unit for k, v in mt.items()},
                    "success_by_level": {str(lv * unit): float(100 * sd[(sd.cell == cell) & (sd.terrain == t)]
                                                               .groupby("train_seed").success.mean().mean())
                                         for t, lv in levels.items()}}
        print("[ral] sweep done")
    # 8. figures
    if args.scalars:
        cdf = PE._load([args.curve], allow_multi_ck=True) if args.curve else None
        J["training"] = fig_train(args.scalars, cdf, args.out)
    J["traj"] = fig_traj(args.results, args.out)
    # 9. controls added after the main analysis. Computed LAST: the bootstrap shares one random stream,
    #    so anything inserted earlier would shift the CIs/p-values of every later cell. Append new blocks below this one.
    jc = "ppo_mlp_joint:ORIG"
    if jc in set(df.cell):
        for m in ("success", "spl", "vel_rmse", "progress"):
            bench.setdefault(jc, {})[m] = {t: stat(df, jc, [t], m) for t in T10}
            bench[jc][m]["suite"] = stat(df, jc, T10, m)
        J["outcomes"][jc] = {m: stat(df, jc, T10, m, require_all=(m != "cot")) for m in om}
        J["outcomes_terrain"][jc] = {m: {t: stat(df, jc, [t], m) for t in T10}
                                     for m in ("fall", "collision", "joint_viol_per_min", "cot", "footsteps")}
        J["seeds"][jc] = {m: stat(df, jc, T10, m) for m in ("success", "spl")}
        qm = ("success", "spl", "joint_viol_per_min", "limit_time_pct", "target_viol_pct", "jerk_rms", "vel_rmse",
              "progress", "cot")
        J["vs_ppo_mlp_joint"] = {m: per_terrain(df, ref, jc, m, pair=False) for m in qm}
        J["vs_ppo_mlp_joint_suite"] = {m: independent(df, ref, jc, T10, m) for m in qm}
        J["jmlp_vs_mlp"] = {m: per_terrain(df, jc, "ppo_mlp:ORIG", m, pair=False) for m in qm}
        J["jmlp_vs_mlp_suite"] = {m: independent(df, jc, "ppo_mlp:ORIG", T10, m) for m in qm}
        print("[ral] ppo_mlp_joint control done")
    with open(os.path.join(args.out, "ral_numbers.json"), "w") as fh:
        json.dump(clean(J), fh, indent=1)
    print(f"[ral] -> {os.path.join(args.out, 'ral_numbers.json')}")


if __name__ == "__main__":
    main()
