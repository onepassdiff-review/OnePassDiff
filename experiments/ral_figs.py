#!/usr/bin/env python3
"""Compact RA-L figures at the heights the 6-page layout allows. Plots values already computed by
ral_numbers.py / plot_ral.py (their CSVs and JSON) and the trajectory files of run_e0.py; adds no statistics.

  fig_train.pdf   3.45 x 1.2 in  (a) training reward per seed (thin) and mean, three controllers; (b) fixed-set success
                                 of saved OnePassDiff checkpoints, SP and AD, mean and 95% CI over seeds
  fig_modes.pdf   7.16 x 1.6 in  two panels (PPO-only, OnePassDiff): SP row = success (%); other rows = paired difference
                                 from SP (pp), bold + "*" where Holm-corrected p < 0.05; right column = mean NFE
  fig_traj.pdf    3.45 x 1.3 in  one stair-ascent episode per controller: (a) base height over forward position,
                                 (b) forward velocity (0.8 s moving average) and command; legend = episode RMSE
  fig_sweep.pdf   3.45 x 1.15 in success (%) over slope angle / stair rise at 0.5 m/s, rows = controllers, dashed line =
                                 training maximum, right column = hardest level with >= 50% success (mean +- s.d.)
  fig_bars3d.pdf  3.45 x 1.5 in  the same values as 3D bars (style of REFINE-DP Fig. 10(a)):
                                 x = difficulty level, rows = controllers (one colour each), height = success (%); values
                                 from ral_numbers.json sweep.*.success_by_level, also written to fig_bars3d.csv
Heights of fig_train / fig_modes / fig_traj can be changed with --train-h / --modes-h / --traj-h; --only runs a subset.

Usage: python ral_figs.py --numbers OUT/ral_numbers.json --modes-csv OUT/heatmap_modes.csv
                          --sweep-csv OUT/heatmap_difficulty.csv --train-dir OUT --results RESULTS --out OUT/figs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

NAMES = {"full": "OnePassDiff", "ppo_only": "OnePassDiff-PPO-only", "ppo_mlp": "PPO-MLP", "birodiff": "BiRoDiff-R",
         "diffuseloco": "DiffuseLoco-R"}
COL = {"full": "#2a78d6", "ppo_only": "#eb6834", "ppo_mlp": "#1baf7a", "birodiff": "#eda100", "diffuseloco": "#e87ba4",
       "ppo_mlp_joint": "#008300"}   # slot 6 of the validated categorical palette
LS = {"full": "-", "ppo_only": "--", "ppo_mlp": "-.", "birodiff": ":", "diffuseloco": (0, (5, 1, 1, 1))}
TSHORT = {"flat": "Flat", "rough": "Rough", "slope10": "Slope\n10°", "slope18": "Slope\n18°", "stair_up": "Stair\nup",
          "stair_down": "Stair\ndown", "discrete": "Discr.", "heldout_slope": "H/O\nslope", "heldout_stairs": "H/O\nstairs",
          "heldout_gaps": "H/O\nstones"}
T10 = list(TSHORT)
INK, MUTED = "#0b0b0b", "#52514e"
GREY = ["#f0efec", "#c3c2b7", "#8e8d86", "#52514e"]
DIVERGING = ["#8f1d1c", "#c43a39", "#e98f8e", "#f0efec", "#86b6ef", "#2a78d6", "#104281"]
BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 5.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2, "axes.labelpad": 1.5})
    return plt


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


def _ink_on(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, pad_inches=0.01, bbox_inches="tight")
    print(f"figure -> {os.path.join(out, name)}.pdf")


def fig_train(train_dir, out, h=1.1):
    plt = _plt()
    r = pd.read_csv(os.path.join(train_dir, "fig_train_reward.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, h), gridspec_kw={"wspace": 0.38})
    for v in ("ppo_mlp", "ppo_only", "full"):
        d = r[r.variant == v]
        if d.empty:
            continue
        for s, g in d.groupby("seed"):
            a1.plot(g.iteration, g.reward, color=COL[v], lw=0.3, alpha=0.5)
        m = d.groupby("iteration").reward.mean()
        a1.plot(m.index, m.values, color=COL[v], lw=1.0, ls=LS[v], label=f"{NAMES[v]} ({d.seed.nunique()})")
    a1.set_xlabel("PPO iteration")
    a1.set_ylabel("Training reward")
    a1.set_xticks([0, 2500, 5000])
    a1.legend(frameon=False, loc="lower right", fontsize=5, handlelength=2.0, borderaxespad=0.1)
    a1.set_title("(a)", loc="left", pad=2)
    c = pd.read_csv(os.path.join(train_dir, "fig_train_curve.csv"))
    n = int(c.n_seeds.max()) if len(c) else 0
    for mode, ls, mk, col in (("SP", "-", "o", COL["full"]), ("AD", "--", "s", "#184f95")):
        d = c[c["mode"] == mode].sort_values("iteration")
        if d.empty:
            continue
        a2.plot(d.iteration, d["mean"], ls=ls, marker=mk, ms=1.8, lw=0.8, color=col, label=mode)
        a2.fill_between(d.iteration, d.lo, d.hi, color=col, alpha=0.15, lw=0)
    a2.set_xlabel("PPO iteration")
    a2.set_ylabel("Fixed-set success (%)")
    a2.set_xticks([0, 2500, 5000])
    a2.set_ylim(0, 100)
    a2.legend(frameon=False, loc="lower right", fontsize=5.5, borderaxespad=0.1)
    a2.set_title(f"(b) OnePassDiff, {n} seeds", loc="left", pad=2)
    _save(fig, out, "fig_train")


def fig_modes(modes_csv, numbers, out, rows=("SP", "D1", "D2", "D3", "D5", "D7", "D10", "D7@0", "AD", "AD-mean", "ES"), h=1.55,
              vmax=None):
    plt = _plt()
    from matplotlib.colors import Normalize
    # values from ral_numbers.json: SP row = per-terrain success of the single pass (bench), other rows = paired
    # difference from SP with Holm-corrected p (modes.<method>.<mode>.terrain_success)
    panels = []
    vm = 0.0
    for method in ("ppo_only", "full"):
        M = numbers["modes"][method]
        modes = [m for m in rows if m == "SP" or m in M]
        val = np.full((len(modes), len(T10)), np.nan)
        sig = np.zeros_like(val, bool)
        for i, m in enumerate(modes):
            for j, t in enumerate(T10):
                if m == "SP":
                    st = numbers["bench"][f"{method}:SP"]["success"].get(t)
                    if st:
                        val[i, j] = st["mean"]
                    continue
                r = M[m].get("terrain_success", {}).get(t)
                if r:
                    val[i, j] = r["diff"]
                    sig[i, j] = r.get("p_holm", 1.0) < 0.05
                    vm = max(vm, abs(val[i, j]))
        nfe = []
        for m in modes:
            if m == "SP":
                nfe.append(1.0)
            elif m.startswith("D"):
                nfe.append(float(m[1:].replace("@0", "")))
            else:
                lv = numbers["modes"][method].get(m, {}).get("level", {}).get("mean_n")
                nfe.append(lv["mean"] if lv else np.nan)
        panels.append((method, modes, val, sig, nfe))
    vmax = vmax or max(10.0, 10.0 * np.ceil(vm / 10.0))
    seq, div = _cmap(GREY, "grey"), _cmap(DIVERGING, "div")
    nseq, ndiv = Normalize(0, 100), Normalize(-vmax, vmax)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, h), gridspec_kw={"wspace": 0.34})
    for ax, (method, modes, val, sig, nfe) in zip(axes, panels):
        for i, m in enumerate(modes):
            for j in range(len(T10)):
                v = val[i, j]
                c = (1, 1, 1, 1) if np.isnan(v) else (seq(nseq(v)) if m == "SP" else div(ndiv(np.clip(v, -vmax, vmax))))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="white", lw=0.6))
                if not np.isnan(v):
                    txt = f"{v:.0f}" if m == "SP" else (f"{v:+.0f}" if round(v) != 0 else "0") + ("*" if sig[i, j] else "")
                    ax.text(j + 0.5, i + 0.52, txt, ha="center", va="center", fontsize=4.6, color=_ink_on(c),
                            fontweight="bold" if sig[i, j] else "normal")
            ax.text(len(T10) + 0.12, i + 0.52, "—" if np.isnan(nfe[i]) else f"{nfe[i]:.1f}".rstrip("0").rstrip("."),
                    ha="left", va="center", fontsize=4.8, color=MUTED)
        ax.axhline(1, color=INK, lw=0.6)
        ax.set_xlim(0, len(T10))
        ax.set_ylim(len(modes), 0)
        ax.set_xticks(np.arange(len(T10)) + 0.5)
        ax.set_xticklabels([TSHORT[t] for t in T10], fontsize=5.0, linespacing=0.95)
        ax.set_yticks(np.arange(len(modes)) + 0.5)
        ax.set_yticklabels([m.replace("@0", r" ($x_N$=0)").replace("D", "D-", 1) if m.startswith("D") else m
                            for m in modes], fontsize=5.2)
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(len(T10) + 0.12, -0.15, "NFE", ha="left", va="bottom", fontsize=4.8, color=MUTED)
        ax.set_title(NAMES[method], pad=2.5, fontsize=6.5)
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(ndiv, div), ax=list(axes), fraction=0.015, pad=0.035)
    cb.set_label("Difference from SP (pp)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_visible(False)
    _save(fig, out, "fig_modes")


def _gsmooth(y, sigma_steps):
    """Gaussian smoothing (numpy only), normalized at the ends so the curve is not pulled to zero."""
    y = np.asarray(y, dtype=float)
    if sigma_steps <= 0 or len(y) < 3:
        return y
    r = int(3 * sigma_steps)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_steps) ** 2)
    num = np.convolve(y, k, mode="full")[r:r + len(y)]   # "full" then centre slice: works when the kernel is longer
    den = np.convolve(np.ones_like(y), k, mode="full")[r:r + len(y)]
    return num / den


def fig_traj(results, out, terrain="stair_up", seed=1, robot=0, smooth_s=0.8, h=1.05, sig_z=0.0, sig_v=0.0):
    """sig_z / sig_v > 0 (seconds): Gaussian smoothing of height and velocity instead of the moving average."""
    import torch
    plt = _plt()
    cells = ["ppo_mlp:ORIG", "birodiff:ORIG", "diffuseloco:ORIG", "ppo_only:SP", "full:SP"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, h), gridspec_kw={"wspace": 0.36})
    handles = []
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
        rmse = float(np.sqrt(np.mean(((d["vel"][:T, robot] - d["cmd"][:T, robot]) ** 2).sum(-1).numpy())))
        fell = not alive.all()
        x = pos[:, 0] - pos[0, 0]
        t = np.arange(T) * dt
        z = pos[:, 2] - pos[0, 2]
        if sig_z > 0:
            z = _gsmooth(z, sig_z / dt)
        hh, = a1.plot(t, z, color=COL[method], ls=LS[method], lw=0.8,
                      label=f"{NAMES[method]} {rmse:.2f}" + (" (fell)" if fell else ""))
        handles.append(hh)
        if fell:
            a1.plot(t[-1], z[-1], "x", color=COL[method], ms=3, mew=0.8)
        if sig_v > 0:
            vs = _gsmooth(vx, sig_v / dt)
        else:
            vs = pd.Series(vx).rolling(max(1, int(round(smooth_s / dt))), min_periods=1).mean()
        a2.plot(np.arange(T) * dt, vs, color=COL[method], ls=LS[method], lw=0.8)
    a2.axhline(0.5, color="k", lw=0.5, ls=(0, (2, 2)))
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("Height gained (m)")
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("$v_x$ (m/s)")
    a1.set_title("(a)", loc="left", pad=2)
    a2.set_title("(b)", loc="left", pad=2)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=5,
               handlelength=1.8, columnspacing=0.8, handletextpad=0.4)
    _save(fig, out, "fig_traj")


def fig_grid3d(grid_csv, out, cells=("ppo_mlp:ORIG", "full:SP"), h=1.5, elev=24.0, azim=-58.0):
    """Speed x difficulty grid as 3D bars (REFINE-DP Fig. 10 style): x = difficulty level, rows = commanded speed,
    paired bars per cell (one colour per controller). Values from plot_ral.py grid (CSV: kind, cell, speed, level,
    success_pct, n_seeds); this function adds no statistics."""
    plt = _plt()
    import math
    from matplotlib.patches import Patch
    label = {"ppo_mlp:ORIG": "PPO-MLP", "full:SP": "OnePassDiff (SP)", "full:AD": "OnePassDiff (AD)",
             "birodiff:ORIG": "BiRoDiff-R"}
    color = {"ppo_mlp:ORIG": COL["ppo_mlp"], "full:SP": COL["full"], "full:AD": "#104281", "birodiff:ORIG": COL["birodiff"]}
    train_max = {"slope": 21.8, "stair": 15.0}
    xlab = {"slope": "Slope (deg)", "stair": "Stair rise (cm)"}
    d = pd.read_csv(grid_csv)
    d = d[d.cell.isin(cells)]
    kinds = [k for k in ("slope", "stair") if (d.kind == k).any()]
    speeds = sorted(d.speed.unique())
    fig = plt.figure(figsize=(3.45, h))
    box = (1.75, 1.05, 0.85)
    cam = (math.cos(math.radians(azim)), math.sin(math.radians(azim)))
    k = len(cells)
    w = 0.72 / k
    for a, kind in enumerate(kinds):
        dk = d[d.kind == kind]
        lv = sorted(dk.level.unique())
        ax = fig.add_axes([-0.07 + 0.5 * a, -0.12, 0.62, 1.12], projection="3d")
        ax.set_facecolor((1, 1, 1, 0))   # transparent: the right panel must not hide the left panel's v_x label
        try:
            ax.computed_zorder = False
        except Exception:
            pass
        bars = []
        for r, sp in enumerate(speeds):
            for j, x in enumerate(lv):
                for c, cell in enumerate(cells):
                    m = dk[(dk.cell == cell) & np.isclose(dk.speed, sp) & np.isclose(dk.level, x)]
                    if not len(m):
                        continue
                    x0 = j - 0.36 + c * w
                    depth = cam[0] * (x0 + w / 2 + 0.5) / len(lv) * box[0] + cam[1] * (r + 0.5) / len(speeds) * box[1]
                    bars.append((depth, x0, r, float(m.success_pct.iloc[0]), cell))
        for _, x0, r, v, cell in sorted(bars):
            ax.bar3d(x0, r - 0.3, 0, w * 0.92, 0.6, max(v, 0.6), color=color[cell], shade=True, edgecolor="none")
        kb = sum(1 for x in lv if x <= train_max[kind] + 1e-9) - 0.5
        ax.plot([kb, kb], [-0.5, len(speeds) - 0.5], [0, 0], color=INK, lw=0.7, ls="--")
        ax.set_xlim(-0.5, len(lv) - 0.5)
        ax.set_ylim(-0.5, len(speeds) - 0.5)
        ax.set_zlim(0, 100)
        try:
            ax.set_box_aspect(box)
        except Exception:
            pass
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([f"{x:g}" for x in lv], fontsize=4.8)
        ax.set_yticks(range(len(speeds)))
        ax.set_yticklabels([f"{v:g}" for v in speeds], fontsize=4.8)
        ax.set_zticks([0, 50, 100])
        ax.tick_params(axis="z", labelsize=4.8, pad=-2)
        ax.tick_params(axis="x", pad=-4.5)
        ax.tick_params(axis="y", pad=-4)
        ax.set_xlabel(xlab[kind], fontsize=5.5, labelpad=-9)
        ax.set_ylabel("$v_x$ (m/s)", fontsize=5.5, labelpad=-9)
        if a < len(kinds) - 1:
            ax.set_zticklabels([])
        else:
            ax.set_zlabel("Success (%)", fontsize=5.5, labelpad=-8)
        ax.view_init(elev=elev, azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.965, 0.965, 0.955, 1.0))
            axis.pane.set_edgecolor("#dcdbd5")
            try:
                axis._axinfo["grid"].update({"linewidth": 0.3, "color": "#dcdbd5"})
                axis._axinfo["tick"].update({"inward_factor": 0.0, "outward_factor": 0.15})
                axis.line.set_linewidth(0.4)
            except Exception:
                pass
    fig.legend(handles=[Patch(color=color[c], label=label[c]) for c in reversed(cells)], loc="upper center",
               bbox_to_anchor=(0.5, 0.935), ncol=len(cells), frameon=False, fontsize=4.8, handlelength=0.9,
               handletextpad=0.3, columnspacing=0.9)
    _save(fig, out, "fig_grid3d")
    d.to_csv(os.path.join(out, "fig_grid3d.csv"), index=False)


def fig_traj3d(results, out, terrain="stair_up", seed=1, robot=0, smooth_s=0.8, h=1.3, elev=24.0, azim=-62.0):
    """3D trajectory: (a) base path in the world frame, start at the origin (dot),
    x where the episode ended, grey floor shadows; (b) forward velocity (moving average) and command. Same episode and
    files as fig_traj; the plotted values are also written to fig_traj3d.csv."""
    import torch
    plt = _plt()
    cells = ["ppo_mlp:ORIG", "birodiff:ORIG", "diffuseloco:ORIG", "ppo_only:SP", "full:SP"]
    fig = plt.figure(figsize=(3.45, h))
    a1 = fig.add_axes([-0.11, -0.1, 0.78, 1.16], projection="3d")
    a2 = fig.add_axes([0.73, 0.25, 0.265, 0.6])
    paths, handles, rows = {}, [], []
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
        pos = pos - pos[0]
        vx = d["vel"][:T, robot, 0].numpy()
        rmse = float(np.sqrt(np.mean(((d["vel"][:T, robot] - d["cmd"][:T, robot]) ** 2).sum(-1).numpy())))
        paths[cell] = (method, pos, not alive.all(), rmse)
        k = max(1, int(round(smooth_s / dt)))
        a2.plot(np.arange(T) * dt, pd.Series(vx).rolling(k, min_periods=1).mean(), color=COL[method], ls=LS[method],
                lw=0.6)
        for i in range(0, T, 25):
            rows.append({"cell": cell, "t": i * dt, "x": pos[i, 0], "y": pos[i, 1], "z": pos[i, 2], "vx": vx[i]})
    if not paths:
        return
    allp = np.concatenate([p for _, p, _, _ in paths.values()])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    z0 = min(lo[2], 0.0) - 0.3
    try:
        a1.computed_zorder = False
    except Exception:
        pass
    for cell, (method, pos, fell, rmse) in paths.items():   # shadows first
        a1.plot(pos[:, 0], pos[:, 1], np.full(len(pos), z0), color="#c9c8c2", lw=0.5, zorder=1)
    for cell, (method, pos, fell, rmse) in paths.items():
        hnd, = a1.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=COL[method], ls=LS[method], lw=0.85, zorder=3,
                       label=f"{NAMES[method]} {rmse:.2f}" + (" (fell)" if fell else ""))
        handles.append(hnd)
        if fell:
            a1.plot([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]], "x", color=COL[method], ms=3.5, mew=0.9, zorder=4)
    a1.plot([0], [0], [0], "o", color=INK, ms=2.2, zorder=5)
    pad = 0.6
    a1.set_xlim(lo[0] - pad, hi[0] + pad)
    a1.set_ylim(lo[1] - pad, hi[1] + pad)
    a1.set_zlim(z0, hi[2] + 0.2)
    try:
        rx, ry = hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad   # x, y to scale; z exaggerated for readability
        a1.set_box_aspect((rx, ry, 0.5 * max(rx, ry)))
    except Exception:
        pass
    a1.view_init(elev=elev, azim=azim)
    from matplotlib.ticker import MaxNLocator
    for ax_ in (a1.xaxis, a1.yaxis):
        ax_.set_major_locator(MaxNLocator(3, integer=True))
    a1.zaxis.set_major_locator(MaxNLocator(3, integer=True))
    a1.tick_params(axis="both", labelsize=4.8, pad=-3)
    a1.tick_params(axis="z", pad=-1.5)
    a1.set_xlabel("$x$ (m)", fontsize=5.5, labelpad=-8)
    a1.set_ylabel("$y$ (m)", fontsize=5.5, labelpad=-8)
    a1.set_zlabel("$z$ (m)", fontsize=5.5, labelpad=-8)
    for axis in (a1.xaxis, a1.yaxis, a1.zaxis):
        axis.pane.set_facecolor((0.965, 0.965, 0.955, 1.0))
        axis.pane.set_edgecolor("#dcdbd5")
        try:
            axis._axinfo["grid"].update({"linewidth": 0.3, "color": "#dcdbd5"})
            axis.line.set_linewidth(0.4)
        except Exception:
            pass
    a1.text2D(0.12, 0.84, "(a)", transform=a1.transAxes, fontsize=6.5)
    a2.axhline(0.5, color="k", lw=0.5, ls=(0, (2, 2)))
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("$v_x$ (m/s)")
    a2.set_title("(b)", loc="left", pad=2)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=5,
               handlelength=1.8, columnspacing=0.8, handletextpad=0.4)
    _save(fig, out, "fig_traj3d")
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig_traj3d.csv"), index=False)


def fig_sweep(sweep_csv, numbers, out, cells=("ppo_mlp:ORIG", "birodiff:ORIG", "diffuseloco:ORIG", "full:SP", "full:AD")):
    plt = _plt()
    from matplotlib.colors import Normalize
    d = pd.read_csv(sweep_csv)
    kinds = [k for k in ("slope", "stair") if (d.kind == k).any()]
    lv = {k: sorted(d[d.kind == k].level.unique()) for k in kinds}
    seq = _cmap(BLUE, "seq0")
    norm = Normalize(0, 100)
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.45, 1.15),
                             gridspec_kw={"width_ratios": [len(lv[k]) + 1.6 for k in kinds], "wspace": 0.1})
    train_max = {"slope": 21.8, "stair": 15.0}
    label = {"ppo_mlp:ORIG": "PPO-MLP", "birodiff:ORIG": "BiRoDiff-R", "diffuseloco:ORIG": "DiffuseLoco-R",
             "full:SP": "OnePassDiff (SP)", "full:AD": "OnePassDiff (AD)"}
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        L = lv[kind]
        for i, cell in enumerate(cells):
            for j, x in enumerate(L):
                r = d[(d.cell == cell) & (d.kind == kind) & (np.isclose(d.level, x))]
                v = r.success_pct.iloc[0] if len(r) else np.nan
                c = (1, 1, 1, 1) if np.isnan(v) else seq(norm(v))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="white", lw=0.6))
                if not np.isnan(v):
                    ax.text(j + 0.5, i + 0.52, f"{v:.0f}", ha="center", va="center", fontsize=4.6, color=_ink_on(c))
            sw = numbers.get("sweep", {}).get(cell, {}).get(kind)
            if sw and sw.get("max_mean") is not None:
                ax.text(len(L) + 0.1, i + 0.52, f"{sw['max_mean']:.0f}±{(sw['max_sd'] or 0):.0f}", ha="left",
                        va="center", fontsize=4.6, color=MUTED)
        k = sum(1 for x in L if x <= train_max[kind] + 1e-9)
        ax.plot([k, k], [0, len(cells)], color=INK, lw=0.7, ls="--")
        ax.text(len(L) + 0.1, -0.1, "max", ha="left", va="bottom", fontsize=4.6, color=MUTED)
        ax.set_xlim(0, len(L) + 1.6)
        ax.set_ylim(len(cells), 0)
        ax.set_xticks(np.arange(len(L)) + 0.5)
        ax.set_xticklabels([f"{x:g}" for x in L], fontsize=5)
        ax.set_xlabel("Slope (deg)" if kind == "slope" else "Stair rise (cm)", fontsize=5.8)
        ax.set_yticks(np.arange(len(cells)) + 0.5)
        ax.set_yticklabels([label[c] for c in cells] if kind == kinds[0] else [], fontsize=5.2)
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
    _save(fig, out, "fig_sweep")


def fig_sheet3d(success_csv, progress_csv, out, cells=("full:SP", "ppo_mlp:ORIG", "birodiff:ORIG", "diffuseloco:ORIG"),
                row_h=0.62, top=0.13, elev=24.0, azim=-58.0):
    """Sheet of 3D bars (REFINE-DP Fig. 10(a) style), one row per controller, columns = uphill slope / stair rise:
    x = difficulty level, y = commanded speed, height = success (%), colour = progress (% of the commanded distance,
    one-hue blue ramp, clipped to 0-100). Colour shows where success came from standing still (tall, pale bars).
    Inputs: two plot_ral.py grid CSVs (--metric success / --metric progress; the value column is success_pct in both).
    Adds no statistics; writes fig_sheet3d.csv with both values per cell."""
    plt = _plt()
    import math
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    label = {"ppo_mlp:ORIG": "PPO-MLP", "full:SP": "OnePassDiff", "birodiff:ORIG": "BiRoDiff-R",
             "diffuseloco:ORIG": "DiffuseLoco-R"}
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # blue 100..700
    cmap = LinearSegmentedColormap.from_list("prog", ramp)
    norm = Normalize(0, 100)
    train_max = {"slope": 21.8, "stair": 15.0}
    head = {"slope": "Uphill slope (deg)", "stair": "Stair rise (cm)"}
    key = ["kind", "cell", "speed", "level"]
    s = pd.read_csv(success_csv)
    p = pd.read_csv(progress_csv).rename(columns={"success_pct": "progress_pct"})
    d = s.merge(p[key + ["progress_pct"]], on=key, how="left")
    d = d[d.cell.isin(cells)]
    kinds = [k for k in ("slope", "stair") if (d.kind == k).any()]
    speeds = sorted(d.speed.unique())
    H = top + row_h * len(cells)
    fig = plt.figure(figsize=(3.45, H))
    box = (1.75, 0.95, 0.52)
    cam = (math.cos(math.radians(azim)), math.sin(math.radians(azim)))
    for r_c, cell in enumerate(cells):
        y0 = (H - top - row_h * (r_c + 1)) / H
        for a, kind in enumerate(kinds):
            dk = d[(d.kind == kind) & (d.cell == cell)]
            lv = sorted(d[d.kind == kind].level.unique())
            ax = fig.add_axes([-0.06 + 0.5 * a, y0 - 0.22 * row_h / H, 0.6, 1.42 * row_h / H], projection="3d")
            ax.set_facecolor((1, 1, 1, 0))
            try:
                ax.computed_zorder = False
            except Exception:
                pass
            bars = []
            for r, sp in enumerate(speeds):
                for j, x in enumerate(lv):
                    m = dk[np.isclose(dk.speed, sp) & np.isclose(dk.level, x)]
                    if not len(m):
                        continue
                    depth = cam[0] * (j + 0.5) / len(lv) * box[0] + cam[1] * (r + 0.5) / len(speeds) * box[1]
                    bars.append((depth, j, r, float(m.success_pct.iloc[0]), float(m.progress_pct.iloc[0])))
            for _, j, r, v, pr in sorted(bars):
                ax.bar3d(j - 0.32, r - 0.3, 0, 0.64, 0.6, max(v, 0.6), color=cmap(norm(np.clip(pr, 0, 100))),
                         shade=True, edgecolor="none")
            kb = sum(1 for x in lv if x <= train_max[kind] + 1e-9) - 0.5
            ax.plot([kb, kb], [-0.5, len(speeds) - 0.5], [0, 0], color=INK, lw=0.7, ls="--")
            ax.set_xlim(-0.5, len(lv) - 0.5)
            ax.set_ylim(-0.5, len(speeds) - 0.5)
            ax.set_zlim(0, 100)
            try:
                ax.set_box_aspect(box)
            except Exception:
                pass
            ax.set_xticks(range(len(lv)))
            ax.set_xticklabels([f"{x:g}" for x in lv], fontsize=4.5)
            ax.set_yticks(range(len(speeds)))
            ax.set_yticklabels([f"{v:g}" for v in speeds], fontsize=4.5)
            ax.set_zticks([0, 50, 100])
            ax.tick_params(axis="z", labelsize=4.5, pad=-2.5)
            ax.tick_params(axis="x", pad=-5)
            ax.tick_params(axis="y", pad=-4.5)
            ax.set_ylabel("$v_x$ (m/s)", fontsize=4.8, labelpad=-10)
            if a < len(kinds) - 1:
                ax.set_zticklabels([])
            else:
                ax.set_zlabel("Success (%)", fontsize=4.8, labelpad=-9)
            ax.view_init(elev=elev, azim=azim)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((0.965, 0.965, 0.955, 1.0))
                axis.pane.set_edgecolor("#dcdbd5")
                try:
                    axis._axinfo["grid"].update({"linewidth": 0.3, "color": "#dcdbd5"})
                    axis._axinfo["tick"].update({"inward_factor": 0.0, "outward_factor": 0.15})
                    axis.line.set_linewidth(0.4)
                except Exception:
                    pass
        fig.text(0.012, y0 + 0.5 * row_h / H, label[cell], rotation=90, ha="left", va="center", fontsize=5.5,
                 color=INK, fontweight="bold")
    yh = 1 - 0.06 / H   # one band: column headers left and right, colour bar between them
    for a, kind in enumerate(kinds):
        fig.text(0.22 + 0.56 * a, yh, head[kind], ha="center", va="center", fontsize=5.5, color=INK)
    cax = fig.add_axes([0.47, yh - 0.022 / H, 0.17, 0.044 / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    cb.outline.set_linewidth(0.3)
    cax.tick_params(labelsize=4.3, length=1.2, width=0.3, pad=0.8)
    cax.text(-0.04, 0.5, "Progress (%)", transform=cax.transAxes, ha="right", va="center", fontsize=4.8, color=INK)
    _save(fig, out, "fig_sheet3d")
    d.to_csv(os.path.join(out, "fig_sheet3d.csv"), index=False)


def fig_surf3d(success_csv, progress_csv, out, cells=("diffuseloco:ORIG", "birodiff:ORIG", "ppo_mlp:ORIG", "full:SP"),
               row_h=1.2, legend_h=0.2, elev=24.0, azim=-58.0, alpha=0.5):
    """REFINE-DP Fig. 10 style: one translucent triangulated surface per controller over the level x speed grid,
    (a) success and (b) progress (% of the commanded distance; negative = slid back), columns = uphill slope / stair rise.
    Inputs: two plot_ral.py grid CSVs (--metric success / --metric progress; value column success_pct in both).
    Adds no statistics; writes fig_surf3d.csv."""
    plt = _plt()
    from matplotlib.colors import to_rgb, to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    label = {"ppo_mlp:ORIG": "PPO-MLP", "full:SP": "OnePassDiff (SP)", "birodiff:ORIG": "BiRoDiff-R",
             "diffuseloco:ORIG": "DiffuseLoco-R"}
    color = {"ppo_mlp:ORIG": COL["ppo_mlp"], "full:SP": COL["full"], "birodiff:ORIG": COL["birodiff"],
             "diffuseloco:ORIG": COL["diffuseloco"]}
    dark = lambda c: tuple(0.55 * v for v in to_rgb(c))  # noqa: E731
    train_max = {"slope": 21.8, "stair": 15.0}
    xlab = {"slope": "Slope (deg)", "stair": "Stair rise (cm)"}
    key = ["kind", "cell", "speed", "level"]
    s = pd.read_csv(success_csv)
    p = pd.read_csv(progress_csv).rename(columns={"success_pct": "progress_pct"})
    d = s.merge(p[key + ["progress_pct"]], on=key, how="left")
    d = d[d.cell.isin(cells)]
    kinds = [k for k in ("slope", "stair") if (d.kind == k).any()]
    speeds = sorted(d.speed.unique())
    metrics = [("success_pct", "Success (%)", (0.0, 100.0), "(a) Success"),
               ("progress_pct", "Progress (%)", (-25.0, 100.0), "(b) Progress")]
    H = legend_h + row_h * len(metrics)
    fig = plt.figure(figsize=(3.45, H))
    box = (1.75, 1.0, 0.8)
    for m_i, (col, zl, zlim, rowlab) in enumerate(metrics):
        yb = (legend_h + row_h * (len(metrics) - 1 - m_i)) / H
        for a, kind in enumerate(kinds):
            dk = d[d.kind == kind]
            lv = sorted(dk.level.unique())
            ax = fig.add_axes([-0.06 + 0.5 * a, yb - 0.10 * row_h / H, 0.6, 1.22 * row_h / H], projection="3d")
            ax.set_facecolor((1, 1, 1, 0))
            try:
                ax.computed_zorder = False   # markers after the surfaces
            except Exception:
                pass
            # all triangles of all controllers in ONE collection, so that mplot3d depth-sorts them face by face
            # (separate surfaces are only sorted as wholes and overlap wrongly); regular diagonals as in REFINE-DP
            polys, fcs, ecs, pts = [], [], [], []
            for cell in cells:
                Z = np.full((len(speeds), len(lv)), np.nan)
                for r, sp in enumerate(speeds):
                    for j, x in enumerate(lv):
                        m = dk[(dk.cell == cell) & np.isclose(dk.speed, sp) & np.isclose(dk.level, x)]
                        if len(m):
                            Z[r, j] = float(m[col].iloc[0])
                if np.isnan(Z).any():
                    continue
                fc, ec = to_rgba(color[cell], alpha), dark(color[cell]) + (0.9,)
                for r in range(len(speeds) - 1):
                    for j in range(len(lv) - 1):
                        q = [(j, r, Z[r, j]), (j + 1, r, Z[r, j + 1]), (j + 1, r + 1, Z[r + 1, j + 1]),
                             (j, r + 1, Z[r + 1, j])]
                        for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                            polys.append(list(t)); fcs.append(fc); ecs.append(ec)
                pts.append((cell, Z))
            ax.add_collection3d(Poly3DCollection(polys, facecolors=fcs, edgecolors=ecs, linewidths=0.22))
            for cell, Z in pts:
                Xg, Yg = np.meshgrid(np.arange(len(lv), dtype=float), np.arange(len(speeds), dtype=float))
                ax.scatter(Xg.ravel(), Yg.ravel(), Z.ravel(), s=2.0, color=color[cell], edgecolors=INK,
                           linewidths=0.2, depthshade=False)
            kb = sum(1 for x in lv if x <= train_max[kind] + 1e-9) - 0.5
            ax.plot([kb, kb], [0, len(speeds) - 1], [zlim[0], zlim[0]], color=INK, lw=0.7, ls="--")
            ax.plot([kb, kb], [len(speeds) - 1, len(speeds) - 1], list(zlim), color=INK, lw=0.7, ls="--")
            ax.set_xlim(-0.2, len(lv) - 0.8)
            ax.set_ylim(-0.15, len(speeds) - 0.85)
            ax.set_zlim(*zlim)
            try:
                ax.set_box_aspect(box)
            except Exception:
                pass
            ax.set_xticks(range(len(lv)))
            ax.set_xticklabels([f"{x:g}" for x in lv], fontsize=4.5)
            ax.set_yticks(range(len(speeds)))
            ax.set_yticklabels([f"{v:g}" for v in speeds], fontsize=4.5)
            ax.set_zticks([0, 50, 100])
            ax.tick_params(axis="z", labelsize=4.5, pad=-2.5)
            ax.tick_params(axis="x", pad=-5)
            ax.tick_params(axis="y", pad=-4.5)
            ax.set_xlabel(xlab[kind], fontsize=4.8, labelpad=-10)
            ax.set_ylabel("$v_x$ (m/s)", fontsize=4.8, labelpad=-10)
            if a < len(kinds) - 1:
                ax.set_zticklabels([])
            else:
                ax.set_zlabel(zl, fontsize=4.8, labelpad=-9)
            ax.view_init(elev=elev, azim=azim)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((0.965, 0.965, 0.955, 1.0))
                axis.pane.set_edgecolor("#dcdbd5")
                try:
                    axis._axinfo["grid"].update({"linewidth": 0.3, "color": "#dcdbd5"})
                    axis._axinfo["tick"].update({"inward_factor": 0.0, "outward_factor": 0.15})
                    axis.line.set_linewidth(0.4)
                except Exception:
                    pass
        fig.text(0.012, yb + 0.5 * row_h / H, rowlab, rotation=90, ha="left", va="center", fontsize=5.5,
                 color=INK, fontweight="bold")
    handles = [Patch(facecolor=color[c], alpha=alpha + 0.15, edgecolor=dark(color[c]), linewidth=0.4, label=label[c])
               for c in reversed(cells)]
    handles.append(Line2D([0], [0], color=INK, lw=0.7, ls="--", label="Training maximum"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.45 * legend_h / H), ncol=5, frameon=False,
               fontsize=4.6, handlelength=1.1, handletextpad=0.3, columnspacing=0.8)
    _save(fig, out, "fig_surf3d")
    d.to_csv(os.path.join(out, "fig_surf3d.csv"), index=False)


def fig_combo3d(success_csv, progress_csv, out, cells=("full:SP", "ppo_mlp:ORIG", "birodiff:ORIG", "diffuseloco:ORIG"),
                head_h=0.12, bar_h=0.58, surf_h=1.12, legend_h=0.16, elev=24.0, azim=-58.0, alpha=0.5, legend_ncol=5):
    """Fig. 6 of the letter: (a) success as 3D bars,
    one row per controller (level x speed, colour = controller); (b) progress as translucent surfaces, one per controller
    on shared axes (negative = slid back). Each metric is drawn once. Columns: uphill slope / stair rise.
    Inputs: two plot_ral.py grid CSVs (--metric success / --metric progress; value column success_pct in both)."""
    plt = _plt()
    import math
    from matplotlib.colors import to_rgb, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    label = {"ppo_mlp:ORIG": "PPO-MLP", "full:SP": "OnePassDiff", "birodiff:ORIG": "BiRoDiff-R",
             "diffuseloco:ORIG": "DiffuseLoco-R", "ppo_mlp_joint:ORIG": "PPO-MLP+$\\mathcal{L}_\\mathrm{joint}$"}
    row_label = dict(label, **{"ppo_mlp_joint:ORIG": "PPO-MLP\n+$\\mathcal{L}_\\mathrm{joint}$"})   # two lines: fits a row
    color = {"ppo_mlp:ORIG": COL["ppo_mlp"], "full:SP": COL["full"], "birodiff:ORIG": COL["birodiff"],
             "diffuseloco:ORIG": COL["diffuseloco"], "ppo_mlp_joint:ORIG": COL["ppo_mlp_joint"]}
    dark = lambda c: tuple(0.55 * v for v in to_rgb(c))  # noqa: E731
    train_max = {"slope": 21.8, "stair": 15.0}
    head = {"slope": "Uphill slope (deg)", "stair": "Stair rise (cm)"}
    key = ["kind", "cell", "speed", "level"]
    s = pd.read_csv(success_csv)
    p = pd.read_csv(progress_csv).rename(columns={"success_pct": "progress_pct"})
    d = s.merge(p[key + ["progress_pct"]], on=key, how="left")
    d = d[d.cell.isin(cells)]
    kinds = [k for k in ("slope", "stair") if (d.kind == k).any()]
    speeds = sorted(d.speed.unique())
    H = head_h + bar_h * len(cells) + surf_h + legend_h
    fig = plt.figure(figsize=(3.45, H))
    cam = (math.cos(math.radians(azim)), math.sin(math.radians(azim)))

    def style(ax, lv, zlim, zl, right, xl=None):
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([f"{x:g}" for x in lv], fontsize=4.5)
        ax.set_yticks(range(len(speeds)))
        ax.set_yticklabels([f"{v:g}" for v in speeds], fontsize=4.5)
        ax.set_zlim(*zlim)
        ax.set_zticks([0, 50, 100])
        ax.tick_params(axis="z", labelsize=4.5, pad=-2.5)
        ax.tick_params(axis="x", pad=-5)
        ax.tick_params(axis="y", pad=-4.5)
        ax.set_ylabel("$v_x$ (m/s)", fontsize=4.8, labelpad=-10)
        if xl:
            ax.set_xlabel(xl, fontsize=4.8, labelpad=-10)
        if not right:
            ax.set_zticklabels([])
        else:
            ax.set_zlabel(zl, fontsize=4.8, labelpad=-9)
        ax.view_init(elev=elev, azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.965, 0.965, 0.955, 1.0))
            axis.pane.set_edgecolor("#dcdbd5")
            try:
                axis._axinfo["grid"].update({"linewidth": 0.3, "color": "#dcdbd5"})
                axis._axinfo["tick"].update({"inward_factor": 0.0, "outward_factor": 0.15})
                axis.line.set_linewidth(0.4)
            except Exception:
                pass

    def grid_vals(dk, cell, lv, col):
        Z = np.full((len(speeds), len(lv)), np.nan)
        for r, sp in enumerate(speeds):
            for j, x in enumerate(lv):
                m = dk[(dk.cell == cell) & np.isclose(dk.speed, sp) & np.isclose(dk.level, x)]
                if len(m):
                    Z[r, j] = float(m[col].iloc[0])
        return Z

    # (a) success: 3D bars, one row per controller
    box_b = (1.75, 0.95, 0.48)
    for r_c, cell in enumerate(cells):
        yb = (H - head_h - bar_h * (r_c + 1)) / H
        for a, kind in enumerate(kinds):
            dk = d[d.kind == kind]
            lv = sorted(dk.level.unique())
            ax = fig.add_axes([-0.06 + 0.5 * a, yb - 0.22 * bar_h / H, 0.6, 1.42 * bar_h / H], projection="3d")
            ax.set_facecolor((1, 1, 1, 0))
            try:
                ax.computed_zorder = False
            except Exception:
                pass
            Z = grid_vals(dk, cell, lv, "success_pct")
            bars = []
            for r in range(len(speeds)):
                for j in range(len(lv)):
                    if not np.isnan(Z[r, j]):
                        depth = cam[0] * (j + 0.5) / len(lv) * box_b[0] + cam[1] * (r + 0.5) / len(speeds) * box_b[1]
                        bars.append((depth, j, r, Z[r, j]))
            for _, j, r, v in sorted(bars):
                ax.bar3d(j - 0.32, r - 0.3, 0, 0.64, 0.6, max(v, 0.6), color=color[cell], shade=True, edgecolor="none")
            kb = sum(1 for x in lv if x <= train_max[kind] + 1e-9) - 0.5
            ax.plot([kb, kb], [-0.5, len(speeds) - 0.5], [0, 0], color=INK, lw=0.7, ls="--")
            ax.set_xlim(-0.5, len(lv) - 0.5)
            ax.set_ylim(-0.5, len(speeds) - 0.5)
            try:
                ax.set_box_aspect(box_b)
            except Exception:
                pass
            style(ax, lv, (0, 100), "Success (%)", a == len(kinds) - 1)
        fig.text(0.035, yb + 0.5 * bar_h / H, row_label[cell], rotation=90, ha="left", va="center", fontsize=5.2,
                 color=INK, fontweight="bold", linespacing=1.0, multialignment="center")
    ya = (H - head_h - 0.5 * bar_h * len(cells)) / H
    fig.text(0.0, ya, "(a) Success", rotation=90, ha="left", va="center", fontsize=5.5, color=INK, fontweight="bold")
    for a, kind in enumerate(kinds):
        fig.text(0.25 + 0.5 * a, 1 - 0.055 / H, head[kind], ha="center", va="center", fontsize=5.5, color=INK)

    # (b) progress: translucent surfaces, all triangles in one depth-sorted collection
    box_s = (1.75, 1.0, 0.72)
    yb = legend_h / H
    zlim = (-25.0, 100.0)
    for a, kind in enumerate(kinds):
        dk = d[d.kind == kind]
        lv = sorted(dk.level.unique())
        ax = fig.add_axes([-0.06 + 0.5 * a, yb - 0.10 * surf_h / H, 0.6, 1.22 * surf_h / H], projection="3d")
        ax.set_facecolor((1, 1, 1, 0))
        try:
            ax.computed_zorder = False
        except Exception:
            pass
        polys, fcs, ecs, pts = [], [], [], []
        for cell in reversed(cells):
            Z = grid_vals(dk, cell, lv, "progress_pct")
            if np.isnan(Z).any():
                continue
            fc, ec = to_rgba(color[cell], alpha), dark(color[cell]) + (0.9,)
            for r in range(len(speeds) - 1):
                for j in range(len(lv) - 1):
                    q = [(j, r, Z[r, j]), (j + 1, r, Z[r, j + 1]), (j + 1, r + 1, Z[r + 1, j + 1]), (j, r + 1, Z[r + 1, j])]
                    for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                        polys.append(list(t)); fcs.append(fc); ecs.append(ec)
            pts.append((cell, Z))
        ax.add_collection3d(Poly3DCollection(polys, facecolors=fcs, edgecolors=ecs, linewidths=0.22))
        Xg, Yg = np.meshgrid(np.arange(len(lv), dtype=float), np.arange(len(speeds), dtype=float))
        for cell, Z in pts:
            ax.scatter(Xg.ravel(), Yg.ravel(), Z.ravel(), s=2.0, color=color[cell], edgecolors=INK, linewidths=0.2,
                       depthshade=False)
        kb = sum(1 for x in lv if x <= train_max[kind] + 1e-9) - 0.5
        ax.plot([kb, kb], [0, len(speeds) - 1], [zlim[0], zlim[0]], color=INK, lw=0.7, ls="--")
        ax.plot([kb, kb], [len(speeds) - 1, len(speeds) - 1], list(zlim), color=INK, lw=0.7, ls="--")
        ax.set_xlim(-0.2, len(lv) - 0.8)
        ax.set_ylim(-0.15, len(speeds) - 0.85)
        try:
            ax.set_box_aspect(box_s)
        except Exception:
            pass
        style(ax, lv, zlim, "Progress (%)", a == len(kinds) - 1, xl=head[kind])
    fig.text(0.0, yb + 0.5 * surf_h / H, "(b) Progress", rotation=90, ha="left", va="center", fontsize=5.5, color=INK,
             fontweight="bold")
    handles = [Patch(facecolor=color[c], edgecolor=dark(color[c]), linewidth=0.4, label=label[c]) for c in cells]
    handles.append(Line2D([0], [0], color=INK, lw=0.7, ls="--", label="Training maximum"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.42 * legend_h / H), ncol=legend_ncol, frameon=False,
               fontsize=4.6, handlelength=1.1, handletextpad=0.3, columnspacing=0.8)
    _save(fig, out, "fig_combo3d")
    d.to_csv(os.path.join(out, "fig_combo3d.csv"), index=False)


def fig_bars3d(numbers, out, cells=("diffuseloco:ORIG", "birodiff:ORIG", "ppo_mlp:ORIG", "full:AD", "full:SP"),
               elev=24.0, azim=-58.0, h=1.5):
    """3D bars of success over difficulty level (x) and controller (rows, front to back in the order of cells)."""
    plt = _plt()
    import math
    from matplotlib.patches import Patch
    label = {"ppo_mlp:ORIG": "PPO-MLP", "birodiff:ORIG": "BiRoDiff-R", "diffuseloco:ORIG": "DiffuseLoco-R",
             "full:SP": "OnePassDiff (SP)", "full:AD": "OnePassDiff (AD)"}
    color = {"ppo_mlp:ORIG": COL["ppo_mlp"], "birodiff:ORIG": COL["birodiff"], "diffuseloco:ORIG": COL["diffuseloco"],
             "full:SP": COL["full"], "full:AD": "#104281"}
    train_max = {"slope": 21.8, "stair": 15.0}
    xlab = {"slope": "Slope (deg)", "stair": "Stair rise (cm)"}
    sw = numbers["sweep"]
    kinds = [k for k in ("slope", "stair") if all(k in sw.get(c, {}) for c in cells)]
    fig = plt.figure(figsize=(3.45, h))
    rows, box = [], (1.75, 1.25, 0.85)
    cam = (math.cos(math.radians(azim)), math.sin(math.radians(azim)))
    for a, kind in enumerate(kinds):
        lv = sorted({float(x) for c in cells for x in sw[c][kind]["success_by_level"]})
        ax = fig.add_axes([-0.07 + 0.5 * a, -0.12, 0.62, 1.12], projection="3d")
        try:
            ax.computed_zorder = False   # painter's order below: bars far from the camera first
        except Exception:
            pass
        bars = []
        for r, cell in enumerate(cells):
            sbl = {float(k): v for k, v in sw[cell][kind]["success_by_level"].items()}
            for j, x in enumerate(lv):
                v = sbl.get(x)
                if v is None:
                    continue
                rows.append({"kind": kind, "cell": cell, "level": x, "success_pct": v})
                depth = cam[0] * (j + 0.5) / len(lv) * box[0] + cam[1] * (r + 0.5) / len(cells) * box[1]
                bars.append((depth, j, r, v, cell))
        for _, j, r, v, cell in sorted(bars):
            ax.bar3d(j - 0.32, r - 0.3, 0, 0.64, 0.6, max(v, 0.6), color=color[cell], shade=True, edgecolor="none")
        kb = sum(1 for x in lv if x <= train_max[kind] + 1e-9) - 0.5
        ax.plot([kb, kb], [-0.5, len(cells) - 0.5], [0, 0], color=INK, lw=0.7, ls="--")
        ax.set_xlim(-0.5, len(lv) - 0.5)
        ax.set_ylim(-0.5, len(cells) - 0.5)
        ax.set_zlim(0, 100)
        try:
            ax.set_box_aspect(box)
        except Exception:
            pass
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([f"{x:g}" for x in lv], fontsize=4.8)
        ax.set_yticks([])
        ax.set_zticks([0, 50, 100])
        ax.tick_params(axis="z", labelsize=4.8, pad=-2)
        ax.tick_params(axis="x", pad=-4.5)
        ax.set_xlabel(xlab[kind], fontsize=5.5, labelpad=-9)
        if a < len(kinds) - 1:   # the z axis is drawn on the right; the left panel's labels would sit under the next panel
            ax.set_zticklabels([])
        else:
            ax.set_zlabel("Success (%)", fontsize=5.5, labelpad=-8)
        ax.view_init(elev=elev, azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.965, 0.965, 0.955, 1.0))
            axis.pane.set_edgecolor("#dcdbd5")
            try:
                axis._axinfo["grid"].update({"linewidth": 0.3, "color": "#dcdbd5"})
                axis._axinfo["tick"].update({"inward_factor": 0.0, "outward_factor": 0.15})
                axis.line.set_linewidth(0.4)
            except Exception:
                pass
    fig.legend(handles=[Patch(color=color[c], label=label[c]) for c in reversed(cells)], loc="upper center",
               bbox_to_anchor=(0.5, 0.935), ncol=5, frameon=False, fontsize=4.8, handlelength=0.9, handletextpad=0.3,
               columnspacing=0.7)
    _save(fig, out, "fig_bars3d")
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig_bars3d.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--modes-csv", default=None, help="unused (values come from --numbers)")
    ap.add_argument("--sweep-csv", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: train modes traj sweep bars3d traj3d grid3d combo3d")
    ap.add_argument("--train-h", type=float, default=1.1)
    ap.add_argument("--modes-h", type=float, default=1.55)
    ap.add_argument("--traj-h", type=float, default=1.05)
    ap.add_argument("--bars-h", type=float, default=1.5)
    ap.add_argument("--traj3d-h", type=float, default=1.3)
    ap.add_argument("--grid-csv", default=None, help="plot_ral.py grid CSV (speed x difficulty), for grid3d")
    ap.add_argument("--grid-success", default=None, help="plot_ral.py grid --metric success CSV, for combo3d")
    ap.add_argument("--grid-progress", default=None, help="plot_ral.py grid --metric progress CSV, for combo3d")
    ap.add_argument("--sig-z", type=float, default=0.0, help="Gaussian smoothing of the height in fig_traj (s)")
    ap.add_argument("--sig-v", type=float, default=0.0, help="Gaussian smoothing of v_x in fig_traj (s)")
    args = ap.parse_args()
    with open(args.numbers) as fh:
        J = json.load(fh)
    for name, fn in (("train", lambda: fig_train(args.train_dir, args.out, h=args.train_h)),
                     ("modes", lambda: fig_modes(None, J, args.out, h=args.modes_h)),
                     ("traj", lambda: fig_traj(args.results, args.out, h=args.traj_h, sig_z=args.sig_z, sig_v=args.sig_v)),
                     ("sweep", lambda: fig_sweep(args.sweep_csv, J, args.out)),
                     ("bars3d", lambda: fig_bars3d(J, args.out, h=args.bars_h)),
                     ("traj3d", lambda: fig_traj3d(args.results, args.out, h=args.traj3d_h)),
                     ("grid3d", lambda: fig_grid3d(args.grid_csv, args.out, h=args.bars_h)),
                     ("combo3d", lambda: fig_combo3d(args.grid_success, args.grid_progress, args.out,
                                                     cells=("full:SP", "ppo_mlp:ORIG", "ppo_mlp_joint:ORIG", "birodiff:ORIG",
                                                            "diffuseloco:ORIG"), bar_h=0.50, legend_h=0.24, legend_ncol=3))):
        if args.only is not None and name not in args.only:
            continue
        try:
            fn()
        except Exception as e:  # one failed figure must not stop the others
            print(f"[ral_figs] {name} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
