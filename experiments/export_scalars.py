#!/usr/bin/env python3
"""Export training scalars from the TensorBoard event files: no TensorBoard window or screenshots needed.

  cd experiments
  python export_scalars.py --logdir ../outputs/Biped/train --runs 'onepass_*' --out scalars_out

Writes, for every run directory that contains event files:
  <out>/scalars/<run>.csv       every scalar: tag, step, wall_time, value
  <out>/plots/<group>.png       key curves, all runs overlaid (if matplotlib is available)
  <out>/summary.txt             iterations, seconds per iteration, start/end values of key scalars,
                                and automatic sanity checks
Needs the `tensorboard` Python package (installed with rl_games; otherwise `pip install tensorboard`).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import math
import os
import statistics as st
from collections import defaultdict

KEY_TAGS = [  # (group, tag pattern) shown in plots and summary
    ("reward", "rewards/iter"), ("reward", "rewards/step"),
    ("episode_length", "episode_lengths/iter"),
    ("curriculum", "Episode/terrain_level"), ("curriculum", "Episode/terrain_level_*"),
    ("onepass_losses", "onepass/loss_dn"), ("onepass_losses", "onepass/loss_tnp"),
    ("onepass_losses", "onepass/loss_joint"),
    ("onepass_eta", "onepass/eta_*"), ("onepass_target", "onepass/target_*"),
    ("onepass_misc", "onepass/joint_active_frac"), ("onepass_misc", "onepass/awr_w_clipped_frac"),
    ("onepass_misc", "onepass/kappa"),
    ("ppo", "losses/a_loss"), ("ppo", "losses/c_loss"), ("ppo", "losses/entropy"),
    ("ppo", "info/kl"), ("ppo", "info/lr"),
]


def find_runs(logdir, pattern):
    runs = {}
    for ev in glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True):
        rel = os.path.relpath(ev, logdir).split(os.sep)
        name = rel[0]
        if fnmatch.fnmatch(name, pattern):
            runs.setdefault(name, set()).add(os.path.dirname(ev))
    return runs


def load_scalars(dirs):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    data = defaultdict(list)
    for d in sorted(dirs):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            data[tag] += [(e.step, e.wall_time, e.value) for e in ea.Scalars(tag)]
    for tag in data:
        data[tag].sort(key=lambda x: (x[1], x[0]))
    return data


def match(tags, pat):
    return sorted(t for t in tags if fnmatch.fnmatch(t, pat))


def ends(series, frac=0.1):
    """mean of the first and last 10% of the values"""
    v = [x[2] for x in series if not math.isnan(x[2])]
    if not v:
        return float("nan"), float("nan")
    k = max(1, int(len(v) * frac))
    return st.mean(v[:k]), st.mean(v[-k:])


def sec_per_iter(data):
    for tag in ("Episode/terrain_level", "rewards/iter", "episode_lengths/iter"):
        s = data.get(tag)
        if s and len(s) > 10:
            (s0, w0, _), (s1, w1, _) = s[len(s) // 10], s[-1]
            if s1 > s0:
                return (w1 - w0) / (s1 - s0), s[-1][0], tag
    return float("nan"), None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default="../outputs/Biped/train")
    ap.add_argument("--runs", default="*", help="glob on run directory names, e.g. 'onepass_*'")
    ap.add_argument("--out", default="scalars_out")
    args = ap.parse_args()

    runs = find_runs(args.logdir, args.runs)
    if not runs:
        raise SystemExit(f"no event files under {args.logdir} matching {args.runs}")
    os.makedirs(os.path.join(args.out, "scalars"), exist_ok=True)
    all_data, lines = {}, []
    for name in sorted(runs):
        data = load_scalars(runs[name])
        all_data[name] = data
        with open(os.path.join(args.out, "scalars", f"{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tag", "step", "wall_time", "value"])
            for tag in sorted(data):
                w.writerows([(tag, s, f"{t:.3f}", v) for s, t, v in data[tag]])
        spi, last, via = sec_per_iter(data)
        lines.append(f"== {name}")
        lines.append(f"   last logged iteration: {last}   seconds per iteration: {spi:.2f} (from {via})")
        shown = set()
        for _, pat in KEY_TAGS:
            for tag in match(data, pat):
                if tag in shown:
                    continue
                shown.add(tag)
                a, b = ends(data[tag])
                lines.append(f"   {tag:42s} start {a:12.4f}   end {b:12.4f}")
        lines.append(f"   ({len(data)} scalar tags in total, all in scalars/{name}.csv)")

    # automatic sanity checks
    lines.append("\n== sanity checks")

    def chk(label, ok):
        lines.append(f"   [{'ok ' if ok else '?? '}] {label}")

    for name, data in all_data.items():
        if "onepass/loss_dn" in data:
            a, b = ends(data["onepass/loss_dn"])
            chk(f"{name}: loss_dn falls ({a:.4f} -> {b:.4f})", b < a)
        if "onepass/loss_tnp" in data:
            a, b = ends(data["onepass/loss_tnp"])
            chk(f"{name}: loss_tnp falls ({a:.4f} -> {b:.4f})", b < a)
        # per-kind tags only: eta_mean / eta_std (and target_mean) are summaries, not terrain kinds
        for what in ("eta", "target"):
            per_kind = {t: ends(data[t])[1] for t in match(data, f"onepass/{what}_*")
                        if t.rsplit("/", 1)[1] not in (f"{what}_mean", f"{what}_std")}
            if len(per_kind) > 1:
                spread = max(per_kind.values()) - min(per_kind.values())
                chk(f"{name}: {what} differs between terrain kinds (spread of {len(per_kind)} kind means "
                    f"{spread:.3f})", spread > 0.02)
        if ("onepass/loss_dn" in data or "onepass/loss_joint" in data):
            if "onepass/aux_minibatches" in data:
                lo = min(v for _, _, v in data["onepass/aux_minibatches"])
                chk(f"{name}: M1/M3 losses reached the gradient in every iteration (min minibatches {lo:.0f})", lo > 0)
            else:
                chk(f"{name}: M1/M3 losses reached the gradient (no aux_minibatches tag: check the "
                    f"rl_games version, see onepass_agent.py)", False)
        if "onepass/loss_joint" in data:
            a, b = ends(data["onepass/loss_joint"])
            chk(f"{name}: joint-limit loss on targets does not grow ({a:.4f} -> {b:.4f})", b <= max(a, 1e-3) * 1.5)
        if "Episode/terrain_level" in data:
            a, b = ends(data["Episode/terrain_level"])
            chk(f"{name}: terrain level rises ({a:.2f} -> {b:.2f})", b > a)
    rew = {n: ends(d["rewards/iter"])[1] for n, d in all_data.items() if "rewards/iter" in d}
    pil = [n for n in rew if "pilot" in n]
    po = [n for n in rew if "ppo_only" in n]
    if pil and po:
        r_p, r_o = rew[pil[0]], rew[po[0]]
        chk(f"pilot reward tracks ppo_only (end {r_p:.2f} vs {r_o:.2f})", r_p >= 0.9 * r_o if r_o > 0 else r_p >= r_o)
    text = "\n".join(lines)
    open(os.path.join(args.out, "summary.txt"), "w").write(text + "\n")
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: CSVs and summary written, no plots")
        return
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    groups = defaultdict(list)
    for g, pat in KEY_TAGS:
        groups[g].append(pat)
    for g, pats in groups.items():
        tags = sorted({t for d in all_data.values() for p in pats for t in match(d, p)})
        if not tags:
            continue
        n = len(tags)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
        for ax, tag in zip(axes.flat, tags):
            for name, d in all_data.items():
                if tag in d:
                    ax.plot([x[0] for x in d[tag]], [x[2] for x in d[tag]], label=name, lw=1)
            ax.set_title(tag, fontsize=9)
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "plots", f"{g}.png"), dpi=110)
        plt.close(fig)
    print(f"wrote {args.out}/summary.txt, scalars/*.csv, plots/*.png")


if __name__ == "__main__":
    main()
