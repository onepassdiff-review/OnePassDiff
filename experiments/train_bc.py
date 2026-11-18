#!/usr/bin/env python3
"""Train BiRoDiff-R or DiffuseLoco-R by behaviour cloning on demos from collect_demos.py.

Torch only. Writes <out>/bc.pt (format bc_diffusion), <out>/train_log.csv and a copy of the
expert run's .hydra/ (the evaluation adapter rebuilds the environment from it).

  python train_bc.py --demos demos/ppo_mlp_seed1.pt --arch birodiff    --seed 1 \
      --out ../outputs/Biped/train/bc_birodiff_seed1
  python train_bc.py --demos demos/ppo_mlp_seed1.pt --arch diffuseloco --seed 1 \
      --out ../outputs/Biped/train/bc_diffuseloco_seed1

The defaults are the originals' settings (see bc_baselines.py). --steps / --epochs override the
training length (e.g. for a quick check).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_baselines import ARCH_DEFAULTS, EMA, BCDiffusion, DemoWindows, safe_load  # noqa: E402


def lr_at(step, total, base, warmup, sched):
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    if sched == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--arch", required=True, choices=sorted(ARCH_DEFAULTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="gradient steps (overrides the default length)")
    ap.add_argument("--epochs", type=float, default=None, help="epochs over the training split (alternative)")
    ap.add_argument("--batch", type=int, default=None, help="override the batch size (tests only)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    cfg = dict(ARCH_DEFAULTS[args.arch])
    if args.batch:
        cfg["batch"] = args.batch
    demos = safe_load(args.demos)
    E, S, D = demos["obs"].shape
    A = demos["act"].shape[-1]
    Fd = int(demos["frame_dim"])

    # split by robot (whole trajectories), seeded
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(E, generator=g)
    n_val = max(1, int(round(cfg["val_frac"] * E)))
    val_ids, tr_ids = perm[:n_val].sort().values, perm[n_val:].sort().values
    tr, va = DemoWindows(demos, dev, tr_ids), DemoWindows(demos, dev, val_ids)
    print(f"[bc] {args.arch}: {len(tr)} training / {len(va)} validation pairs, obs {D}, frame {Fd}, act {A}")

    model = BCDiffusion(args.arch, D, A, Fd, cfg).to(dev)
    # normalisers from the training split only
    if args.arch == "birodiff":
        model.obs_norm.fit(tr.obs[tr.idx[:, 0], tr.idx[:, 1]].float())
    else:
        model.obs_norm.fit(tr.frames_all()[tr.idx[:, 0], tr.idx[:, 1]].float())
    model.act_norm.fit(tr.act[tr.idx[:, 0], tr.idx[:, 1]])
    print(f"[bc] denoiser parameters: {model.n_params():,}")

    B = cfg["batch"]
    if args.steps is not None:
        total = args.steps
    elif args.epochs is not None:
        total = int(math.ceil(args.epochs * len(tr) / B))
    elif "steps" in cfg:
        total = cfg["steps"]
    else:
        total = int(math.ceil(cfg["epochs"] * len(tr) / B))
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.Adam(params, lr=cfg["lr"])
    ema = EMA(model, cfg["ema_power"], cfg["ema_max"]) if cfg.get("ema") else None
    warm = min(cfg.get("warmup", 0), total // 10) if cfg.get("warmup") else 0

    os.makedirs(args.out, exist_ok=True)
    hydra_src = demos.get("hydra_dir")
    if hydra_src and os.path.isdir(hydra_src):
        dst = os.path.join(args.out, ".hydra")
        if not os.path.exists(dst):
            shutil.copytree(hydra_src, dst)
    else:
        print("[bc] WARNING: the demos file has no expert .hydra dir; set ONEPASS_HYDRA_CFG at evaluation")
    log_f = open(os.path.join(args.out, "train_log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "lr", "train_loss", "val_loss", "seconds"])

    gv = torch.Generator(device=dev).manual_seed(12345)
    val_sel = torch.randint(0, len(va), (min(len(va), 20000),), device=dev, generator=gv)

    def val_loss(m):
        m.eval()
        torch.manual_seed(0)
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, val_sel.numel(), B):
                c, a = va.batch(val_sel[i:i + B], args.arch, cfg.get("h", 8), cfg.get("n_pred", 4))
                tot += float(m.loss(c, a)) * a.shape[0]
                n += a.shape[0]
        m.train()
        return tot / max(1, n)

    t0, run = time.time(), 0.0
    model.train()
    for step in range(total):
        lr = lr_at(step, total, cfg["lr"], warm, cfg["lr_schedule"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        sel = torch.randint(0, len(tr), (B,), device=dev)
        c, a = tr.batch(sel, args.arch, cfg.get("h", 8), cfg.get("n_pred", 4))
        loss = model.loss(c, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ema is not None:
            ema.update(model)
        run = 0.98 * run + 0.02 * float(loss) if step else float(loss)
        if (step + 1) % args.eval_every == 0 or step + 1 == total:
            vl = val_loss(ema.shadow if ema is not None else model)
            log.writerow([step + 1, f"{lr:.3e}", f"{run:.5f}", f"{vl:.5f}", f"{time.time() - t0:.0f}"])
            log_f.flush()
            print(f"[bc] step {step + 1}/{total} lr {lr:.2e} train {run:.4f} val {vl:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    final = ema.shadow if ema is not None else model
    ck = final.checkpoint({"train_seed": args.seed, "demos": os.path.abspath(args.demos), "steps": total,
                           "expert_checkpoint": demos.get("expert_checkpoint", ""),
                           "pairs_per_kind": int(demos.get("pairs_per_kind", -1)),
                           "n_params": final.n_params()})
    torch.save(ck, os.path.join(args.out, "bc.pt"))
    log_f.close()
    print(f"[bc] wrote {os.path.join(args.out, 'bc.pt')}")


if __name__ == "__main__":
    main()
