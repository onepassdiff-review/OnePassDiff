#!/usr/bin/env python3
"""Pool the step counts N chosen by an adaptive mode across ALL terrains of one checkpoint.

Writes adstats_<method>_seed<seed>_<token>.json with
  mean_n  : mean N over all logged control steps (AD-mean, AD-random)
  n_pool  : a random subsample of the pooled N values (AD-shuffle)
  per_terrain : mean N and eta per terrain (for the eta/N figure)
"""
import argparse
import glob
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--token", default="AD")
    ap.add_argument("--pool-size", type=int, default=100_000)
    args = ap.parse_args()

    tag = f"{args.method}_seed{args.train_seed}"
    files = sorted(glob.glob(os.path.join(args.out, f"steplog_{tag}_{args.token}_*.pt")))
    if not files:
        raise SystemExit("no step logs found; run pass 1 with the AD mode first")
    pooled, per_terrain = [], {}
    for f in files:
        terrain = os.path.basename(f)[len(f"steplog_{tag}_{args.token}_"):-3]
        log = torch.load(f)
        n = torch.cat([e["n"][e["alive"]] for e in log])
        etas = [e["eta"][e["alive"]] for e in log if e["eta"] is not None]
        eta = torch.cat(etas) if etas else None
        pooled.append(n)
        per_terrain[terrain] = {
            "mean_n": float(n.float().mean()),
            "n_hist": {int(k): int(v) for k, v in zip(*torch.unique(n, return_counts=True))},
            "mean_eta": None if eta is None else float(eta.mean()),
            "std_eta": None if eta is None else float(eta.std()),
        }
    pool = torch.cat(pooled)
    g = torch.Generator().manual_seed(0)
    sub = pool[torch.randperm(pool.numel(), generator=g)[: args.pool_size]]
    out = {"mean_n": float(pool.float().mean()), "n_pool": sub.tolist(), "per_terrain": per_terrain,
           "n_terrains": len(files)}
    path = os.path.join(args.out, f"adstats_{tag}_{args.token}.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"pooled {len(files)} terrains, mean N = {out['mean_n']:.3f} -> {path}")
    for t, d in per_terrain.items():
        print(f"  {t:15s} mean N {d['mean_n']:.2f}  eta {d['mean_eta']}")


if __name__ == "__main__":
    main()
