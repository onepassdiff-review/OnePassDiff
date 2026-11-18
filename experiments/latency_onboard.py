#!/usr/bin/env python3
"""Batch-1 inference latency of every controller (latency columns of the outcomes table).

Step 1, on the training machine (needs the repo; loads the checkpoints):
  python latency_onboard.py export --out bundle.pt \
      --onepass <full seed-1 .pth> --mlp <ppo_mlp seed-1 .pth> \
      --bc <bc_birodiff_seed1/bc.pt> <bc_diffuseloco_seed1/bc.pt> \
      --obs results/obs_*.pt          # real observations dumped by run_e0.py --dump-obs

Step 2, timing (torch only; copy latency_onboard.py, onepass_kit.py, bc_baselines.py and
envs/common/diffusion_policy.py next to bundle.pt if run elsewhere):
  python latency_onboard.py time --bundle bundle.pt --device cuda --out latency_cuda.json
  OMP_NUM_THREADS=1 taskset -c 2 python latency_onboard.py time --bundle bundle.pt --device cpu --threads 1 \
      --iters 1000 --warmup 100 --out latency_cpu1.json      # single CPU core (paper: median of three runs)

Every controller is timed on the same real observations, one at a time (batch 1), so the adaptive
mode's N follows the observation. Reported: mean / median / p95 / p99 / max (ms) and, for AD, the mean
N over the observation set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for cand in (os.path.join(HERE, "..", "envs", "common"),
             os.path.join(os.environ.get("ONEPASS_REPO", os.path.join(HERE, "..")), "envs", "common"), HERE):
    if os.path.exists(os.path.join(cand, "diffusion_policy.py")) and cand not in sys.path:
        sys.path.insert(0, cand)

from onepass_kit import InferencePolicy  # noqa: E402
from bc_baselines import BCDiffusion, BCPolicy, safe_load  # noqa: E402

DIFF_MODES = ["SP", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "AD"]


# ----------------------------------------------------------------------------- export
def _actor_critic_mlp(sd_net, net_params):
    """Plain-torch copy of the rl_games actor (actor_mlp + mu), from the a2c_network state dict."""
    act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}[net_params["mlp"]["activation"]]
    idx = sorted({int(k.split(".")[1]) for k in sd_net if k.startswith("actor_mlp.") and k.endswith(".weight")})
    layers, sd = [], {}
    for j, i in enumerate(idx):
        w = sd_net[f"actor_mlp.{i}.weight"]
        layers += [nn.Linear(w.shape[1], w.shape[0]), act()]
        sd[f"{2 * j}.weight"], sd[f"{2 * j}.bias"] = w, sd_net[f"actor_mlp.{i}.bias"]
    w = sd_net["mu.weight"]
    layers.append(nn.Linear(w.shape[1], w.shape[0]))
    sd[f"{2 * len(idx)}.weight"], sd[f"{2 * len(idx)}.bias"] = w, sd_net["mu.bias"]
    return layers, sd


def export(args):
    from my_adapter import Adapter
    from omegaconf import OmegaConf
    b = {"version": 1}

    def diffusion_entry(path, what):
        ad = Adapter(path, "cpu")
        if ad.net_name != "diffusion_actor_critic":
            raise SystemExit(f"{what} must be a diffusion checkpoint")
        return {"net_params": OmegaConf.to_container(ad.cfg.train.params.network, resolve=True),
                "obs_dim": int(ad.obs_mean.numel()), "act_dim": ad.act_dim,
                "state_dict": {k: v.cpu() for k, v in ad.net.state_dict().items()},
                "obs_mean": ad.obs_mean.cpu(), "obs_var": ad.obs_var.cpu(), "tnp_head": ad.tnp_head,
                "checkpoint": os.path.basename(path)}

    if args.onepass:
        b["onepass"] = diffusion_entry(args.onepass, "--onepass")
    if args.mlp:
        ad = Adapter(args.mlp, "cpu")
        np_ = OmegaConf.to_container(ad.cfg.train.params.network, resolve=True)
        sd_net = {k: v for k, v in ad.net.state_dict().items()}
        layers, sd = _actor_critic_mlp(sd_net, np_)
        b["mlp"] = {"activation": np_["mlp"]["activation"], "state_dict": sd,
                    "obs_mean": ad.obs_mean.cpu(), "obs_var": ad.obs_var.cpu(),
                    "checkpoint": os.path.basename(args.mlp)}
    b["bc"] = {}
    for p in args.bc or []:
        ck = safe_load(p)
        b["bc"][f"{ck['arch']}"] = ck
    files = [f for pat in (args.obs or []) for f in sorted(glob.glob(pat))]
    if files:
        b["obs"] = torch.cat([safe_load(f).float() for f in files])
        b["obs_files"] = [os.path.basename(f) for f in files]
    torch.save(b, args.out)
    print(f"wrote {args.out}: {', '.join(k for k in ('onepass', 'mlp') if k in b)} "
          f"{list(b['bc'])}, {0 if 'obs' not in b else b['obs'].shape[0]} observations")


# ----------------------------------------------------------------------------- load
class Loaded:
    def __init__(self, bundle, device):
        self.dev = torch.device(device)
        self.b = bundle
        self.ad = None
        if "onepass" in bundle:
            from diffusion_policy import DiffusionActorCriticBuilder
            a = bundle["onepass"]
            net = DiffusionActorCriticBuilder.Network(a["net_params"], actions_num=a["act_dim"],
                                                      input_shape=(a["obs_dim"],), value_size=1)
            net.load_state_dict(a["state_dict"], strict=True)
            self.ad = net.to(self.dev).eval()
            self.ad_mean, self.ad_var = a["obs_mean"].to(self.dev), a["obs_var"].to(self.dev)
            self.tnp_head = a.get("tnp_head", "selfsup")
        self.mlp = None
        if "mlp" in bundle:
            m = bundle["mlp"]
            act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}[m["activation"]]
            n_lin = len([k for k in m["state_dict"] if k.endswith(".weight")])
            layers = []
            for j in range(n_lin):
                w = m["state_dict"][f"{2 * j}.weight"]
                layers.append(nn.Linear(w.shape[1], w.shape[0]))
                if j < n_lin - 1:
                    layers.append(act())
            self.mlp = nn.Sequential(*layers)
            self.mlp.load_state_dict(m["state_dict"])
            self.mlp.to(self.dev).eval()
            self.mlp_mean, self.mlp_var = m["obs_mean"].to(self.dev), m["obs_var"].to(self.dev)
        self.bc = {k: BCDiffusion.from_checkpoint(ck, self.dev) for k, ck in bundle.get("bc", {}).items()}
    @staticmethod
    def _norm(o, mean, var):
        return ((o - mean) / torch.sqrt(var + 1e-5)).clamp(-5.0, 5.0)

    def policies(self, seed=0):
        """name -> callable(obs [1, D]) -> (action, n, eta)"""
        out = {}
        if self.ad is not None:
            actor = self.ad.actor
            enc = lambda o: actor.encode(self._norm(o, self.ad_mean, self.ad_var))  # noqa: E731
            den = lambda zs, zt, x, t: actor.diffusion_net(zs, zt, x, t)              # noqa: E731
            tnp = lambda zt: self.ad.tnp_head_logits(self.tnp_head, zt)              # noqa: E731
            ab = self.ad.alpha_bar
            for m in DIFF_MODES:
                kw = dict(encode=enc, denoiser=den, alpha_bar=ab, act_dim=actor.action_dim,
                          tnp_logits=tnp, seed=seed)
                if m == "SP" or m == "AD":
                    out[f"onepass:{m}"] = InferencePolicy(mode=m, **kw)
                else:
                    out[f"onepass:{m}"] = InferencePolicy(mode="D", n_fixed=int(m[1:]), **kw)
            out["onepass:TNP-only"] = lambda o: (tnp(enc(o)[1]), None, None)
        if self.mlp is not None:
            def mlp_pol(o):
                a = self.mlp(self._norm(o, self.mlp_mean, self.mlp_var))
                return a, torch.ones(o.shape[0], dtype=torch.long, device=o.device), None
            out["ppo_mlp"] = mlp_pol
        for k, m in self.bc.items():
            out[f"{k}"] = BCPolicy(m, seed=seed)
        return out


def _sysinfo(device):
    info = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "machine": platform.machine()}
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


@torch.no_grad()
def time_one(pol, obs, n_iter, warmup, dev):
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    M = obs.shape[0]
    for i in range(warmup):
        pol(obs[i % M: i % M + 1])
    sync()
    ts, ns = [], []
    for i in range(n_iter):
        o = obs[i % M: i % M + 1]
        t0 = time.perf_counter()
        _, n, _ = pol(o)
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
        if n is not None:
            ns.append(float(n.float().mean()))
    t = torch.tensor(ts)
    r = {"mean_ms": float(t.mean()), "p50_ms": float(t.median()), "p95_ms": float(torch.quantile(t, 0.95)),
         "p99_ms": float(torch.quantile(t, 0.99)), "max_ms": float(t.max()), "n_iter": n_iter}
    if ns:
        r["mean_n"] = sum(ns) / len(ns)
    return r


def run_time(args):
    dev = torch.device(args.device)
    torch.set_num_threads(args.threads) if args.threads else None
    L = Loaded(safe_load(args.bundle), dev)
    obs = L.b.get("obs")
    if obs is None:
        print("WARNING: bundle has no real observations; using N(0,1) inputs (AD's N is then not "
              "representative). Re-export with --obs.")
        D = L.b["onepass"]["obs_dim"] if "onepass" in L.b else next(iter(L.bc.values())).obs_dim
        obs = torch.randn(256, D, generator=torch.Generator().manual_seed(0))
    obs = obs.float().to(dev)
    res = {"system": _sysinfo(dev), "n_obs": int(obs.shape[0]), "latency": {}}
    for name, pol in L.policies().items():
        if args.only and not any(name.startswith(x) for x in args.only):
            continue
        r = time_one(pol, obs, args.iters, args.warmup, dev)
        res["latency"][name] = r
        print(f"{name:22s} mean {r['mean_ms']:7.3f} ms  p99 {r['p99_ms']:7.3f}  max {r['max_ms']:7.3f}"
              + (f"  mean N {r['mean_n']:.2f}" if "mean_n" in r else ""), flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--onepass")
    e.add_argument("--mlp")
    e.add_argument("--bc", nargs="*")
    e.add_argument("--obs", nargs="*", help="obs_<terrain>.pt files (glob patterns allowed)")
    t = sub.add_parser("time")
    t.add_argument("--bundle", required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--iters", type=int, default=1000)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    t.add_argument("--only", nargs="*", help="name prefixes to time, e.g. onepass:AD ppo_mlp")
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    {"export": export, "time": run_time}[args.cmd](args)


if __name__ == "__main__":
    main()
