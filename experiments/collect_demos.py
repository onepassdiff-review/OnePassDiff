#!/usr/bin/env python3
"""Collect demonstrations for BiRoDiff-R / DiffuseLoco-R from a trained expert (Isaac Gym).

The expert is the capacity-matched PPO-MLP (ppo_mlp seed 1, final checkpoint). It runs in its own
TRAINING environment, rebuilt from the run's .hydra config: the same mixed-terrain curriculum,
domain randomisation, observation noise and command distribution. Robots start on all difficulty
levels. The expert executes its deterministic action, which is also the recorded label (the same
action the evaluation harness executes for PPO-MLP). Stepping stones (proportion 0 in training)
and the evaluation suite's held-out terrains are never seen.

Exactly --pairs-per-kind state-action pairs are kept for every terrain kind with robots on it.

  cd OnePassDiff/experiments
  CK=$(python -c "from my_adapter import Adapter as A; print(A.find_checkpoint('../outputs/Biped/train/onepass_ppo_mlp_seed1', 5000))")
  python collect_demos.py --expert "$CK" --out demos/ppo_mlp_seed1.pt

Output (torch.save dict): obs [E, S, D] fp16 (stacked observation, newest frame first),
act [E, S, A], done [E, S] (True = last step of an episode), valid [E, S], kind [E], level [E, S]
(uint8), frame_dim, kind_names, pairs_per_kind, expert_checkpoint, hydra_dir.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_adapter import Adapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", required=True, help="PPO-MLP checkpoint (.pth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-per-kind", type=int, default=250_000)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from envs.common.onepass_agent import KIND_NAMES, env_terrain_kinds

    ad = Adapter(args.expert, args.device)
    if ad.net_name != "actor_critic":
        print(f"[demos] note: expert network is {ad.net_name}; ORIG mode is used as the expert action")
    env = ad.make_train_env(args.num_envs, args.seed)
    task = ad.task
    kinds = env_terrain_kinds(task)
    if kinds is None:
        raise SystemExit("the expert's training config has no terrain curriculum; cannot label kinds")
    kinds = torch.as_tensor(kinds, dtype=torch.long)
    present = sorted(set(kinds.tolist()))
    n_per = {k: int((kinds == k).sum()) for k in present}
    S = max(math.ceil(args.pairs_per_kind / n_per[k]) for k in present)
    print("[demos] robots per kind:", {KIND_NAMES[k]: n_per[k] for k in present}, f"-> {S} steps")

    frames = int(ad.cfg.task.env.get("num_stacked_obs_frame", 1))
    policy = ad.new_original_policy(seed=args.seed)
    obs = ad.reset(env, args.seed)
    E, D = obs.shape
    obs_buf = torch.empty(E, S, D, dtype=torch.float16)
    act_buf = torch.empty(E, S, ad.act_dim)
    done_buf = torch.zeros(E, S, dtype=torch.bool)
    lvl_buf = torch.zeros(E, S, dtype=torch.uint8)
    with torch.no_grad():
        for s in range(S):
            a = policy(obs)[0]
            obs_buf[:, s] = obs.half().cpu()
            act_buf[:, s] = a.cpu()
            if hasattr(task, "terrain_levels"):
                lvl_buf[:, s] = task.terrain_levels.to(torch.uint8).cpu()
            obs, done, _ = ad.step(env, a)
            done_buf[:, s] = done.cpu()
            if s % 200 == 0:
                print(f"[demos] step {s}/{S}", flush=True)

    # exactly pairs_per_kind per kind: the first ceil(N / E_k) steps of each robot, then trim
    valid = torch.zeros(E, S, dtype=torch.bool)
    for k in present:
        ids = (kinds == k).nonzero().flatten()
        s_k = math.ceil(args.pairs_per_kind / len(ids))
        m = torch.zeros(len(ids), S, dtype=torch.bool)
        m[:, :s_k] = True
        flat = m.flatten()
        extra = int(flat.sum()) - args.pairs_per_kind
        if extra > 0:
            on = flat.nonzero().flatten()
            flat[on[-extra:]] = False
        valid[ids] = flat.view(len(ids), S)
    hydra_dir = os.path.dirname(ad.cfg_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"obs": obs_buf, "act": act_buf, "done": done_buf, "valid": valid, "kind": kinds,
                "level": lvl_buf, "frame_dim": D // frames, "num_stacked": frames,
                "kind_names": list(KIND_NAMES), "pairs_per_kind": args.pairs_per_kind,
                "expert_checkpoint": os.path.abspath(args.expert), "hydra_dir": hydra_dir,
                "seed": args.seed}, args.out)
    for k in present:
        print(f"[demos] {KIND_NAMES[k]:15s} pairs {int(valid[kinds == k].sum())}  "
              f"episodes ended {int(done_buf[kinds == k].sum())}")
    ad.close(env)
    print(f"[demos] wrote {args.out}")


if __name__ == "__main__":
    main()
