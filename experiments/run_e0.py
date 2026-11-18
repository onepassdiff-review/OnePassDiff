#!/usr/bin/env python3
"""E0 / benchmark evaluation harness.

Evaluates one checkpoint under several inference modes on the frozen terrain suite, with the
same episodes (terrain seed, initial states) for every mode. Writes one CSV row per episode,
plus per-step eta/N logs for the step-predictor figures and batch-1 latency per mode.

Isaac Gym usually allows ONE simulation per process, so run one terrain per invocation
(launch_eval.sh does this). Within a process the env is created once and every mode is replayed
on it after a seeded reset, so all modes see the same episodes.

Example (one checkpoint), two passes per terrain:
  # pass 1: modes that need nothing else
  python run_e0.py --adapter my_adapter.py --checkpoint <run dir>/nn/<checkpoint>.pth \
      --method ppo_only --train-seed 42 --out results/ --terrains rough \
      --modes ORIG SP D1 D2 D3 D5 D7 D8 D10 AD ES
  # after pass 1 has run on ALL terrains: pool AD step counts across terrains
  python pool_ad_stats.py --out results/ --method ppo_only --train-seed 42 --token AD
  # pass 2: compute-matched controls
  python run_e0.py ... --terrains rough --modes AD-mean AD-shuffle AD-random \
      --ad-stats results/adstats_ppo_only_seed42_AD.json

Mode tokens:
  ORIG        the checkpoint's own policy (adapter.original_policy): DiffusionPolicy.sample() for
              diffusion checkpoints, the action mean for PPO-MLP, and the
              baseline's own DDPM sampler for BiRoDiff-R / DiffuseLoco-R (bc.pt)
  SP          single pass at t=0, x=0 (the policy PPO optimised)
  RP          the rollout policy itself: SP + Gaussian noise with the std PPO executed (Adapter.action_std)
  D<N>        deterministic DDIM with fixed N (e.g. D7)
  AD          adaptive N from the step predictor
  AD-mean     fixed N = rounded mean N of AD over all terrains        (needs --ad-stats)
  AD-shuffle  N drawn from the pooled AD N values of all terrains     (needs --ad-stats)
  AD-random   N in {floor(m), ceil(m)} with mean m = AD mean          (needs --ad-stats)
  ES          training-free early stopping (--tau)
Append "@0" to any DDIM mode to use x_N = 0 instead of x_N ~ N(0, I), e.g. D7@0, AD@0.
Append "-retest" to any token to run that policy again under its own label, e.g. SP-retest as the last mode: the
  paired difference SP vs SP-retest is the run-to-run variation of the simulator (GPU contact on mesh terrain is not
  bitwise reproducible; the plane is) plus any effect of the mode order.

Further experiments:
  speed sweep       ONEPASS_EVAL_VX=0.3 python run_e0.py ... --out results_speed/v0.3   (column cmd_speed)
  difficulty sweep  --terrains sweep_slope_6 ... sweep_slope_30 sweep_stair_40 ... sweep_stair_220 (my_adapter)
  trajectories      --traj-envs 4   -> traj_<method>_seed<s>_<mode>_<terrain>.pt
  learning curves   --curve-checkpoints <ck_500> <ck_1000> ...   (one simulation per terrain, weights swapped)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time

# Isaac Gym must be imported before torch. Harmless when Isaac Gym is not installed (mock tests).
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onepass_kit import InferencePolicy, time_policy  # noqa: E402

DEFAULT_TERRAINS = ["flat", "rough", "slope10", "slope18", "stair_up", "stair_down", "discrete",
                    "heldout_slope", "heldout_stairs", "heldout_gaps"]
FIELDS = ["method", "train_seed", "checkpoint", "mode", "terrain", "episode", "success", "fell",
          "seconds_alive", "progress", "vel_rmse", "mean_n", "mean_eta", "dev_from_sp", "cmd_speed",
          "start_x", "start_y",
          "cot", "joint_viol_per_min", "limit_time_pct", "max_exc_rad", "target_viol_pct", "jerk_rms",
          "path_len", "footsteps", "stumble_pct", "term_cause"]
MQ_FIELDS = FIELDS[FIELDS.index("cot"):FIELDS.index("path_len")]
LOCO_FIELDS = FIELDS[FIELDS.index("path_len"):]
# Locomotion outcome fields (NaN if the adapter lacks term_cause/foot_contact/foot_stumble). Raw
# per-episode quantities only; SPL, fall/collision shares and step length are derived in analyze_results.py:
#   path_len     horizontal base path length while alive, sum of |xy_k+1 - xy_k| at the control rate [m]
#   footsteps    touchdowns of either foot (rising edges of the task's filtered foot contact, its own threshold)
#   stumble_pct  % of control steps with a foot stumble as the task defines it (horizontal foot force > 5 N while
#                the vertical force is < 1 N: the foot pushes into a riser or an edge without bearing load)
#   term_cause   0 = no termination, 1 = fall (base tilt beyond the task limit or base contact), 2 = collision (a leg
#                link other than the feet in contact while the base was upright), 3 = terminated, cause not classified
RETEST = "-retest"   # suffix: "SP-retest" replays SP after the other modes (run-to-run variation, order effects)
MIN_COT_DIST = 1.0   # [m] CoT is NaN for episodes that covered less than this (falls at the start, standing still)
# Joint limits (URDF limits = the ones L_joint uses; the task config sets no soft margin):
#   joint_viol_per_min  onsets per minute: control steps where some measured joint is beyond its limit and none was
#                       at the previous step (a 5-step excursion counts once)
#   limit_time_pct      % of control steps with some measured joint beyond its limit
#   max_exc_rad         largest excursion beyond a limit in the episode [rad] (tells solver-level contact with the
#                       joint stop, ~1e-3 rad, from real overshoot)
#   target_viol_pct     % of control steps whose commanded PD target lies outside the limits (what L_joint acts on)


def load_adapter(path, checkpoint, device):
    spec = importlib.util.spec_from_file_location("user_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Adapter(checkpoint, device)


def make_policy(ad, token, ad_stats, tau, seed):
    if token.endswith(RETEST):   # test-retest pass: the same policy again, under its own label
        token = token[:-len(RETEST)]
    x0 = token.endswith("@0")
    base = token[:-2] if x0 else token
    common = dict(encode=ad.encode, denoiser=ad.denoiser, alpha_bar=ad.alpha_bar, act_dim=ad.act_dim,
                  tnp_logits=getattr(ad, "tnp_logits", None), x_init_zero=x0, seed=seed)
    if base == "ORIG":
        fn = ad.new_original_policy(seed) if hasattr(ad, "new_original_policy") else ad.original_policy
        return InferencePolicy(mode="custom", custom_fn=fn, **common)
    if base == "SP":
        return InferencePolicy(mode="SP", **common)
    if base == "RP":   # the stochastic rollout policy PPO optimized (single pass + executed Gaussian noise)
        if not hasattr(ad, "action_std"):
            raise SystemExit("mode RP needs Adapter.action_std()")
        return InferencePolicy(mode="RP", action_std=ad.action_std(), **common)
    if base.startswith("D") and base[1:].isdigit():
        return InferencePolicy(mode="D", n_fixed=int(base[1:]), **common)
    if base == "AD":
        return InferencePolicy(mode="AD", **common)
    if base in ("AD-mean", "AD-shuffle", "AD-random"):
        if ad_stats is None:
            raise SystemExit(f"{token} needs --ad-stats (see pool_ad_stats.py)")
        return InferencePolicy(mode=base, mean_n=ad_stats["mean_n"], n_pool=ad_stats["n_pool"], **common)
    if base == "ES":
        return InferencePolicy(mode="ES", tau=tau, **common)
    raise SystemExit(f"unknown mode token {token}")


def _traj_sample(ad, env, alive, n, a, K):
    """One control step of the trajectory log for the first K robots (see --traj-envs)."""
    sel = slice(0, K)
    pos = ad.base_pos(env)[sel] if hasattr(ad, "base_pos") else torch.cat(
        [ad.base_xy(env)[sel], torch.full((K, 1), float("nan"), device=alive.device)], -1)
    snap = lambda x: x.detach().float().cpu().clone()   # noqa: E731  (.cpu() of a CPU tensor is a view)
    rec = {"alive": alive[sel].cpu().clone(), "pos": snap(pos), "vel": snap(ad.base_lin_vel_xy(env)[sel]),
           "cmd": snap(ad.command_xy(env)[sel]), "n": n[sel].cpu().clone(), "a": snap(a[sel])}
    if hasattr(ad, "joint_state"):
        try:
            rec["q"] = snap(ad.joint_state(env)[0][sel])
        except (AttributeError, TypeError):
            pass
    return rec


def _stack_traj(traj, dt):
    """list of per-step dicts -> dict of [T, K, ...] tensors plus dt."""
    out = {k: torch.stack([r[k] for r in traj]) for k in traj[0]}
    out["dt"] = dt
    return out


@torch.no_grad()
def run_episodes(ad, env, policy, sp_policy, args, step_log, obs_dump=None, traj=None):
    obs = ad.reset(env, args.eval_seed)
    B, dev = obs.shape[0], obs.device
    H = int(round(args.episode_seconds / ad.dt))
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    fell = torch.zeros(B, dtype=torch.bool, device=dev)
    steps_alive = torch.zeros(B, device=dev)
    start_xy = ad.base_xy(env).clone()
    last_xy = start_xy.clone()
    heading = ad.heading_xy(env).clone()
    cmd_speed = ad.command_xy(env).norm(dim=-1).clone()
    vel_sq = torch.zeros(B, device=dev)
    n_sum = torch.zeros(B, device=dev)
    eta_sum = torch.zeros(B, device=dev)
    have_eta = False
    dev_sum = torch.zeros(B, device=dev)
    # motion quality (NaN if the adapter has no joint accessors, e.g. the mock adapter)
    mq = all(hasattr(ad, f) for f in ("joint_state", "total_weight", "joint_limits"))
    if mq:
        try:   # the adapter may wrap a task without joint tensors (test fakes): then the metrics are NaN
            mg = ad.total_weight(env)
            q_lo, q_hi = ad.joint_limits(env)
            ad.joint_state(env)
        except (AttributeError, TypeError):
            mq = False
    if mq:
        q_prev = [None, None, None]                  # q at t-1, t-2, t-3 (control steps)
        run_len = torch.zeros(B, device=dev)         # consecutive alive steps (jerk needs 4 samples)
        energy = torch.zeros(B, device=dev)          # integral of sum_j |tau_j qd_j| dt  [J]
        viol_steps = torch.zeros(B, device=dev)
        viol_events = torch.zeros(B, device=dev)
        v_prev = torch.zeros(B, dtype=torch.bool, device=dev)
        max_exc = torch.zeros(B, device=dev)
        tgt_steps = torch.zeros(B, device=dev)
        jerk_sq = torch.zeros(B, device=dev)
        jerk_n = torch.zeros(B, device=dev)
        has_tgt = hasattr(ad, "joint_target")   # the task creates its PD-target tensor in the first step
    loco = all(hasattr(ad, f) for f in ("term_cause", "foot_contact", "foot_stumble"))
    if loco:
        try:   # test fakes may lack the task tensors: then the locomotion outcome fields are NaN
            fc_prev = ad.foot_contact(env).bool().clone()
            ad.term_cause(env)
            ad.foot_stumble(env)
        except (AttributeError, TypeError):
            loco = False
    if loco:
        path_len = torch.zeros(B, device=dev)
        footsteps = torch.zeros(B, device=dev)
        stumble_steps = torch.zeros(B, device=dev)
        cause = torch.zeros(B, dtype=torch.long, device=dev)
    K = min(getattr(args, "traj_envs", 0), B) if traj is not None else 0
    for k in range(H):
        last_xy = torch.where(alive.unsqueeze(-1), ad.base_xy(env), last_xy)
        if obs_dump is not None and k % 25 == 0:
            obs_dump.append(obs[alive][:8].cpu())
        a, n, eta = policy(obs)
        a_sp = sp_policy(obs)[0] if sp_policy is not None else a
        m = alive.float()
        n_sum += m * n.float()
        if eta is not None:
            have_eta = True
            eta_sum += m * eta.float()
        dev_sum += m * (a - a_sp).norm(dim=-1)
        v_err = (ad.base_lin_vel_xy(env) - ad.command_xy(env)).pow(2).sum(-1)
        vel_sq += m * v_err
        steps_alive += m
        if step_log is not None and k % args.log_every == 0:
            sel = slice(0, min(args.log_envs, B))
            step_log.append({"k": k, "alive": alive[sel].cpu().clone(), "n": n[sel].cpu().clone(),   # clone: CPU views
                             "eta": None if eta is None else eta[sel].cpu().clone(),
                             "x": ad.base_xy(env)[sel, 0].cpu().clone()})
        if K:   # state before the step, the action applied in it, N used for it
            traj.append(_traj_sample(ad, env, alive, n, a, K))
        obs, done, timeout = ad.step(env, a)
        newly = done & alive
        fell |= newly & ~timeout
        alive &= ~done
        if loco:   # after the step; robots that ended in it were reset by the task and are masked out
            ml = alive.float()
            path_len += ml * (ad.base_xy(env) - last_xy).norm(dim=-1)
            fc = ad.foot_contact(env).bool()
            footsteps += ml * (fc & ~fc_prev).sum(-1).float()
            fc_prev = fc.clone()
            stumble_steps += ml * ad.foot_stumble(env).float()
            cause = torch.where(newly & ~timeout, ad.term_cause(env).long(), cause)
        if mq:   # state after the step; only robots still in their episode count (done envs were reset)
            q, qd, tau = ad.joint_state(env)
            ma = alive.float()
            energy += ma * (tau * qd).abs().sum(-1) * ad.dt
            exc = torch.maximum(q_lo - q, q - q_hi).clamp(min=0).max(-1).values   # [B] rad beyond a limit
            v = exc > 0
            viol_steps += ma * v.float()
            viol_events += ma * (v & ~v_prev).float()
            v_prev = v & alive
            max_exc = torch.maximum(max_exc, ma * exc)
            if has_tgt:
                try:
                    qt = ad.joint_target(env)
                except (AttributeError, TypeError):
                    has_tgt = False
                else:
                    tgt_steps += ma * ((qt < q_lo) | (qt > q_hi)).any(-1).float()
            run_len = (run_len + 1) * ma
            if q_prev[2] is not None:
                jerk = (q - 3 * q_prev[0] + 3 * q_prev[1] - q_prev[2]) / ad.dt ** 3
                ok = (run_len >= 4).float()
                jerk_sq += ok * jerk.pow(2).mean(-1)
                jerk_n += ok
            q_prev = [q.clone(), q_prev[0], q_prev[1]]
        if not alive.any():
            break
    last_xy = torch.where(alive.unsqueeze(-1), ad.base_xy(env), last_xy)
    secs = steps_alive * ad.dt
    disp = ((last_xy - start_xy) * heading).sum(-1)
    progress = disp / (cmd_speed * args.episode_seconds).clamp(min=1e-6)
    denom = steps_alive.clamp(min=1)
    out = {
        "success": (~fell).cpu(), "fell": fell.cpu(), "seconds_alive": secs.cpu(),
        "progress": progress.cpu(), "vel_rmse": (vel_sq / denom).sqrt().cpu(),
        "mean_n": (n_sum / denom).cpu(),
        "mean_eta": (eta_sum / denom).cpu() if have_eta else torch.full((B,), float("nan")),
        "dev_from_sp": (dev_sum / denom).cpu(),
        "cmd_speed": cmd_speed.cpu(),
        "start_x": start_xy[:, 0].cpu(), "start_y": start_xy[:, 1].cpu(),   # pairing audit: same for every mode
    }
    nan = torch.full((B,), float("nan"))
    if mq:
        dist = disp.clamp(min=0)
        cot = energy / (mg * dist.clamp(min=1e-6))
        out["cot"] = torch.where(dist >= MIN_COT_DIST, cot, torch.full_like(cot, float("nan"))).cpu()
        out["joint_viol_per_min"] = (viol_events / (denom * ad.dt / 60.0)).cpu()
        out["limit_time_pct"] = (100.0 * viol_steps / denom).cpu()
        out["max_exc_rad"] = max_exc.cpu()
        out["target_viol_pct"] = (100.0 * tgt_steps / denom).cpu() if has_tgt else nan.clone()
        out["jerk_rms"] = torch.where(jerk_n > 0, (jerk_sq / jerk_n.clamp(min=1)).sqrt(),
                                      torch.full_like(jerk_sq, float("nan"))).cpu()
    else:
        for f in MQ_FIELDS:
            out[f] = nan.clone()
    if loco:
        out["path_len"] = path_len.cpu()
        out["footsteps"] = footsteps.cpu()
        out["stumble_pct"] = (100.0 * stumble_steps / denom).cpu()
        out["term_cause"] = cause.float().cpu()
    else:
        for f in LOCO_FIELDS:
            out[f] = nan.clone()
    return out


def run_isolated(argv, terrains):
    """Run each terrain in its own process. Isaac Gym cannot hold a second simulation in one process: the second
    LeggedTerrain fails to build (with or without destroy_sim), and destroy_sim itself can hang on the CPU
    pipeline. The children append to the same episodes CSV and merge into the same latency JSON; latency is timed
    in the first child only. A failed terrain is reported and the others still run."""
    import subprocess
    rest, skip = [], False
    for tok in argv:                       # drop "--terrains a b c" (or "--terrains=a"), keep everything else
        if tok == "--terrains" or tok.startswith("--terrains="):
            skip = tok == "--terrains"
            continue
        if skip and tok.startswith("--"):
            skip = False
        if not skip:
            rest.append(tok)
    failed = []
    for i, terrain in enumerate(terrains):
        cmd = [sys.executable, os.path.abspath(__file__), *rest, "--terrains", terrain, "--no-isolate"]
        if i > 0:
            cmd.append("--skip-latency")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[run_e0] terrain {terrain} FAILED (exit {rc})", flush=True)
            failed.append(terrain)
    if failed:
        print(f"[run_e0] failed terrains: {' '.join(failed)}", flush=True)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", required=True, help="label, e.g. ppo_only, full, ppo_mlp")
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--modes", nargs="+", default=["SP", "D5", "D7", "D10", "AD", "AD-mean", "AD-shuffle"])
    ap.add_argument("--terrains", nargs="+", default=DEFAULT_TERRAINS)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--episode-seconds", type=float, default=60.0)
    ap.add_argument("--eval-seed", type=int, default=12345, help="same for every method and mode")
    ap.add_argument("--tau", type=float, default=0.01, help="early-stopping threshold (mode ES)")
    ap.add_argument("--ad-stats", default=None, help="json from pool_ad_stats.py (AD-mean/-shuffle/-random)")
    ap.add_argument("--no-sp-deviation", action="store_true", help="skip ||a - a_SP|| (saves one pass)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--log-envs", type=int, default=32)
    ap.add_argument("--latency-iters", type=int, default=500)
    ap.add_argument("--dump-obs", action="store_true",
                    help="save real observations (first mode, every 25 steps, <= 8 robots) to "
                         "obs_<terrain>.pt for latency_onboard.py")
    ap.add_argument("--traj-envs", type=int, default=0,
                    help="log the trajectory (base position and velocity, command, N, action, joint angles) of the "
                         "first K robots at every control step, per mode and terrain, to traj_<tag>_<mode>_<terrain>.pt")
    ap.add_argument("--curve-checkpoints", nargs="+", default=[],
                    help="learning curves: also evaluate these checkpoints of the same run on the same simulation "
                         "(one env per terrain, weights swapped); rows are told apart by the checkpoint column")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-isolate", action="store_true", help=argparse.SUPPRESS)    # internal: child process
    ap.add_argument("--skip-latency", action="store_true", help=argparse.SUPPRESS)  # internal: not the first child
    args = ap.parse_args()
    if len(args.terrains) > 1 and not args.no_isolate:
        sys.exit(run_isolated(sys.argv[1:], args.terrains))

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.method}_seed{args.train_seed}"
    csv_path = os.path.join(args.out, f"episodes_{tag}.csv")
    ad = load_adapter(args.adapter, args.checkpoint, args.device)
    if not args.no_sp_deviation and getattr(ad, "actor", True) is None:   # PPO-MLP or a BC baseline
        print("[run_e0] checkpoint has no diffusion actor: ||a - a_SP|| not computed (as --no-sp-deviation)", flush=True)
        args.no_sp_deviation = True
    sp = None if args.no_sp_deviation else make_policy(ad, "SP", None, args.tau, 0)

    ad_stats = None
    if args.ad_stats:
        with open(args.ad_stats) as fh:
            js = json.load(fh)
        ad_stats = {"mean_n": js["mean_n"], "n_pool": torch.tensor(js["n_pool"], dtype=torch.long)}
    latency = {}
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    fields = FIELDS
    if not new_file:   # appending to a file written by an older run_e0: keep its columns, drop the new ones
        with open(csv_path, newline="") as fh:
            fields = next(csv.reader(fh), FIELDS)
        missing = [f for f in FIELDS if f not in fields]
        if missing:
            print(f"[run_e0] {csv_path} has the older header: not written for these rows: {', '.join(missing)}",
                  flush=True)
    with open(csv_path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new_file:
            wr.writeheader()
        if hasattr(ad, "eval_seconds"):
            ad.eval_seconds = args.episode_seconds   # lets the adapter size the terrain tiles
        for terrain in args.terrains:
            env = ad.make_env(terrain, args.episodes, args.eval_seed)
            for c, ck in enumerate([args.checkpoint] + list(args.curve_checkpoints)):
                if c == 0:
                    adc, spc = ad, sp
                else:   # same simulation, other weights: transplant the env state the adapter holds
                    adc = load_adapter(args.adapter, ck, args.device)
                    for attr in ("task", "dt", "eval_seconds"):
                        if hasattr(ad, attr):
                            setattr(adc, attr, getattr(ad, attr))
                    spc = None if sp is None else make_policy(adc, "SP", None, args.tau, 0)
                ck_tag = tag if c == 0 else f"{tag}_{os.path.splitext(os.path.basename(ck))[0]}"
                for j, token in enumerate(args.modes):
                    dump = [] if (args.dump_obs and j == 0 and c == 0) else None
                    policy = make_policy(adc, token, ad_stats, args.tau, seed=args.eval_seed)
                    log = [] if (c == 0 and (token.startswith("AD") or token.startswith("ORIG"))) else None
                    traj = [] if (args.traj_envs > 0 and c == 0) else None
                    t0 = time.time()
                    res = run_episodes(adc, env, policy, spc, args, log, dump, traj)
                    if dump:
                        torch.save(torch.cat(dump), os.path.join(args.out, f"obs_{terrain}.pt"))
                    if traj:
                        torch.save(_stack_traj(traj, adc.dt), os.path.join(args.out, f"traj_{tag}_{token}_{terrain}.pt"))
                    for i in range(len(res["success"])):
                        wr.writerow({"method": args.method, "train_seed": args.train_seed,
                                     "checkpoint": os.path.basename(ck), "mode": token,
                                     "terrain": terrain, "episode": i,
                                     **{k: float(v[i]) for k, v in res.items()}})
                    fh.flush()
                    if log:
                        torch.save(log, os.path.join(args.out, f"steplog_{tag}_{token}_{terrain}.pt"))
                    print(f"[{ck_tag}] {terrain:15s} {token:10s} success={res['success'].float().mean():.3f} "
                          f"mean_n={res['mean_n'].mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
                    if (c == 0 and token not in latency and not token.endswith(RETEST)
                            and not args.skip_latency):  # batch-1 latency on this device
                        obs1 = ad.reset(env, args.eval_seed)[:1]
                        latency[token] = time_policy(lambda o: policy(o), obs1, n_iter=args.latency_iters)
            ad.close(env)

    lat_path = os.path.join(args.out, f"latency_{tag}.json")
    old = json.load(open(lat_path)) if os.path.exists(lat_path) else {"latency": {}}
    old["latency"].update(latency)
    old["device"] = args.device
    with open(lat_path, "w") as fh:
        json.dump(old, fh, indent=2)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
