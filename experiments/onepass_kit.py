"""OnePassDiff kit: simulator-independent pieces.

Contents
  - cosine schedule and x0-prediction DDIM sampler (fixed, per-sample and early-stopping)
  - M1  advantage-weighted denoising loss
  - M2  refinement-need target + step-predictor loss
  - M3  joint-limit loss on predicted targets
  - InferencePolicy: the E0 inference modes
  - checkpoint diagnostics (untrained noisy-action weights)

Interfaces expected from your code (see my_adapter.py):
  denoiser(z_s, z_t, x_t, t) -> x0_hat   z_s [B,128], z_t [B,16], x_t [B,A], t LongTensor [B] in {0..T}
  tnp_logits(z_t) -> [B] or [B,1]        pre-sigmoid output of the Terrain Noise Predictor
  encode(obs) -> (z_s, z_t)
"""
from __future__ import annotations

import math
from collections import deque
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

Denoiser = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ----------------------------------------------------------------------------
# Schedule and sampler
# ----------------------------------------------------------------------------
def cosine_alpha_bar(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cumulative retention factors of the cosine schedule (Nichol & Dhariwal 2021).

    Returns a tensor of length T+1 with alpha_bar[0] = 1 (clean) and alpha_bar[t] for t = 1..T.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=max_beta)
    ab = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, 0)])
    return ab.float()


def ddim_timesteps(T: int, n_steps: int) -> torch.Tensor:
    """Descending, uniformly spaced timesteps in [1, T]; the final update goes to t = 0."""
    ts = torch.linspace(T, 1, n_steps).round().long()
    return torch.unique_consecutive(ts)


def _ddim_update(x_t, x0, ab_t, ab_next, eta_s, noise):
    eps = (x_t - ab_t.sqrt() * x0) / (1 - ab_t).clamp(min=1e-12).sqrt()
    sigma = eta_s * ((1 - ab_next) / (1 - ab_t).clamp(min=1e-12)).sqrt() * (1 - ab_t / ab_next).clamp(min=0).sqrt()
    dir_coef = (1 - ab_next - sigma ** 2).clamp(min=0).sqrt()
    return ab_next.sqrt() * x0 + dir_coef * eps + sigma * noise


@torch.no_grad()
def ddim_sample(denoiser: Denoiser, z_s, z_t, n_steps: int, alpha_bar: torch.Tensor, act_dim: int,
                eta_s: float = 0.0, x_init: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None, return_x0_traj: bool = False):
    """x0-prediction DDIM with a fixed number of steps for the whole batch.

    Returns the final action [B, A], and optionally the list of x0 estimates (one per step).
    """
    B, dev = z_s.shape[0], z_s.device
    ab = alpha_bar.to(dev)
    T = ab.numel() - 1
    x = torch.randn(B, act_dim, device=dev, generator=generator) if x_init is None else x_init.clone()
    ts = ddim_timesteps(T, n_steps).tolist()
    traj = []
    for i, t in enumerate(ts):
        t_next = ts[i + 1] if i + 1 < len(ts) else 0
        tt = torch.full((B,), t, dtype=torch.long, device=dev)
        x0 = denoiser(z_s, z_t, x, tt)
        traj.append(x0)
        if t_next == 0:
            x = x0
        else:
            noise = torch.randn(B, act_dim, device=dev, generator=generator) if eta_s > 0 else torch.zeros_like(x)
            x = _ddim_update(x, x0, ab[t], ab[t_next], eta_s, noise)
    return (x, traj) if return_x0_traj else x


@torch.no_grad()
def ddim_sample_per_sample(denoiser: Denoiser, z_s, z_t, n_per_sample: torch.Tensor, alpha_bar, act_dim,
                           eta_s: float = 0.0, x_init=None, generator=None):
    """DDIM where each sample has its own step count (adaptive modes). Groups samples by N."""
    B, dev = z_s.shape[0], z_s.device
    if x_init is None:
        x_init = torch.randn(B, act_dim, device=dev, generator=generator)
    out = torch.empty(B, act_dim, device=dev)
    for n in torch.unique(n_per_sample).tolist():
        idx = (n_per_sample == n).nonzero(as_tuple=True)[0]
        out[idx] = ddim_sample(denoiser, z_s[idx], z_t[idx], int(n), alpha_bar, act_dim,
                               eta_s=eta_s, x_init=x_init[idx], generator=generator)
    return out


@torch.no_grad()
def ddim_early_stop(denoiser: Denoiser, z_s, z_t, alpha_bar, act_dim, n_min: int, n_max: int, tau: float,
                    x_init=None, generator=None):
    """Training-free baseline: run the n_max schedule, stop per sample after >= n_min steps once
    ||x0^(k) - x0^(k-1)|| < tau; return the current clean-action estimate and the steps used."""
    B, dev = z_s.shape[0], z_s.device
    ab = alpha_bar.to(dev)
    T = ab.numel() - 1
    x = torch.randn(B, act_dim, device=dev, generator=generator) if x_init is None else x_init.clone()
    ts = ddim_timesteps(T, n_max).tolist()
    out = torch.empty(B, act_dim, device=dev)
    used = torch.full((B,), len(ts), dtype=torch.long, device=dev)
    active = torch.arange(B, device=dev)
    prev_x0 = None
    for k, t in enumerate(ts, start=1):
        t_next = ts[k] if k < len(ts) else 0
        tt = torch.full((active.numel(),), t, dtype=torch.long, device=dev)
        x0 = denoiser(z_s[active], z_t[active], x[active], tt)
        done = torch.zeros(active.numel(), dtype=torch.bool, device=dev)
        if prev_x0 is not None and k >= n_min:
            done = (x0 - prev_x0).norm(dim=-1) < tau
        if t_next == 0:
            done[:] = True
        if done.any():
            out[active[done]] = x0[done]
            used[active[done]] = k
        keep = ~done
        if not keep.any():
            break
        x_new = _ddim_update(x[active], x0, ab[t], ab[t_next], 0.0, torch.zeros_like(x0))
        x[active] = x_new
        active, prev_x0 = active[keep], x0[keep]
    return out, used


# ----------------------------------------------------------------------------
# M1: advantage-weighted denoising loss
# ----------------------------------------------------------------------------
def denoising_loss(denoiser: Denoiser, z_s, z_t, actions, advantages, alpha_bar,
                   beta: float = 1.0, w_max: float = 20.0, target: str = "executed",
                   mean_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Eq. (dn) of the manuscript. Add lambda_dn * this to the PPO minibatch loss.

    actions     executed actions a_i from the rollout buffer        [B, A]
    advantages  normalised GAE advantages (as used by PPO)          [B] or [B,1]
    target      "executed"      -> AWR weights exp(A/beta), clipped at w_max (main method)
                "self_distill"  -> target sg(mean_actions), unit weights (ablation)
    Pass detached z_s, z_t for the "no encoder gradient" ablation.
    Timestep 0 is excluded, so PPO stays the only objective on the single-pass output.
    """
    dev = actions.device
    ab_all = alpha_bar.to(dev)
    T = ab_all.numel() - 1
    B = actions.shape[0]
    if target == "executed":
        a0 = actions.detach()
        with torch.no_grad():
            w = torch.exp(advantages.reshape(B) / beta).clamp(max=w_max)
    elif target == "self_distill":
        assert mean_actions is not None, "self_distill needs mean_actions"
        a0 = mean_actions.detach()
        w = torch.ones(B, device=dev)
    else:
        raise ValueError(target)
    t = torch.randint(1, T + 1, (B,), device=dev)
    ab = ab_all[t].unsqueeze(-1)
    x_t = ab.sqrt() * a0 + (1 - ab).sqrt() * torch.randn_like(a0)
    x0 = denoiser(z_s, z_t, x_t, t)
    per_sample = (x0 - a0).pow(2).sum(-1)
    return (w * per_sample).sum() / w.sum()


# ----------------------------------------------------------------------------
# M2: step predictor
# ----------------------------------------------------------------------------
def eta_to_steps(eta: torch.Tensor, n_min: int = 5, n_max: int = 10) -> torch.Tensor:
    """N = N_min + floor(eta * (N_max - N_min + 1)), as in the paper."""
    k = torch.floor(eta * (n_max - n_min + 1)).long().clamp(0, n_max - n_min)
    return n_min + k


@torch.no_grad()
def refinement_need(denoiser: Denoiser, z_s, z_t, alpha_bar, act_dim, n_min=5, n_max=10, generator=None):
    """r(O) = ||D(O; N_max, x_N) - D(O; N_min, x_N)||, deterministic DDIM, shared x_N."""
    x_init = torch.randn(z_s.shape[0], act_dim, device=z_s.device, generator=generator)
    a_lo = ddim_sample(denoiser, z_s, z_t, n_min, alpha_bar, act_dim, 0.0, x_init)
    a_hi = ddim_sample(denoiser, z_s, z_t, n_max, alpha_bar, act_dim, 0.0, x_init)
    return (a_hi - a_lo).norm(dim=-1)


class RunningQuantile:
    """Running q-quantile of recent values (kappa in the paper)."""

    def __init__(self, q: float = 0.95, maxlen: int = 50):
        self.q, self.buf = q, deque(maxlen=maxlen)

    def update(self, values: torch.Tensor) -> float:
        self.buf.append(values.detach().float().cpu())
        return float(torch.quantile(torch.cat(list(self.buf)), self.q).clamp(min=1e-8))


def tnp_loss(tnp_logits_fn: Callable, z_t, r: torch.Tensor, kappa: float) -> torch.Tensor:
    """BCE between eta = sigmoid(g(sg(z_t))) and y = min(r / kappa, 1). No gradient reaches the encoder."""
    y = (r / kappa).clamp(max=1.0)
    logits = tnp_logits_fn(z_t.detach()).reshape(-1)
    return F.binary_cross_entropy_with_logits(logits, y)


def privileged_tnp_loss(tnp_logits_fn: Callable, z_t, difficulty01: torch.Tensor) -> torch.Tensor:
    """Ablation: supervise eta with a simulator terrain-difficulty label in [0, 1]."""
    logits = tnp_logits_fn(z_t.detach()).reshape(-1)
    return F.binary_cross_entropy_with_logits(logits, difficulty01.clamp(0, 1))


# ----------------------------------------------------------------------------
# M3: joint-limit loss on predicted targets
# ----------------------------------------------------------------------------
def joint_limit_loss(mean_actions, q_default, action_scale, q_lower, q_upper) -> torch.Tensor:
    """Penalise predicted joint-position targets q_hat = q0 + k_a * a_hat outside [q_lower, q_upper].

    Differentiable w.r.t. the actor, unlike a penalty on measured joint positions from the buffer.
    Returns (loss, fraction of samples where the penalty is active).
    """
    q_hat = q_default + action_scale * mean_actions
    viol = F.relu(q_hat - q_upper).pow(2) + F.relu(q_lower - q_hat).pow(2)
    per = viol.sum(-1)
    return per.mean(), (per > 0).float().mean().detach()


# ----------------------------------------------------------------------------
# E0 inference modes
# ----------------------------------------------------------------------------
MODES = ("SP", "RP", "D", "AD", "AD-mean", "AD-shuffle", "AD-random", "ES", "custom")


class InferencePolicy:
    """Wraps encode/denoiser/tnp into one of the E0 inference modes.

    mode        SP | RP | D | AD | AD-mean | AD-shuffle | AD-random | ES | custom
                RP = the rollout policy PPO optimized: the single-pass mean plus Gaussian noise with the
                learned, executed std (action_std); the control for "iterative sampling reproduces the
                rollout distribution"
    n_fixed     N for mode D
    mean_n      mean N measured for AD on the same suite (AD-mean, AD-random)
    n_pool      1-D tensor of N values pooled over all terrains from an AD run (AD-shuffle)
    x_init_zero use x_N = 0 instead of x_N ~ N(0, I)
    action_std  for mode RP: [A] tensor, the std PPO executed in training (Adapter.action_std())
    custom_fn   for mode "custom": callable(obs) -> (actions, n, eta); use it to run your ORIGINAL
                deployment code unchanged, so the preliminary numbers can be reproduced.
    Returns (actions [B,A], n_steps [B] long, eta [B] or None).
    """

    def __init__(self, encode, denoiser, alpha_bar, act_dim, mode, tnp_logits=None, n_fixed=None,
                 n_min=5, n_max=10, eta_s=0.0, x_init_zero=False, mean_n=None, n_pool=None,
                 tau=None, custom_fn=None, seed=0, action_std=None):
        assert mode in MODES, mode
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.gen = None
        self._seed = seed

    def _generator(self, dev):
        if self.gen is None or self.gen.device != torch.device(dev):
            self.gen = torch.Generator(device=dev).manual_seed(self._seed)
        return self.gen

    @torch.no_grad()
    def __call__(self, obs):
        if self.mode == "custom":
            return self.custom_fn(obs)
        z_s, z_t = self.encode(obs)
        B, dev = z_s.shape[0], z_s.device
        A = self.act_dim
        if self.mode in ("SP", "RP"):
            t0 = torch.zeros(B, dtype=torch.long, device=dev)
            a = self.denoiser(z_s, z_t, torch.zeros(B, A, device=dev), t0)
            if self.mode == "RP":
                std = self.action_std.to(dev).reshape(1, A)
                a = a + std * torch.randn(B, A, device=dev, generator=self._generator(dev))
            return a, torch.ones(B, dtype=torch.long, device=dev), None
        g = self._generator(dev)
        x_init = torch.zeros(B, A, device=dev) if self.x_init_zero else torch.randn(B, A, device=dev, generator=g)
        eta = None
        if self.mode == "ES":
            a, n = ddim_early_stop(self.denoiser, z_s, z_t, self.alpha_bar, A, self.n_min, self.n_max,
                                   self.tau, x_init=x_init)
            return a, n, None
        if self.mode == "D":
            n = torch.full((B,), int(self.n_fixed), dtype=torch.long, device=dev)
        elif self.mode == "AD":
            eta = torch.sigmoid(self.tnp_logits(z_t).reshape(-1))
            n = eta_to_steps(eta, self.n_min, self.n_max)
        elif self.mode == "AD-mean":
            n = torch.full((B,), int(round(self.mean_n)), dtype=torch.long, device=dev)
        elif self.mode == "AD-random":
            lo = math.floor(self.mean_n)
            p_hi = self.mean_n - lo
            n = lo + (torch.rand(B, device=dev, generator=g) < p_hi).long()
        elif self.mode == "AD-shuffle":
            pool = self.n_pool.to(dev)
            n = pool[torch.randint(0, pool.numel(), (B,), device=dev, generator=g)]
        a = ddim_sample_per_sample(self.denoiser, z_s, z_t, n, self.alpha_bar, A, eta_s=self.eta_s,
                                   x_init=x_init, generator=g)
        return a, n, eta


# ----------------------------------------------------------------------------
# Checkpoint diagnostics (untrained noisy-action weights)
# ----------------------------------------------------------------------------
def zero_grad_columns(first_linear: nn.Linear, loss: torch.Tensor, x_cols: slice) -> dict:
    """Gradient of `loss` w.r.t. the first linear layer, split into noisy-action columns and the rest.
    Call with a loss computed through the single-pass path (x = 0, t = 0)."""
    (g,) = torch.autograd.grad(loss, first_linear.weight, retain_graph=True)
    col = g.norm(dim=0)
    mask = torch.zeros_like(col, dtype=torch.bool)
    mask[x_cols] = True
    return {"x_cols_grad_max": float(col[mask].max()), "other_cols_grad_median": float(col[~mask].median())}


def init_likeness(first_linear: nn.Linear, x_cols: slice) -> dict:
    """Heuristic check of whether noisy-action weights are still at PyTorch's default init.

    nn.Linear default init is U(-b, b) with b = 1/sqrt(fan_in). Untrained columns should have
    max|W|/b <= 1 and std close to b/sqrt(3); trained columns usually drift away from that.
    """
    W = first_linear.weight.detach()
    b = 1.0 / math.sqrt(W.shape[1])
    mask = torch.zeros(W.shape[1], dtype=torch.bool)
    mask[x_cols] = True
    Wx, Wo = W[:, mask], W[:, ~mask]
    return {
        "bound_b": b,
        "x_cols_max_over_b": float(Wx.abs().max() / b),
        "x_cols_std_over_uniform_std": float(Wx.std() / (b / math.sqrt(3))),
        "other_cols_max_over_b": float(Wo.abs().max() / b),
        "other_cols_std_over_uniform_std": float(Wo.std() / (b / math.sqrt(3))),
    }


@torch.no_grad()
def time_policy(policy: Callable, obs_one: torch.Tensor, n_iter: int = 500, warmup: int = 50) -> dict:
    """Wall-clock latency per control step for batch size 1 (use on the robot's computer too)."""
    import time
    dev = obs_one.device
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        policy(obs_one)
    sync()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        policy(obs_one)
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    times = torch.tensor(times)
    return {"mean_ms": float(times.mean()), "p50_ms": float(times.median()),
            "p95_ms": float(torch.quantile(times, 0.95))}
