"""BiRoDiff-R and DiffuseLoco-R: behaviour-cloned diffusion baselines.

Torch only (no Isaac Gym, no rl_games). Both originals are offline imitation methods, so we retrain them on our own
demonstrations (collect_demos.py) with the same observation, robot and 50 Hz control rate.

BiRoDiff-R  (Mothish, Tayal, Kolathaya, arXiv:2407.05424, Sec. IV and Table II)
  - latent observation network M: MLP with 3 hidden layers of 48 units, GELU, latent size 48
  - denoiser eps_theta: MLP with 7 hidden layers of 256 units, GELU; input = [a^k, t_emb, O^l]
  - 128-D sinusoidal timestep embedding; epsilon prediction; MSE loss
  - linear beta schedule, K = 60 DDPM steps (training and sampling)
  - Adam, lr 1e-4, batch 4000; one action per control step (no chunking)
  Not reported in the paper, chosen here: beta end points 1e-4 and 0.02 rescaled by 1000/K (so the
  60-step chain ends close to N(0, I)); z-score normalisation of observations and actions; the
  number of gradient steps matches the original budget (10,000 epochs x 500,500 / 4,000).

DiffuseLoco-R  (Huang et al., CoRL 2024, arXiv:2404.19264; released config
  diffusion_policy/config_files/cyber_diffusion_policy_n=8.yaml)
  - transformer denoiser in the style of Diffusion Policy's TransformerForDiffusion:
    6 decoder layers, 8 heads, 256-D tokens, attention dropout 0.3, causal self-attention over
    the predicted action tokens, cross-attention to [timestep token, history tokens]
  - history h = 8 single-step frames (proprioception, command, previous action), each embedded
    by a 2-layer MLP; prediction horizon n = 4; only the first action is executed and the policy
    re-plans at every control step (receding horizon, paper App. B-A)
  - DDPM, squaredcos_cap_v2 schedule, K = 10 steps for training and sampling, epsilon prediction,
    clip_sample, fixed_small variance; min-max normalisation to [-1, 1]
  - AdamW (betas 0.9/0.95), lr 1e-4, weight decay 1e-3, batch 4096, cosine lr with warm-up,
    EMA (power 0.75, max 0.9999), 100 epochs (paper App. B-C)
  Deviations, all in the baseline's favour or neutral: no one-step input delay (the simulator does
  not model latency for any controller); the command is part of each frame (one frame encoder)
  instead of a separate goal encoder.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "bc_diffusion"

ARCH_DEFAULTS = {
    "birodiff": dict(K=60, schedule="linear_rescaled", clip_sample=False, norm="zscore",
                     latent_dim=48, latent_hidden=48, latent_layers=3,
                     den_hidden=256, den_layers=7, t_emb=128,
                     optimizer="adam", lr=1e-4, weight_decay=0.0, batch=4000,
                     steps=int(10000 * math.ceil(500500 / 4000)), warmup=0, lr_schedule="constant",
                     ema=False, val_frac=0.2),
    "diffuseloco": dict(K=10, schedule="squaredcos_cap_v2", clip_sample=True, norm="limits",
                        h=8, n_pred=4, n_emb=256, n_head=8, n_layer=6, p_drop_attn=0.3,
                        optimizer="adamw", lr=1e-4, weight_decay=1e-3, betas=(0.9, 0.95), batch=4096,
                        epochs=100, warmup=10000, lr_schedule="cosine",
                        ema=True, ema_power=0.75, ema_max=0.9999, val_frac=0.02),
}


# ----------------------------------------------------------------------------
# DDPM schedule and sampler (diffusers DDPMScheduler conventions, timesteps 0..K-1)
# ----------------------------------------------------------------------------
def make_betas(K: int, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, K, dtype=torch.float64)
    if schedule == "linear_rescaled":   # Nichol & Dhariwal (2021): same end points per 1000 steps
        s = 1000.0 / K
        return torch.linspace(s * 1e-4, s * 0.02, K, dtype=torch.float64)
    if schedule == "squaredcos_cap_v2":  # diffusers betas_for_alpha_bar(cosine)
        f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
        return torch.tensor([min(1 - f((i + 1) / K) / f(i / K), 0.999) for i in range(K)], dtype=torch.float64)
    raise ValueError(schedule)


class DDPM:
    def __init__(self, K: int, schedule: str, clip_sample: bool):
        self.K, self.clip = K, clip_sample
        betas = make_betas(K, schedule)
        self.alphas = (1.0 - betas).float()
        self.alpha_bar = torch.cumprod(1.0 - betas, 0).float()
        self.betas = betas.float()

    def to(self, dev):
        for k in ("alphas", "alpha_bar", "betas"):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    def add_noise(self, x0, eps, t):
        ab = self.alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    @torch.no_grad()
    def sample(self, eps_fn, shape, device, generator=None):
        """Full K-step ancestral DDPM chain (variance 'fixed_small'). Returns x_0."""
        x = torch.randn(shape, device=device, generator=generator)
        for t in range(self.K - 1, -1, -1):
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = eps_fn(x, tt)
            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones((), device=device)
            beta_t = 1 - ab_t / ab_prev
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if self.clip:
                x0 = x0.clamp(-1, 1)
            mean = (ab_prev.sqrt() * beta_t / (1 - ab_t)) * x0 + ((1 - beta_t).sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
            if t > 0:
                var = ((1 - ab_prev) / (1 - ab_t) * beta_t).clamp(min=1e-20)
                x = mean + var.sqrt() * torch.randn(shape, device=device, generator=generator)
            else:
                x = mean
        return x


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


# ----------------------------------------------------------------------------
# Normalisers
# ----------------------------------------------------------------------------
class Normalizer(nn.Module):
    """zscore: (x - mean) / std ;  limits: map [min, max] -> [-1, 1] (Diffusion Policy 'limits')."""

    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.mode = mode
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("offset", torch.zeros(dim))

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.mode == "zscore":
            mean, std = x.mean(0), x.std(0)
            self.scale.copy_(1.0 / std.clamp(min=1e-4))
            self.offset.copy_(-mean * self.scale)
        else:
            lo, hi = x.min(0).values, x.max(0).values
            rng = (hi - lo)
            const = rng < 1e-4
            rng = torch.where(const, torch.full_like(rng, 2.0), rng)
            self.scale.copy_(2.0 / rng)
            self.offset.copy_(torch.where(const, -lo * self.scale, -1.0 - lo * self.scale))
        return self

    def forward(self, x):
        return x * self.scale + self.offset

    def inverse(self, y):
        return (y - self.offset) / self.scale


def mlp(sizes, act=nn.GELU, final_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_act:
            layers.append(act())
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------
# BiRoDiff denoiser
# ----------------------------------------------------------------------------
class BiRoDiffNet(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=48, latent_hidden=48, latent_layers=3,
                 den_hidden=256, den_layers=7, t_emb=128, **_):
        super().__init__()
        self.t_emb, self.act_dim = t_emb, act_dim
        self.latent = mlp([obs_dim] + [latent_hidden] * latent_layers + [latent_dim])
        self.den = mlp([act_dim + t_emb + latent_dim] + [den_hidden] * den_layers + [act_dim])

    def cond(self, obs_n):
        return self.latent(obs_n)

    def eps(self, x, t, cond):
        return self.den(torch.cat([x, sinusoidal(t, self.t_emb), cond], dim=-1))


# ----------------------------------------------------------------------------
# DiffuseLoco transformer denoiser (TransformerForDiffusion-style, obs as condition)
# ----------------------------------------------------------------------------
class DiffuseLocoNet(nn.Module):
    def __init__(self, frame_dim, act_dim, h=8, n_pred=4, n_emb=256, n_head=8, n_layer=6,
                 p_drop_attn=0.3, **_):
        super().__init__()
        self.h, self.n_pred, self.n_emb, self.act_dim = h, n_pred, n_emb, act_dim
        self.input_emb = nn.Linear(act_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_pred, n_emb))
        self.cond_obs_emb = nn.Sequential(nn.Linear(frame_dim, n_emb), nn.Mish(), nn.Linear(n_emb, n_emb))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + h, n_emb))
        self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb))
        layer = nn.TransformerDecoderLayer(d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                                           dropout=p_drop_attn, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layer)
        causal = torch.triu(torch.full((n_pred, n_pred), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, act_dim)
        self.apply(self._init)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def cond(self, frames_n):            # [B, h, F] -> [B, h, n_emb]
        return self.cond_obs_emb(frames_n)

    def eps(self, x, t, cond):           # x [B, n_pred, A]
        tok_t = sinusoidal(t, self.n_emb).unsqueeze(1)
        mem = self.encoder(torch.cat([tok_t, cond], dim=1) + self.cond_pos_emb)
        y = self.input_emb(x) + self.pos_emb
        y = self.decoder(tgt=y, memory=mem, tgt_mask=self.causal_mask)
        return self.head(self.ln_f(y))


# ----------------------------------------------------------------------------
# Model container: networks + normalisers + schedule
# ----------------------------------------------------------------------------
class BCDiffusion(nn.Module):
    def __init__(self, arch: str, obs_dim: int, act_dim: int, frame_dim: int, cfg: dict):
        super().__init__()
        self.arch, self.obs_dim, self.act_dim, self.frame_dim = arch, obs_dim, act_dim, frame_dim
        self.cfg = dict(cfg)
        if arch == "birodiff":
            self.net = BiRoDiffNet(obs_dim, act_dim, **cfg)
            self.obs_norm = Normalizer(obs_dim, cfg["norm"])
        elif arch == "diffuseloco":
            self.net = DiffuseLocoNet(frame_dim, act_dim, **cfg)
            self.obs_norm = Normalizer(frame_dim, cfg["norm"])
        else:
            raise ValueError(arch)
        self.act_norm = Normalizer(act_dim, cfg["norm"])
        self.ddpm = DDPM(cfg["K"], cfg["schedule"], cfg["clip_sample"])

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())

    def to(self, *a, **k):
        super().to(*a, **k)
        self.ddpm.to(self.obs_norm.scale.device)
        return self

    def loss(self, cond_in, actions):
        """cond_in: obs [B, D] (birodiff) or frames [B, h, F] (diffuseloco); actions [B, A] or [B, n, A]."""
        x0 = self.act_norm(actions)
        cond = self.net.cond(self.obs_norm(cond_in))
        t = torch.randint(0, self.ddpm.K, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        return F.mse_loss(self.net.eps(self.ddpm.add_noise(x0, eps, t), t, cond), eps)

    @torch.no_grad()
    def act(self, cond_in, generator=None):
        cond = self.net.cond(self.obs_norm(cond_in))
        B = cond.shape[0]
        shape = (B, self.act_dim) if self.arch == "birodiff" else (B, self.net.n_pred, self.act_dim)
        x0 = self.ddpm.sample(lambda x, t: self.net.eps(x, t, cond), shape, cond.device, generator)
        a = self.act_norm.inverse(x0)
        return a if self.arch == "birodiff" else a[:, 0]   # receding horizon: execute the first action

    # -------------------------------------------------------------- checkpoint
    def checkpoint(self, extra: Optional[dict] = None):
        sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.cfg.items()}
        return {"format": FORMAT, "arch": self.arch, "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                "frame_dim": self.frame_dim, "cfg": cfg, "state_dict": sd, **(extra or {})}

    @staticmethod
    def from_checkpoint(ck: dict, device="cpu") -> "BCDiffusion":
        if ck.get("format") != FORMAT:
            raise ValueError("not a bc_diffusion checkpoint")
        m = BCDiffusion(ck["arch"], ck["obs_dim"], ck["act_dim"], ck["frame_dim"], ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        return m.to(device).eval()


def safe_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:   # torch < 1.13
        return torch.load(path, map_location=device)


# ----------------------------------------------------------------------------
# Closed-loop policy (stateful for DiffuseLoco's history)
# ----------------------------------------------------------------------------
class BCPolicy:
    """callable(obs [B, D]) -> (actions [B, A], nfe [B], None), the run_e0 'custom' interface.

    obs is the environment's stacked observation, newest frame first (legged_terrain.py,
    get_latest_n): the newest single frame is obs[:, :frame_dim]. DiffuseLoco-R keeps its own
    h-frame history; at an episode start it is filled with the first frame (Diffusion Policy pads
    with the first observation). Call reset_all() at a seeded reset, mark_reset(done) after a step.
    """

    def __init__(self, model: BCDiffusion, seed: int = 0):
        self.m, self.seed = model, seed
        self.hist = None
        self.pending = None
        self.gen = None

    def reset_all(self):
        self.hist = None
        self.pending = None

    def mark_reset(self, done: torch.Tensor):
        if self.hist is not None and done.shape[0] == self.hist.shape[0]:
            self.pending = done.bool().to(self.hist.device) if self.pending is None else (self.pending | done.bool().to(self.hist.device))

    @torch.no_grad()
    def __call__(self, obs):
        B, dev = obs.shape[0], obs.device
        if self.gen is None or self.gen.device != dev:
            self.gen = torch.Generator(device=dev).manual_seed(self.seed)
        if self.m.arch == "birodiff":
            a = self.m.act(obs, self.gen)
        else:
            f = obs[:, : self.m.frame_dim]
            h = self.m.net.h
            if self.hist is None or self.hist.shape[0] != B:
                self.hist = f.unsqueeze(1).repeat(1, h, 1)
                self.pending = None
            else:
                self.hist = torch.cat([self.hist[:, 1:], f.unsqueeze(1)], dim=1)   # oldest ... newest
                if self.pending is not None and self.pending.any():
                    self.hist[self.pending] = f[self.pending].unsqueeze(1).repeat(1, h, 1)
                self.pending = None
            a = self.m.act(self.hist, self.gen)
        return a, torch.full((B,), self.m.ddpm.K, dtype=torch.long, device=dev), None


# ----------------------------------------------------------------------------
# Demonstration dataset (from collect_demos.py) and training
# ----------------------------------------------------------------------------
class DemoWindows:
    """Index (env, step) pairs of a demos file; builds BiRoDiff / DiffuseLoco training samples.

    demos: obs [E, S, D] (fp16), act [E, S, A], done [E, S] (True = last step of an episode),
    valid [E, S] (the fixed number of pairs per terrain kind), frame_dim.
    """

    def __init__(self, demos: dict, device, env_ids: torch.Tensor):
        self.dev = device
        E, S, _ = demos["obs"].shape
        self.F = int(demos["frame_dim"])
        self.obs = demos["obs"][env_ids].to(device)
        self.act = demos["act"][env_ids].float().to(device)
        done = demos["done"][env_ids].bool()
        valid = demos["valid"][env_ids].bool()
        e_n, S = done.shape
        # episode start / end for every (e, s)
        start = torch.zeros(e_n, S, dtype=torch.long)
        end = torch.full((e_n, S), S - 1, dtype=torch.long)
        cur = torch.zeros(e_n, dtype=torch.long)
        for s in range(S):
            if s > 0:
                cur = torch.where(done[:, s - 1], torch.full_like(cur, s), cur)
            start[:, s] = cur
        cur = torch.full((e_n,), S - 1, dtype=torch.long)
        for s in range(S - 1, -1, -1):
            cur = torch.where(done[:, s], torch.full_like(cur, s), cur)
            end[:, s] = cur
        self.start, self.end = start.to(device), end.to(device)
        idx = valid.nonzero(as_tuple=False)
        self.idx = idx.to(device)     # [N, 2] (env, step)
        self.S = S

    def __len__(self):
        return self.idx.shape[0]

    def frames_all(self):
        return self.obs[..., : self.F]

    def batch(self, sel: torch.Tensor, arch: str, h: int = 8, n_pred: int = 4):
        e, s = self.idx[sel, 0], self.idx[sel, 1]
        if arch == "birodiff":
            return self.obs[e, s].float(), self.act[e, s]
        off_h = torch.arange(-h + 1, 1, device=self.dev)
        sh = torch.maximum(s[:, None] + off_h[None], self.start[e, s][:, None])
        frames = self.obs[e[:, None], sh, : self.F].float()               # [B, h, F] oldest..newest
        off_a = torch.arange(0, n_pred, device=self.dev)
        sa = torch.minimum(s[:, None] + off_a[None], self.end[e, s][:, None])
        acts = self.act[e[:, None], sa]                                     # [B, n, A]
        return frames, acts


class EMA:
    """Diffusion Policy's EMAModel: decay = 1 - (1 + step / inv_gamma) ** -power, clipped."""

    def __init__(self, model: nn.Module, power=0.75, max_value=0.9999, inv_gamma=1.0):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.power, self.max_value, self.inv_gamma, self.step = power, max_value, inv_gamma, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = 1 - (1 + self.step / self.inv_gamma) ** -self.power
        d = min(max(d, 0.0), self.max_value)
        for ps, p in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(d).add_(p.detach(), alpha=1 - d)
        for bs, b in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(b)
