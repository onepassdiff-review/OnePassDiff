"""
Diffusion Policy Module for Legged Robot Locomotion

Architecture matches BiRoDiff (arxiv 2407.05424) adapted for PPO training:

┌─────────────────────────────────────────────────────────────────┐
│ Observations (N-dim state history)                              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Encoder: obs → 256 → GELU → LN → 128 → GELU → LN             │
│    → state_head(128→128)    → terrain_head(128→16)              │
│  State Latent (128D)        Terrain Latent (16D)                │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Denoiser  (paper: 7-layer MLP hidden=256, GELU)               │
│  entry(144+action+128t → 256) → GELU → LN                      │
│  → ResBlock(256,t=128) × 3                                      │
│  → Linear(256 → action_dim)                                     │
│  DDIM inference: 5-10 adaptive steps                            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Output: Joint Angles (action_dim) — direct x0 prediction       │
└─────────────────────────────────────────────────────────────────┘

"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embeddings  (128D, learnable GELU projection — paper style)
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """
    128-dim sinusoidal timestep embeddings + learnable 2-layer GELU projection.
    BiRoDiff uses 128D time embeddings.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device   = time.device
        half_dim = self.dim // 2
        scale    = math.log(10000) / (half_dim - 1)
        freqs    = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args     = time[:, None].float() * freqs[None, :]
        emb      = torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Residual block with GELU  (used inside AdaptiveDiffusionNet)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Pre-norm residual block with additive time conditioning and GELU activation.
    Structure:  LN → Linear → GELU  (+t_emb projection)  → LN → Linear → GELU(x + h)
    """
    def __init__(self, dim: int, time_dim: int, residual: bool = True):
        super().__init__()
        self.residual = residual   # False only for the "w/o ResBlocks" ablation (same parameters, no skip)
        self.norm1   = nn.LayerNorm(dim)
        self.fc1     = nn.Linear(dim, dim)
        self.time_fc = nn.Linear(time_dim, dim)
        self.norm2   = nn.LayerNorm(dim)
        self.fc2     = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(self.norm1(x)))
        h = h + self.time_fc(t_emb)
        h = self.fc2(self.norm2(h))
        return F.gelu(x + h) if self.residual else F.gelu(h)


# ---------------------------------------------------------------------------
# Unified Encoder  (GELU, wider state_latent)
# ---------------------------------------------------------------------------

class UnifiedEncoder(nn.Module):
    """
    obs_dim → Linear(256) → GELU → LN(256) → Linear(128) → GELU → LN(128)
           → state_head  Linear(128 → state_latent_dim=128)   [identity-like, wide]
           → terrain_head Linear(128 → terrain_latent_dim=16)  [compression]

    Paper encoder (BiRoDiff): 3-layer MLP, hidden=48, GELU → 48D latent.
    We use a wider version to match our larger obs_dim and action_dim.
    """
    def __init__(self, obs_dim: int,
                 state_latent_dim: int = 128,
                 terrain_latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.state_head   = nn.Linear(128, state_latent_dim)
        self.terrain_head = nn.Linear(128, terrain_latent_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(obs)
        return self.state_head(enc), self.terrain_head(enc)


# ---------------------------------------------------------------------------
# Terrain Noise Predictor  (used in sample() for adaptive DDIM steps)
# ---------------------------------------------------------------------------

class TerrainNoisePredictor(nn.Module):
    """
    Predicts terrain-adaptive noise scale σ ∈ (0, 1) from terrain latent.
    Output through Sigmoid ensures stable [0,1] range for DDIM step scheduling.
    Only used in sample() — no direct PPO gradient, but benefits from terrain
    latent gradients that flow through the shared encoder.
    """
    def __init__(self, terrain_latent_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(terrain_latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, terrain_latent: torch.Tensor) -> torch.Tensor:
        return self.network(terrain_latent)   # [B, 1]  ∈ (0, 1)


# ---------------------------------------------------------------------------
# Adaptive Diffusion Net  (paper-style: 7 layers ≈ 3 ResBlocks, hidden=256)
# ---------------------------------------------------------------------------

class AdaptiveDiffusionNet(nn.Module):
    """
    3-ResBlock denoiser with 256 hidden dim and 128D time conditioning.

    Matches BiRoDiff paper denoiser:
      Paper: 7-layer MLP, hidden=256, GELU, input=[a_t + t_emb(128) + O^l(48)]
      Ours:  entry(256) + 3×ResBlock(256,t=128) + exit(256→action_dim)
             input = [state_latent(128) + terrain_latent(16) + noisy_actions(action_dim)]

    input_dim = state_latent + terrain_latent + noise_dim
    → Linear(256) → GELU → LN(256)
    → ResBlock(256, time_dim=128)  ×3
    → Linear(256 → output_dim)     ← direct x0 prediction
    """
    def __init__(self, state_dim: int = 128, terrain_dim: int = 16,
                 noise_dim: int = 10, time_dim: int = 128, output_dim: int = 10,
                 residual: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.time_mlp   = SinusoidalPositionEmbeddings(time_dim)

        input_dim = state_dim + terrain_dim + noise_dim
        self.entry = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.res1 = ResidualBlock(256, time_dim, residual)
        self.res2 = ResidualBlock(256, time_dim, residual)
        self.res3 = ResidualBlock(256, time_dim, residual)
        # Direct x0 prediction — no sigmoid gate (v3 fix)
        self.exit = nn.Linear(256, output_dim)

    def forward(self, state_latent: torch.Tensor, terrain_latent: torch.Tensor,
                noisy_actions: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timesteps.float())                          # [B, 128]
        x     = torch.cat([state_latent, terrain_latent, noisy_actions], dim=-1)
        x     = self.entry(x)
        x     = self.res1(x, t_emb)
        x     = self.res2(x, t_emb)
        x     = self.res3(x, t_emb)
        return self.exit(x)                                               # [B, action_dim]


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """
    Terrain-Adaptive Diffusion Policy for Legged Robot Locomotion.

    PPO training path  (forward):
      obs → encoder → [state_latent, terrain_latent]
          → denoiser(state, terrain, zeros, t=0) → action_mean
      Always t=0 with zero noise: clean single-step prediction.
      No train/eval mismatch.

    DDIM inference path  (sample):
      Runs num_ddim_steps of correct x0-prediction DDIM.
      Stochasticity η comes from TerrainNoisePredictor ∈ (0,1).

      ᾱ_t  = cos²(((t/T)+s)/(1+s)·π/2)
      ε̂    = (x_t − √ᾱ_t·x̂_0) / √(1−ᾱ_t)
      σ²_t = η²·(1−ᾱ_{t-1})/(1−ᾱ_t)·(1−ᾱ_t/ᾱ_{t-1})
      x_{t-1} = √ᾱ_{t-1}·x̂_0 + √(1−ᾱ_{t-1}−σ²_t)·ε̂ + σ_t·z
    """
    def __init__(self,
                 obs_dim: int,
                 action_dim: int = 10,
                 state_latent_dim: int = 128,
                 terrain_latent_dim: int = 16,
                 num_ddim_steps: int = 10,
                 min_ddim_steps: int = 5,
                 training_noise_scale: float = 0.0,
                 residual: bool = True,
                 zero_terrain_latent: bool = False):
        super().__init__()
        self.obs_dim              = obs_dim
        self.action_dim           = action_dim
        self.num_ddim_steps       = num_ddim_steps
        self.min_ddim_steps       = min_ddim_steps
        # training_noise_scale kept for API compat but defaults to 0 (cleaner PPO signal)
        self.training_noise_scale = training_noise_scale
        # ablation "w/o terrain latent": z_terrain is replaced by zeros everywhere (same parameters)
        self.zero_terrain_latent  = zero_terrain_latent

        self.unified_encoder = UnifiedEncoder(
            obs_dim=obs_dim,
            state_latent_dim=state_latent_dim,
            terrain_latent_dim=terrain_latent_dim,
        )
        self.noise_predictor = TerrainNoisePredictor(
            terrain_latent_dim=terrain_latent_dim,
        )
        self.diffusion_net = AdaptiveDiffusionNet(
            state_dim=state_latent_dim,
            terrain_dim=terrain_latent_dim,
            noise_dim=action_dim,
            time_dim=128,                 # paper: 128D time embedding
            output_dim=action_dim,
            residual=residual,
        )
        # Learnable log_std; init std=1.0 (exp(0)) — matches baseline exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    # ------------------------------------------------------------------
    # Pieces used by the training losses M1-M3 and the evaluation modes.
    # forward() and sample() below are unchanged and give identical outputs.
    # ------------------------------------------------------------------
    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs (already normalised) -> (z_state [B,128], z_terrain [B,16])."""
        z_s, z_t = self.unified_encoder(obs)
        if self.zero_terrain_latent:
            z_t = torch.zeros_like(z_t)
        return z_s, z_t

    def act_mean(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """The single-pass PPO action mean: f(z, x=0, t=0)."""
        B = z_s.shape[0]
        zero_noise = torch.zeros(B, self.action_dim, device=z_s.device, dtype=z_s.dtype)
        timesteps  = torch.zeros(B, dtype=torch.long, device=z_s.device)
        return self.diffusion_net(z_s, z_t, zero_noise, timesteps)

    def tnp_logits(self, z_t: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid output of the Terrain Noise Predictor, shape [B] (sigmoid of this = eta)."""
        return self.noise_predictor.network[:-1](z_t).squeeze(-1)

    # ------------------------------------------------------------------
    # forward()  — called by PPO during rollouts and optimization
    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [B, obs_dim]
        Returns:
            action_mean:  [B, action_dim]
            action_std:   [B, action_dim]
            noise_level:  [B, 1]  ∈ (0, 1)
        """
        batch_size = obs.shape[0]
        device     = obs.device

        state_latent, terrain_latent = self.encode(obs)
        noise_level = self.noise_predictor(terrain_latent)   # for sample() use

        # Clean single-step prediction at t=0 with zero noise.
        # No train/eval difference — avoids sinusoidal embedding bias.
        zero_noise = torch.zeros(batch_size, self.action_dim, device=device)
        timesteps  = torch.zeros(batch_size, dtype=torch.long, device=device)

        action_mean = self.diffusion_net(state_latent, terrain_latent, zero_noise, timesteps)
        action_std  = torch.exp(self.log_std).expand(batch_size, -1)
        return action_mean, action_std, noise_level

    # ------------------------------------------------------------------
    # sample()  — DDIM inference  (deployment / eval only)
    # ------------------------------------------------------------------
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample actions with correct x0-prediction DDIM denoising.

        Args:
            obs:           [B, obs_dim]
            deterministic: skip diffusion loop, return single-step prediction.
        Returns:
            actions: [B, action_dim]
        """
        batch_size = obs.shape[0]
        device     = obs.device

        with torch.no_grad():
            state_latent, terrain_latent = self.encode(obs)
            noise_level = self.noise_predictor(terrain_latent)   # [B,1] ∈ (0,1)

            if deterministic:
                zero_noise = torch.zeros(batch_size, self.action_dim, device=device)
                timesteps  = torch.zeros(batch_size, dtype=torch.long, device=device)
                return self.diffusion_net(state_latent, terrain_latent, zero_noise, timesteps)

            eta       = float(noise_level.mean())               # ∈ (0, 1)
            num_steps = int(self.min_ddim_steps
                            + eta * (self.num_ddim_steps - self.min_ddim_steps))
            num_steps = max(self.min_ddim_steps, min(num_steps, self.num_ddim_steps))

            x_t = torch.randn(batch_size, self.action_dim, device=device)

            for t in reversed(range(num_steps)):
                ts      = torch.full((batch_size,), t, device=device, dtype=torch.long)
                x0_pred = self.diffusion_net(state_latent, terrain_latent, x_t, ts)

                if t == 0:
                    x_t = x0_pred
                    break

                ab_t      = self._alpha_bar(t,     num_steps)
                ab_t_prev = self._alpha_bar(t - 1, num_steps)

                sqrt_ab_t      = math.sqrt(ab_t)
                sqrt_1m_ab_t   = math.sqrt(max(1.0 - ab_t, 1e-8))
                sqrt_ab_t_prev = math.sqrt(ab_t_prev)

                # Noise direction (DDIM eq. 12)
                eps = (x_t - sqrt_ab_t * x0_pred) / sqrt_1m_ab_t

                # Stochastic term
                ratio    = (1.0 - ab_t_prev) / (1.0 - ab_t)
                rel_ab   = 1.0 - ab_t / ab_t_prev
                sigma_sq = max(eta ** 2 * ratio * rel_ab, 0.0)
                dir_sq   = max(1.0 - ab_t_prev - sigma_sq, 0.0)

                x_t = (sqrt_ab_t_prev * x0_pred
                       + math.sqrt(dir_sq) * eps
                       + math.sqrt(sigma_sq) * torch.randn_like(x_t))

        return x_t

    @staticmethod
    def _alpha_bar(t: int, num_steps: int) -> float:
        """Cosine noise schedule: ᾱ_t = cos²(((t/T)+s)/(1+s)·π/2), s=0.008."""
        s = 0.008
        return math.cos(((t / num_steps) + s) / (1.0 + s) * math.pi / 2.0) ** 2


# ---------------------------------------------------------------------------
# rl_games NetworkBuilder wrapper
# ---------------------------------------------------------------------------

class DiffusionActorCriticBuilder:
    """rl_games NetworkBuilder-compatible builder wrapping DiffusionPolicy."""

    def __init__(self):
        pass

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return DiffusionActorCriticBuilder.Network(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__()
            actions_num     = kwargs.pop('actions_num')
            input_shape     = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs   = kwargs.pop('num_seqs', 1)

            obs_dim  = input_shape[0]
            diff_cfg = params.get('diffusion', {})

            self.actor = DiffusionPolicy(
                obs_dim              = obs_dim,
                action_dim           = actions_num,
                state_latent_dim     = diff_cfg.get('state_latent_dim',     128),
                terrain_latent_dim   = diff_cfg.get('terrain_latent_dim',    16),
                num_ddim_steps       = diff_cfg.get('num_ddim_steps',        10),
                min_ddim_steps       = diff_cfg.get('min_ddim_steps',         5),
                training_noise_scale = diff_cfg.get('training_noise_scale', 0.0),
                residual             = not diff_cfg.get('no_resblocks', False),
                zero_terrain_latent  = diff_cfg.get('zero_terrain_latent', False),
            )

            # rl_games' continuous_a2c_logstd model applies exp() to the 2nd output of this
            # network, i.e. it expects LOG-std. Returning exp(log_std) would make the executed std
            # exp(exp(log_std)) >= 1. logstd_fix: True returns log_std (set in BipedPPOOnePass.yaml).
            # Deterministic evaluation (mu) is unaffected.
            self.logstd_fix = bool(diff_cfg.get('logstd_fix', False))

            # Diffusion schedule used by the denoising loss (M1) and by every DDIM evaluation mode.
            # Fixed T, independent of the number of sampling steps N (the original sample()
            # rebuilt the schedule for each N, so timestep index t meant a different noise level
            # for different N). Non-persistent buffer: not saved, checkpoints keep their keys.
            self.diffusion_T = int(diff_cfg.get('T', 100))
            self.schedule    = diff_cfg.get('schedule', 'cosine')
            self.register_buffer('alpha_bar', make_alpha_bar(self.diffusion_T, self.schedule),
                                 persistent=False)

            # Extra step-predictor heads for the predictor ablations (same architecture as the
            # TNP; they never feed the actor). 'priv' is trained on a simulator difficulty label,
            # 'untrained' is never updated (reproduces the preliminary, untrained predictor).
            if diff_cfg.get('tnp_extra_heads', False):
                tdim = diff_cfg.get('terrain_latent_dim', 16)
                self.tnp_heads = nn.ModuleDict({
                    'priv':      TerrainNoisePredictor(tdim),
                    'untrained': TerrainNoisePredictor(tdim),
                })
                for p in self.tnp_heads['untrained'].parameters():
                    p.requires_grad_(False)

            # Critic MLP — identical width to baseline for fair value estimation
            units  = params['mlp']['units']
            layers = []
            in_sz  = obs_dim
            for u in units:
                layers += [nn.Linear(in_sz, u), nn.ELU()]
                in_sz = u
            layers.append(nn.Linear(in_sz, self.value_size))
            self.critic = nn.Sequential(*layers)

            # auxiliary-loss state (set by envs/common/onepass_agent.py; inactive otherwise)
            self.aux_cfg   = None
            self.aux_ctx   = None
            self._aux      = None
            self.aux_stats = {}

        # -------------------------------------------------------------- forward
        def forward(self, obs_dict):
            obs      = obs_dict['obs']
            z_s, z_t = self.actor.encode(obs)
            mu       = self.actor.act_mean(z_s, z_t)
            log_std  = self.actor.log_std.expand(mu.shape[0], -1)
            second   = log_std if self.logstd_fix else torch.exp(log_std)
            value    = self.critic(obs)
            if obs_dict.get('is_train', False) and self.aux_ctx is not None and self.aux_cfg is not None:
                self._aux = self._compute_aux(z_s, z_t, mu, obs_dict['prev_actions'],
                                              self.aux_ctx['advantages'])
            return mu, second, value, None

        def is_separate_critic(self):    return False
        def is_rnn(self):                return False
        def get_default_rnn_state(self): return None

        def get_aux_loss(self):
            """Called by rl_games' calc_gradients right after forward(); every value is ADDED to the loss."""
            aux, self._aux = self._aux, None
            return aux

        # ------------------------------------------------------ training losses
        def configure_aux(self, cfg, q_default=None, action_scale=None, q_lower=None, q_upper=None):
            """cfg: the onepass block of the train config (see onepass_agent.DEFAULTS)."""
            self.aux_cfg = dict(cfg)
            if q_default is not None:
                dev = self.actor.log_std.device
                self.q_default    = torch.as_tensor(q_default, dtype=torch.float, device=dev)
                self.action_scale = torch.as_tensor(action_scale, dtype=torch.float, device=dev)
                self.q_lower      = torch.as_tensor(q_lower, dtype=torch.float, device=dev)
                self.q_upper      = torch.as_tensor(q_upper, dtype=torch.float, device=dev)
            elif cfg.get('enable_joint', False):
                raise ValueError('enable_joint needs q_default, action_scale, q_lower, q_upper')

        def _compute_aux(self, z_s, z_t, mu, actions, advantages):
            from experiments import onepass_kit as kit   # repo-root/experiments
            c, out = self.aux_cfg, {}
            if c.get('enable_dn', False):
                zs, zt = (z_s.detach(), z_t.detach()) if c.get('dn_detach_encoder', False) else (z_s, z_t)
                l_dn = kit.denoising_loss(self.actor.diffusion_net, zs, zt, actions.float(), advantages,
                                          self.alpha_bar, beta=c['dn_beta'], w_max=c['dn_wmax'],
                                          target=c['dn_target'], mean_actions=mu)
                out['loss_dn'] = c['lambda_dn'] * l_dn
                with torch.no_grad():
                    w = torch.exp(advantages.reshape(-1).float() / c['dn_beta'])
                    self._stat('loss_dn', l_dn)
                    self._stat('awr_w_mean', w.clamp(max=c['dn_wmax']).mean())
                    self._stat('awr_w_clipped_frac', (w > c['dn_wmax']).float().mean())
            if c.get('enable_joint', False):
                l_j, active = kit.joint_limit_loss(mu.float(), self.q_default, self.action_scale,
                                                   self.q_lower, self.q_upper)
                out['loss_joint'] = c['lambda_joint'] * l_j
                self._stat('loss_joint', l_j)
                self._stat('joint_active_frac', active)
            return out or None

        def _stat(self, k, v):
            self.aux_stats.setdefault(k, []).append(float(v.detach()))

        def pop_aux_stats(self):
            s = {k: sum(v) / len(v) for k, v in self.aux_stats.items() if v}
            self.aux_stats = {}
            return s

        def tnp_head_logits(self, name, z_t):
            """name: 'selfsup' (the TNP itself), 'priv' or 'untrained'. Pre-sigmoid, shape [B]."""
            if name == 'selfsup':
                return self.actor.tnp_logits(z_t)
            return self.tnp_heads[name].network[:-1](z_t).squeeze(-1)


def make_alpha_bar(T: int, schedule: str = 'cosine') -> torch.Tensor:
    """alpha_bar[0..T], alpha_bar[0] = 1 (clean). Cosine: Nichol & Dhariwal (s = 0.008).
    Linear: DDPM beta_t from 1e-4 to 0.02 rescaled to T steps (the linear-schedule ablation)."""
    if schedule == 'cosine':
        steps = torch.arange(T + 1, dtype=torch.float64)
        s = 0.008
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        # identical to experiments/onepass_kit.cosine_alpha_bar (checked in tests)
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        ab = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, 0)])
    elif schedule == 'linear':
        betas = torch.linspace(1e-4 * 1000 / T, 0.02 * 1000 / T, T, dtype=torch.float64).clamp(max=0.999)
        ab = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, 0)])
    else:
        raise ValueError(f'unknown schedule {schedule}')
    return ab.float()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_diffusion_policy(obs_dim: int, action_dim: int = 10, **kwargs) -> DiffusionPolicy:
    return DiffusionPolicy(obs_dim=obs_dim, action_dim=action_dim, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("DIFFUSION POLICY v3  —  SMOKE TEST")
    print("=" * 60)

    obs_dim    = 210   # typical stacked obs (42 per frame × 5)
    action_dim = 10
    batch_size = 4

    policy = create_diffusion_policy(obs_dim=obs_dim, action_dim=action_dim)
    total  = sum(p.numel() for p in policy.parameters())
    enc_p  = sum(p.numel() for p in policy.unified_encoder.parameters())
    net_p  = sum(p.numel() for p in policy.diffusion_net.parameters())
    pred_p = sum(p.numel() for p in policy.noise_predictor.parameters())
    print(f"\nTotal parameters:      {total:,}")
    print(f"  UnifiedEncoder:      {enc_p:,}")
    print(f"  AdaptiveDiffusionNet:{net_p:,}")
    print(f"  TerrainNoisePredictor:{pred_p:,}")

    obs = torch.randn(batch_size, obs_dim)

    policy.train()
    mu, sigma, nlevel = policy(obs)
    print(f"\nTraining forward:")
    print(f"  action_mean : {mu.shape},  range [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  action_std  : {sigma[0].tolist()}")
    print(f"  noise_level : {nlevel.squeeze().tolist()}")

    policy.eval()
    mu_e, _, _ = policy(obs)
    print(f"\nEval forward: {mu_e.shape}")

    det   = policy.sample(obs, deterministic=True)
    stoch = policy.sample(obs, deterministic=False)
    print(f"Deterministic sample: {det.shape}")
    print(f"Stochastic DDIM:      {stoch.shape}")

    print("\n✓ All tests passed!")
