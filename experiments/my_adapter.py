"""Adapter for the OnePassDiff codebase (Isaac Gym Preview 4 + IsaacGymEnvs + rl_games).

Works for
  - OnePassDiff checkpoints (train=BipedPPOOnePass, envs/exp_onepass.sh), and
  - plain diffusion-actor checkpoints (train=BipedPPODiffusion);
  - PPO-MLP checkpoints (network name actor_critic): only mode ORIG (and pass --no-sp-deviation);
  - BiRoDiff-R / DiffuseLoco-R checkpoints (bc.pt from train_bc.py): only mode ORIG (and pass
    --no-sp-deviation). ORIG then runs the baseline's own sampler; mean_n is its NFE per step.

Layout: this file lives in experiments/, and experiments/ sits in the repository root
(OnePassDiff/experiments). Otherwise set ONEPASS_REPO=/path/to/OnePassDiff.

The training run's own Hydra config (.hydra/config.yaml in the run directory, found by walking up
from the checkpoint) is used to rebuild the network and the environment, so robot, observations
and control match training exactly. Override with ONEPASS_HYDRA_CFG=/path/to/config.yaml.

Evaluation environment (the frozen suite, manuscript Table "evalsuite"):
  one terrain kind per process, all robots on one square tile of that kind at a fixed difficulty,
  tile large enough to keep walking on it for the whole episode; fixed forward command
  (ONEPASS_EVAL_VX, default 0.5 m/s); no pushes, no domain randomisation, no observation noise,
  no command resampling; terrain generated from the eval seed; initial states seeded per mode.

Other options (environment variables)
  ONEPASS_TNP_HEAD   selfsup (default) | priv | untrained   step-predictor head used by AD/AD-*
  ONEPASS_T          override the diffusion T (default: from the run config, else 100)
"""
import glob
import math
import os
import re
import sys

try:
    import isaacgym  # noqa: F401  (must be imported before torch)
except ImportError:
    isaacgym = None
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ONEPASS_REPO", os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from onepass_kit import cosine_alpha_bar  # noqa: E402
from bc_baselines import FORMAT as BC_FORMAT, BCDiffusion, BCPolicy, safe_load  # noqa: E402

KIND = {"rough_up": 0, "rough_down": 1, "rough_flat": 2, "stair_up": 3, "stair_down": 4,
        "smooth_up": 5, "smooth_down": 6, "discrete": 7, "stepping_stone": 8}
TAN = lambda deg: math.tan(math.radians(deg))  # noqa: E731

# Frozen evaluation suite. Training curriculum maxima (envs/exp_onepass.sh): slope 0.4 (21.8 deg),
# stair height 0.15 m, rough +-0.15 m, discrete 0.15 m; stepping stones never trained on.
# Note: terrain.py quantises slopes to the 1 mm / 0.1 m grid (10 deg -> 9.6 deg, 18 deg -> 17.7 deg).
EVAL_TERRAINS = {
    "flat":           dict(terrainType="plane"),
    "rough":          dict(kind="rough_flat", uniform_height=0.05),
    "slope10":        dict(kind="smooth_up", slope=TAN(10)),
    "slope18":        dict(kind="smooth_up", slope=TAN(18)),
    "stair_up":       dict(kind="stair_up", stair_height=0.10),
    "stair_down":     dict(kind="stair_down", stair_height=0.10),
    "discrete":       dict(kind="discrete", discrete_height=0.10),
    "heldout_slope":  dict(kind="smooth_up", slope=TAN(25)),
    "heldout_stairs": dict(kind="stair_up", stair_height=0.18),
    "heldout_gaps":   dict(kind="stepping_stone"),   # 0.5 m stones, 0.1 m gaps at difficulty 1
}
# Difficulty sweeps (maximum traversable slope / stair rise), not part of the frozen suite:
#   sweep_slope_<deg>  smooth uphill slope of <deg> degrees      e.g. sweep_slope_26
#   sweep_stair_<mm>   stair ascent with a rise of <mm> millimetres (tread 0.31 m)   e.g. sweep_stair_160
SWEEP_SLOPES_DEG = (6, 10, 14, 18, 22, 26, 30)
SWEEP_STAIRS_MM = (40, 70, 100, 130, 160, 190, 220)


def terrain_spec(name):
    """Suite terrain or difficulty-sweep terrain -> terrain spec (raises ValueError for anything else)."""
    if name in EVAL_TERRAINS:
        return EVAL_TERRAINS[name]
    m = re.fullmatch(r"sweep_slope_(\d+)", name)
    if m and 0 < int(m.group(1)) < 45:
        return dict(kind="smooth_up", slope=TAN(int(m.group(1))))
    m = re.fullmatch(r"sweep_stair_(\d+)", name)
    if m and 0 < int(m.group(1)) <= 300:
        return dict(kind="stair_up", stair_height=int(m.group(1)) / 1000.0)
    raise ValueError(f"terrain {name} is neither in the suite {sorted(EVAL_TERRAINS)} nor a sweep terrain "
                     "(sweep_slope_<deg>, sweep_stair_<mm>); the transition course is not implemented for terrain.py")


def _find_hydra_cfg(checkpoint):
    env = os.environ.get("ONEPASS_HYDRA_CFG")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(checkpoint))
    for _ in range(6):
        c = os.path.join(d, ".hydra", "config.yaml")
        if os.path.exists(c):
            return c
        d = os.path.dirname(d)
    raise FileNotFoundError(f"no .hydra/config.yaml above {checkpoint}; set ONEPASS_HYDRA_CFG")


def _plain(x):
    """numpy scalars (from the repo's arange/linspace resolvers) -> Python scalars, recursively."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x.item() if hasattr(x, "item") and type(x).__module__ == "numpy" else x


def _register_resolvers():
    import numpy as np
    from omegaconf import OmegaConf
    import isaacgymenvs  # noqa: F401  registers eq/contains/if/resolve_default
    for name, fn in {
        "arange": lambda a, b, c: list(np.round(np.arange(a, b, c), 15)),
        "linspace": lambda a, b, n: list(np.round(np.linspace(a, b, n), 15)),
        "eq": lambda x, y: x.lower() == y.lower(),
        "contains": lambda x, y: x.lower() in y.lower(),
        "if": lambda pred, a, b: a if pred else b,
        "resolve_default": lambda default, arg: default if arg == "" else arg,
    }.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


class Adapter:
    act_dim = 10
    dt = 0.02   # replaced by the task's rl_dt in make_env

    def __init__(self, checkpoint, device="cuda:0"):
        from omegaconf import OmegaConf
        self.device = device
        self.eval_seconds = 60.0
        self.eval_vx = float(os.environ.get("ONEPASS_EVAL_VX", 0.5))
        self.tnp_head = os.environ.get("ONEPASS_TNP_HEAD", "selfsup")
        if isaacgym is not None:
            _register_resolvers()
        self.cfg_path = _find_hydra_cfg(checkpoint)
        self.cfg = OmegaConf.load(self.cfg_path)
        net_params = OmegaConf.to_container(self.cfg.train.params.network, resolve=True)
        self.net_name = net_params["name"]

        self.task = None
        self._stateful = None
        ck = safe_load(checkpoint, device)
        if isinstance(ck, dict) and ck.get("format") == BC_FORMAT:
            self.net_name = "bc_diffusion"
            self.bc = BCDiffusion.from_checkpoint(ck, device)
            self.net, self.actor = None, None
            self._ab = cosine_alpha_bar(100)
            print(f"[adapter] {ck['arch']} baseline, {self.bc.n_params():,} parameters, "
                  f"K = {self.bc.ddpm.K} denoising steps")
            return
        sd = ck["model"] if "model" in ck else ck
        self.obs_mean = sd["running_mean_std.running_mean"].float().to(device)
        self.obs_var = sd["running_mean_std.running_var"].float().to(device)
        net_sd = {k[len("a2c_network."):]: v for k, v in sd.items() if k.startswith("a2c_network.")}
        obs_dim = self.obs_mean.numel()

        if self.net_name == "diffusion_actor_critic":
            from envs.common.diffusion_policy import DiffusionActorCriticBuilder
            if "ONEPASS_T" in os.environ:
                net_params.setdefault("diffusion", {})["T"] = int(os.environ["ONEPASS_T"])
            self.net = DiffusionActorCriticBuilder.Network(
                net_params, actions_num=self.act_dim, input_shape=(obs_dim,), value_size=1)
            missing, unexpected = self.net.load_state_dict(net_sd, strict=False)
            bad = [k for k in missing if not k.startswith("tnp_heads.")]
            if bad or unexpected:
                raise RuntimeError(f"checkpoint/network mismatch: missing {bad}, unexpected {unexpected}")
            if missing:
                print("[adapter] checkpoint has no extra step-predictor heads (plain diffusion actor): "
                      "only ONEPASS_TNP_HEAD=selfsup is meaningful")
            self.actor = self.net.actor
            self._ab = self.net.alpha_bar.detach().cpu()
        elif self.net_name == "actor_critic":
            from rl_games.algos_torch.network_builder import A2CBuilder
            b = A2CBuilder()
            b.load(net_params)
            self.net = b.build("a2c", actions_num=self.act_dim, input_shape=(obs_dim,), value_size=1,
                               num_seqs=1)
            self.net.load_state_dict(net_sd, strict=True)
            self.actor = None
            self._ab = cosine_alpha_bar(100)
        else:
            raise ValueError(f"unsupported network {self.net_name}")
        self.net.to(device).eval()

    # ------------------------------------------------------------------ networks
    def _norm(self, obs):
        return ((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)).clamp(-5.0, 5.0)

    @property
    def alpha_bar(self):
        return self._ab

    def _need_diffusion(self):
        if self.actor is None:
            raise RuntimeError("this checkpoint has no diffusion actor: use mode ORIG and --no-sp-deviation")

    def encode(self, obs):
        self._need_diffusion()
        return self.actor.encode(self._norm(obs))

    def denoiser(self, z_s, z_t, x_t, t):
        return self.actor.diffusion_net(z_s, z_t, x_t, t)

    def tnp_logits(self, z_t):
        self._need_diffusion()
        return self.net.tnp_head_logits(self.tnp_head, z_t)

    @torch.no_grad()
    def action_std(self):
        """[A] std of the Gaussian rollout policy as EXECUTED in training (mode RP). rl_games applies exp() to the
        network's 2nd output; with logstd_fix that output is log_std (std = exp(log_std)), without it the output was
        exp(log_std) (std = exp(exp(log_std)), the old bug). clip_actions is False in LeggedTerrainPPO.yaml, and the
        task's own clipActions clamp acts on every evaluated action alike."""
        self._need_diffusion()
        log_std = self.actor.log_std.detach().float()
        return torch.exp(log_std) if self.net.logstd_fix else torch.exp(torch.exp(log_std))

    @torch.no_grad()
    def original_policy(self, obs):
        """ORIG for diffusion checkpoints = the original DiffusionPolicy.sample(), unchanged:
        batch-mean eta, N = int(5 + eta*5) for every robot, a schedule rebuilt for each N, and eta also
        used as the DDIM stochasticity.
        ORIG for MLP checkpoints = the deterministic action mean."""
        B = obs.shape[0]
        if self.net_name == "bc_diffusion":
            if self._stateful is None:
                self._stateful = BCPolicy(self.bc)
            return self._stateful(obs)
        o = self._norm(obs)
        if self.actor is None:
            mu = self.net({"obs": o, "is_train": False})[0]
            return mu, torch.ones(B, dtype=torch.long, device=obs.device), None
        a = self.actor.sample(o, deterministic=False)
        _, z_t = self.actor.encode(o)
        eta = self.actor.noise_predictor(z_t).reshape(-1)
        n = int(self.actor.min_ddim_steps + float(eta.mean()) * (self.actor.num_ddim_steps - self.actor.min_ddim_steps))
        n = max(self.actor.min_ddim_steps, min(n, self.actor.num_ddim_steps))
        return a, torch.full((B,), n, dtype=torch.long, device=obs.device), eta

    def new_original_policy(self, seed=0):
        """A fresh ORIG policy for one evaluation pass (run_e0 calls this per mode and terrain).
        The BC baselines are stateful (DiffuseLoco-R keeps an observation history)."""
        if self.net_name == "bc_diffusion":
            self._stateful = BCPolicy(self.bc, seed=seed)
            return self._stateful
        return self.original_policy

    def first_linear_and_x_cols(self):
        self._need_diffusion()
        enc = self.actor.unified_encoder
        if hasattr(enc, "d_s"):   # encoder variant without separate heads
            s, t = enc.d_s, enc.d_t
        else:
            s, t = enc.state_head.out_features, enc.terrain_head.out_features
        return self.actor.diffusion_net.entry[0], slice(s + t, s + t + self.act_dim)

    # ----------------------------------------------------------------- simulator
    def _eval_task_cfg(self, terrain, num_envs, seed):
        from omegaconf import OmegaConf, open_dict
        spec = terrain_spec(terrain)
        cfg = OmegaConf.create(_plain(OmegaConf.to_container(self.cfg, resolve=True)))
        with open_dict(cfg):
            e = cfg.task.env
            e.numEnvs = num_envs
            e.learn.episodeLength_s = float(self.eval_seconds) + 5.0   # the harness stops first
            e.learn.addNoise = False
            for k, v in e.randomize.items():
                if isinstance(v, dict) or OmegaConf.is_dict(v):
                    if "enable" in v and k != "base_init_pos_xy":
                        v.enable = False
            e.randomCommandVelocityRanges.linear_x = [self.eval_vx, self.eval_vx]
            e.randomCommandVelocityRanges.linear_y = [0.0, 0.0]
            e.randomCommandVelocityRanges.yaw = [0.0, 0.0]
            e.commandZeroProbability = 0.0
            e.keep_still_at_zero_command = False
            if "dataPublisher" in e:
                e.dataPublisher.enable = False
            t = e.terrain
            if spec.get("terrainType") == "plane":
                t.terrainType = "plane"
            else:
                side = 2.0 * (self.eval_vx * float(self.eval_seconds) + 3.0)   # square tile: pyramids need it
                t.terrainType = "trimesh"
                t.curriculum = True          # deterministic layout (the level update is switched off in make_env)
                t.numLevels = 1              # difficulty = (0 + 1) / 1 = 1 -> parameters below are exact
                t.numTerrains = 1
                t.maxInitMapLevel = 0
                t.difficultySale = 1.0
                t.mapLength = side
                t.mapWidth = side
                t.terrainProportions = [1 if i == KIND[spec["kind"]] else 0 for i in range(9)]
                if "slope" in spec:
                    t.slope = float(spec["slope"])
                if "stair_height" in spec:
                    t.stair.height = float(spec["stair_height"])
                if "uniform_height" in spec:
                    t.uniform.height = float(spec["uniform_height"])
                if "discrete_height" in spec:
                    t.discrete.height = float(spec["discrete_height"])
                    t.discrete.num_rects = int(0.4 * side * side)   # same density as 40 per 10 m x 10 m
            cfg.seed = int(seed)
            cfg.headless = True
            cfg.capture_video = False
            cfg.force_render = False
            cfg.task.env.numEnvs = num_envs
        return cfg

    def _train_task_cfg(self, num_envs, seed, spread_levels=True):
        """The run's own training environment (terrain mix, curriculum, domain randomisation,
        observation noise, command distribution), for collecting demonstrations."""
        from omegaconf import OmegaConf, open_dict
        cfg = OmegaConf.create(_plain(OmegaConf.to_container(self.cfg, resolve=True)))
        with open_dict(cfg):
            e = cfg.task.env
            e.numEnvs = num_envs
            if "dataPublisher" in e:
                e.dataPublisher.enable = False
            if spread_levels and e.terrain.get("curriculum", False):
                e.terrain.maxInitMapLevel = int(e.terrain.numLevels) - 1   # start robots on every level
            cfg.seed = int(seed)
            cfg.headless = True
            cfg.capture_video = False
            cfg.force_render = False
        return cfg

    def make_train_env(self, num_envs, seed, spread_levels=True):
        return self._make(self._train_task_cfg(num_envs, seed, spread_levels), num_envs, seed)

    def make_env(self, terrain, num_envs, seed):
        env = self._make(self._eval_task_cfg(terrain, num_envs, seed), num_envs, seed)
        # The eval tile has one level, but reset_idx() still runs update_terrain_level(), which calls
        # torch.randint_like for every robot that walked far in the PREVIOUS pass. That would consume a varying amount of
        # the seeded RNG before the initial-position draw, so on terrain tiles (not on the plane) every mode after the
        # first would start from different, order-dependent initial positions. The layout is already
        # built, and `curriculum` is read only by update_terrain_level and its logging after __init__, so switch it off.
        self.task.curriculum = False
        self._install_termination_probe()
        return env

    def _install_termination_probe(self):
        """Wraps the task's check_termination (unchanged) to record, before the task resets terminated robots, why each
        robot terminated and whether a foot stumbled. reset_idx overwrites projected_gravity of the reset robots, so the
        cause cannot be read after step(). Cause: 0 none or timeout; 1 fall = base tilt beyond the task's limit
        (projected gravity z > -0.7, as in check_termination) or base contact; 2 collision = a leg link other than the
        feet (the task's knee_ids: every link except base and feet) in contact while the base was upright; 3 terminated,
        not classified. Stumble = the task's own (unused, weight 0) stumble term: horizontal foot force > 5 N while the
        vertical force is < 1 N."""
        task = self.task
        if getattr(task, "_probe_installed", False):
            return
        orig = task.check_termination

        def check_termination():
            orig()
            cf = task.contact_force
            tilt = task.projected_gravity[:, 2] > -0.7
            base = cf[:, task.base_id, :].pow(2).sum(-1) > 1.0
            if task.allow_knee_contacts:
                leg = torch.zeros_like(base)
            else:
                leg = (cf[:, task.knee_ids, :].pow(2).sum(-1) > 1.0).any(-1)
            term = task.reset_buf.bool() & ~task.timeout_buf.bool()
            c = torch.full((task.num_envs,), 3, dtype=torch.long, device=task.device)
            c = torch.where(leg, torch.full_like(c, 2), c)
            c = torch.where(tilt | base, torch.ones_like(c), c)
            task._probe_cause = torch.where(term, c, torch.zeros_like(c))
            fxy = cf[:, task.foot_ids, :2].norm(dim=-1)
            task._probe_stumble = ((fxy > 5.0) & (cf[:, task.foot_ids, 2].abs() < 1.0)).any(-1)

        task.check_termination = check_termination
        task._probe_cause = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
        task._probe_stumble = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        task._probe_installed = True

    def _make(self, cfg, num_envs, seed):
        import numpy as np
        import isaacgymenvs
        from isaacgymenvs.tasks import isaacgym_task_map
        from isaacgymenvs.utils.utils import set_seed
        from envs.tasks.legged_terrain import LeggedTerrain
        for k in ("Biped", "A1Terrain", "AnymalTerrain", "RobotDog"):
            isaacgym_task_map[k] = LeggedTerrain
        sim_dev = os.environ.get("ONEPASS_SIM_DEVICE")   # e.g. "cpu": GPU-free smoke tests only
        if sim_dev:
            from omegaconf import open_dict
            with open_dict(cfg):
                cfg.sim_device = cfg.rl_device = sim_dev
                cfg.pipeline = "gpu" if "cuda" in sim_dev else "cpu"
                cfg.graphics_device_id = -1
                cfg.task.sim.use_gpu_pipeline = "cuda" in sim_dev
                cfg.task.sim.physx.use_gpu = "cuda" in sim_dev
        set_seed(int(seed))          # terrain.py draws obstacles with np.random
        np.random.seed(int(seed))
        env = isaacgymenvs.make(cfg.seed, cfg.task_name, num_envs, cfg.sim_device, cfg.rl_device,
                                cfg.graphics_device_id, True, False, False, False, cfg)
        self.task = env if hasattr(env, "reset_idx") else env.env
        self.dt = float(self.task.rl_dt)
        return env

    def reset(self, env, seed):
        task = self.task
        torch.manual_seed(int(seed))
        ids = torch.arange(task.num_envs, device=task.device)
        task.reset_idx(ids)
        if self._stateful is not None:
            self._stateful.reset_all()
        # one settle step at the default joint targets (a = 0), identical for every mode
        obs_dict, _, _, _ = task.step(torch.zeros(task.num_envs, self.act_dim, device=task.device))
        return obs_dict["obs"].to(self.device)

    def step(self, env, actions):
        obs_dict, _, done, extras = self.task.step(actions)
        timeout = extras["time_outs"].bool()
        if self._stateful is not None:
            self._stateful.mark_reset(done.bool().to(self.device))
        return obs_dict["obs"].to(self.device), done.bool().to(self.device), timeout.to(self.device)

    def base_xy(self, env):
        return self.task.root_state[:, :2].to(self.device)

    def base_pos(self, env):
        return self.task.root_state[:, :3].to(self.device)

    def command_xy(self, env):
        return self.task.commands[:, :2].to(self.device)

    def heading_xy(self, env):
        # yaw/heading command is 0: the commanded direction of travel is +x in the world frame
        h = torch.zeros(self.task.num_envs, 2, device=self.device)
        h[:, 0] = 1.0
        return h

    def base_lin_vel_xy(self, env):
        return self.task.base_lin_vel[:, :2].to(self.device)   # body frame, like the command

    # ---- motion-quality metrics (run_e0: CoT, joint-limit violations, jerk)
    def _act(self, x):
        """Actuated joints only, as in onepass_agent.joint_info (the limits L_joint uses)."""
        x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        m = getattr(self.task, "actuated_dof_mask", None)
        if m is not None and x.shape[-1] != self.act_dim:
            x = x[..., m.to(self.device)]
        return x

    def joint_state(self, env):
        """(q, qd, tau) of the actuated joints, [B, act_dim]. tau is the actuation torque applied in the last
        physics substep of the control step (decimation 4), so power is sampled at the 50 Hz control rate."""
        return (self._act(self.task.dof_pos), self._act(self.task.dof_vel),
                self._act(self.task.dof_actuation_force))

    def joint_target(self, env):
        """Commanded PD position targets of the actuated joints after the last step, [B, act_dim]
        (q0 + k_a * action, after the task's action clipping and delay)."""
        return self._act(self.task.dof_pos_target)

    def total_weight(self, env):
        """m g of each robot [B] (sum of rigid-body masses, as created)."""
        return self.task.total_gravity.to(self.device).float()

    def joint_limits(self, env):
        """Soft joint limits of the actuated joints, the ones L_joint penalises on the targets."""
        return self._act(self.task.dof_soft_limit_lower), self._act(self.task.dof_soft_limit_upper)

    # locomotion outcomes (run_e0: term_cause, footsteps, stumble_pct)
    def term_cause(self, env):
        return self.task._probe_cause.to(self.device)

    def foot_contact(self, env):
        return self.task.foot_contact_filt.to(self.device)   # contact filtered over two steps, the task's threshold

    def foot_stumble(self, env):
        return self.task._probe_stumble.to(self.device)

    def close(self, env):
        """No destroy_sim: it can hang on the CPU pipeline, and a process never builds a second simulation anyway
        (run_e0 runs each terrain in its own process). The simulation is freed when the process exits."""
        return None

    # --------------------------------------------------------------- utilities
    @staticmethod
    def find_checkpoint(run_dir, epoch):
        """outputs/Biped/train/<name>/runs/<name>/nn/last_<name>_ep_<epoch>_rew_*.pth"""
        hits = sorted(glob.glob(os.path.join(run_dir, "**", f"*_ep_{epoch}_rew*.pth"), recursive=True))
        if not hits:
            raise FileNotFoundError(f"no checkpoint for epoch {epoch} under {run_dir}")
        # at max_epochs rl_games writes the periodic file (..._rew_<r>.pth) and the final one
        # (..._rew__<r>_.pth): same weights. Prefer the final one, deterministically.
        final = [h for h in hits if "_rew__" in os.path.basename(h)]
        return (final or hits)[0]
