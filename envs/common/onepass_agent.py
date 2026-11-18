"""rl_games PPO agent with the three additional losses of OnePassDiff.

  M1  advantage-weighted denoising loss  (trains the denoiser at t > 0 on executed actions)
  M2  self-supervised step-predictor loss (trains the Terrain Noise Predictor; own optimizer)
  M3  joint-limit loss on predicted joint targets q0 + k_a * a_hat (differentiable)

Everything is configured by the `onepass:` block of the train config
(envs/cfg/train/BipedPPOOnePass.yaml). The network side lives in
DiffusionActorCriticBuilder.Network (envs/common/diffusion_policy.py); the maths lives in
experiments/onepass_kit.py, which must be importable as `experiments.onepass_kit`
(put or symlink the experiments/ folder in the repository root).

Registered in envs/train.py as algo name `a2c_continuous_onepass`.
"""
import inspect

import numpy as np
import torch
from rl_games.algos_torch import a2c_continuous, torch_ext
from rl_games.common import common_losses

from experiments import onepass_kit as kit

# rl_games >= 1.6.5 / master add model.get_aux_loss() to the PPO loss inside calc_gradients. rl_games 1.6.1, which
# IsaacGymEnvs pins, does NOT: there the M1/M3 losses would be computed and logged but never
# reach the gradient. With such a version the agent uses its own copy of the 1.6.1
# calc_gradients below, which adds them.
def _parent_adds_aux():
    fn = getattr(a2c_continuous.A2CAgent, 'calc_gradients', None)
    try:
        return fn is not None and 'get_aux_loss' in inspect.getsource(fn)
    except (OSError, TypeError):
        return False


_PARENT_ADDS_AUX = _parent_adds_aux()

DEFAULTS = dict(
    # M1
    enable_dn=True, lambda_dn=0.1, dn_beta=1.0, dn_wmax=20.0,
    dn_target='executed',            # 'executed' (main) or 'self_distill' (ablation)
    dn_detach_encoder=False,         # ablation: no denoising gradient into the encoder
    # M3
    enable_joint=True, lambda_joint=0.1,
    # M2
    enable_tnp=True, tnp_lr=1e-3, tnp_every=1, tnp_batch=4096,
    tnp_steps=8,                     # Adam steps on the step predictor per PPO iteration (fresh minibatch each)
    tnp_kappa_q=0.95, tnp_kappa_len=50, n_min=5, n_max=10,
    tnp_priv=True,                   # also train the privileged-label head (needs tnp_extra_heads)
)

# Privileged difficulty label for the 'priv' head (ablation only, never seen by the actor):
# label = KIND_WEIGHT[kind] * (level + 1) / num_levels, kinds in terrain.py order.
from envs.common.terrain_kinds import KIND_NAMES, kind_of_envs  # noqa: E402
KIND_NAMES = list(KIND_NAMES)
KIND_WEIGHT = np.array([0.6, 0.6, 0.5, 1.0, 0.9, 0.4, 0.4, 0.8, 1.0], dtype=np.float32)


def _unwrap_task(vec_env):
    """RLGPUEnv -> the Isaac Gym task (LeggedTerrain)."""
    return vec_env if hasattr(vec_env, 'reset_idx') else getattr(vec_env, 'env', vec_env)


def joint_info(task, act_dim, device):
    """(q_default, action_scale, q_lower, q_upper) for the actuated joints, from the env/URDF."""
    mask = getattr(task, 'actuated_dof_mask', None)

    def act(x):
        x = torch.as_tensor(x, dtype=torch.float, device=device).reshape(-1)
        if x.numel() != act_dim and mask is not None:
            x = x[mask.to(device)]
        assert x.numel() == act_dim, f'joint vector has {x.numel()} entries, expected {act_dim}'
        return x

    q_def = task.default_dof_pos
    q_def = q_def.float().mean(0) if q_def.dim() == 2 else q_def   # mean over envs if randomised
    return act(q_def), float(task.action_scale), act(task.dof_soft_limit_lower), act(task.dof_soft_limit_upper)


def env_terrain_kinds(task):
    """Terrain kind index (0..8, terrain.py order) of every env, or None for a plane."""
    terrain = getattr(task, 'terrain', None)
    if terrain is None or not hasattr(terrain, 'proportions'):
        return None
    if not getattr(task, 'curriculum', False):
        return None   # without the curriculum layout, terrain.py picks a random kind per tile
    return kind_of_envs(terrain.proportions, terrain.env_cols, task.terrain_types)


class OnePassAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        user = dict(params['config'].get('onepass', None) or {})
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown onepass options: {sorted(unknown)}')
        self.at = {**DEFAULTS, **user}
        net = self.model.a2c_network
        if not hasattr(net, 'configure_aux'):
            raise TypeError('OnePassAgent needs network name: diffusion_actor_critic')
        if self.at['tnp_priv'] and not hasattr(net, 'tnp_heads'):
            print('[onepass] tnp_priv requested but network has no extra heads '
                  '(set params.network.diffusion.tnp_extra_heads: True); disabling tnp_priv')
            self.at['tnp_priv'] = False

        self._aux_added = 0
        self.task = _unwrap_task(self.vec_env) if self.vec_env is not None else None
        act_dim = net.actor.action_dim
        if self.at['enable_joint']:
            q_def, k_a, lo, hi = joint_info(self.task, act_dim, self.ppo_device)
            net.configure_aux(self.at, q_def, k_a, lo, hi)
            print(f'[onepass] joint loss: k_a={k_a}, q_default={q_def.tolist()}\n'
                  f'             lower={lo.tolist()}\n             upper={hi.tolist()}')
        else:
            net.configure_aux(self.at)

        params_tnp = list(net.actor.noise_predictor.parameters())
        if self.at['tnp_priv']:
            params_tnp += list(net.tnp_heads['priv'].parameters())
        self.tnp_opt = torch.optim.Adam(params_tnp, lr=self.at['tnp_lr'])
        self.kappa = kit.RunningQuantile(q=self.at['tnp_kappa_q'], maxlen=self.at['tnp_kappa_len'])
        print(f'[onepass] config: {self.at}  T={net.diffusion_T} schedule={net.schedule} '
              f'logstd_fix={net.logstd_fix}')
        if not net.logstd_fix:
            print('[onepass] WARNING: logstd_fix is False -> executed std = exp(exp(log_std)) >= 1')

    # ------------------------------------------------------------------ M1, M3
    def calc_gradients(self, input_dict):
        """Make the minibatch advantages visible to the network's forward(), and make sure the M1/M3 losses
        that forward() returns through get_aux_loss() are added to the PPO loss before backward()."""
        net = self.model.a2c_network
        net.aux_ctx = {'advantages': input_dict['advantages']}
        try:
            if _PARENT_ADDS_AUX:
                out = super().calc_gradients(input_dict)
                self._aux_added += int(bool(getattr(self, 'aux_loss_dict', None)))
                return out
            return self._calc_gradients_rlg161(input_dict)
        finally:
            net.aux_ctx = None

    def _calc_gradients_rlg161(self, input_dict):
        """rl_games 1.6.1 A2CAgent.calc_gradients, unchanged except for the block marked AUX."""
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            # ---- AUX: M1 / M3 losses computed by the network's forward() (absent in rl_games 1.6.1)
            aux = self.model.a2c_network.get_aux_loss()
            if aux:
                for v in aux.values():
                    loss = loss + v
                self._aux_added += 1
            # ---- end AUX

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(self, {
            'values': value_preds_batch,
            'returns': return_batch,
            'new_neglogp': action_log_probs,
            'old_neglogp': old_action_log_probs_batch,
            'masks': rnn_masks
        }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, kl_dist, self.last_lr, lr_mul, mu.detach(), sigma.detach(), b_loss)

    # ------------------------------------------------------------------ M2
    def train_epoch(self):
        self._aux_added = 0
        out = super().train_epoch()
        net = self.model.a2c_network
        if (self.at['enable_dn'] or self.at['enable_joint']) and self._aux_added == 0:
            raise RuntimeError('[onepass] M1/M3 are enabled but no auxiliary loss reached the gradient this '
                               'iteration (rl_games calc_gradients changed?)')
        stats = net.pop_aux_stats()
        stats['aux_minibatches'] = float(self._aux_added)   # minibatches whose loss included M1/M3
        if self.at['enable_tnp'] and self.epoch_num % self.at['tnp_every'] == 0:
            stats.update(self._update_tnp())
        if self.writer is not None:
            for k, v in stats.items():
                self.writer.add_scalar(f'onepass/{k}', v, self.frame)
        return out

    def _privileged_label(self, actor_idx):
        kinds = env_terrain_kinds(self.task)
        dev = actor_idx.device
        if kinds is None:
            return torch.zeros(actor_idx.shape[0], device=dev), None
        levels = self.task.terrain_levels.float().to(dev)
        lab = torch.as_tensor(KIND_WEIGHT[kinds], device=dev) * (levels + 1) / self.task.terrain.env_rows
        return lab.clamp(0, 1)[actor_idx], torch.as_tensor(kinds, device=dev)[actor_idx]

    def _update_tnp(self):
        """M2: `tnp_steps` Adam steps on the step predictor, each on a fresh minibatch of this iteration's
        observations. kappa (running quantile of r) is updated once per iteration, from the first minibatch,
        so its window stays `tnp_kappa_len` iterations whatever `tnp_steps` is. Stats are from the last minibatch."""
        net = self.model.a2c_network
        obs_all = self.dataset.values_dict['obs']          # raw obs, [actors*horizon, obs_dim]
        M = min(self.at['tnp_batch'], obs_all.shape[0])
        was_training = self.model.training
        self.model.eval()                                  # running_mean_std must not update here
        try:
            kappa = None
            for _ in range(max(1, int(self.at['tnp_steps']))):
                idx = torch.randint(0, obs_all.shape[0], (M,), device=obs_all.device)
                actor_idx = idx // self.horizon_length     # swap_and_flatten01 order: actor-major
                with torch.no_grad():
                    o = self.model.norm_obs(obs_all[idx])
                    z_s, z_t = net.actor.encode(o)
                    r = kit.refinement_need(net.actor.diffusion_net, z_s, z_t, net.alpha_bar,
                                            net.actor.action_dim, self.at['n_min'], self.at['n_max'])
                if kappa is None:
                    kappa = self.kappa.update(r)
                with torch.enable_grad():
                    loss = kit.tnp_loss(net.actor.tnp_logits, z_t, r, kappa)
                    stats = {'loss_tnp': float(loss.detach())}
                    label, kinds = self._privileged_label(actor_idx)
                    if self.at['tnp_priv']:
                        lp = kit.privileged_tnp_loss(lambda z: net.tnp_head_logits('priv', z), z_t, label)
                        loss = loss + lp
                        stats['loss_tnp_priv'] = float(lp.detach())
                    self.tnp_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.tnp_opt.step()
            with torch.no_grad():
                eta = torch.sigmoid(net.actor.tnp_logits(z_t))
                n = kit.eta_to_steps(eta, self.at['n_min'], self.at['n_max'])
                y = (r / kappa).clamp(max=1.0)
                e, t = eta.reshape(-1), y.reshape(-1)
                corr = ((e - e.mean()) * (t - t.mean())).mean() / (e.std(unbiased=False) * t.std(unbiased=False)).clamp(min=1e-12)
                stats.update({'kappa': kappa, 'r_mean': float(r.mean()), 'target_mean': float(y.mean()),
                              'target_std': float(y.std()), 'tnp_corr': float(corr),   # corr(eta, target) per sample
                              'eta_mean': float(eta.mean()), 'eta_std': float(eta.std()),
                              'n_mean': float(n.float().mean())})
                if kinds is not None:
                    for k in torch.unique(kinds).tolist():
                        m = kinds == k
                        stats[f'eta_{KIND_NAMES[k]}'] = float(eta[m].mean())
                        stats[f'target_{KIND_NAMES[k]}'] = float(y[m].mean())
        finally:
            self.model.train(was_training)
        return stats
