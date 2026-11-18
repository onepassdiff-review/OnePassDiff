"""PPO-MLP + joint-limit loss: the PPO-MLP+L_joint control of the paper.

The PPO-MLP baseline (rl_games' Gaussian MLP actor, network `actor_critic`, config BipedPPOMLPMatched) trained exactly
as PPO-MLP, plus the M3 joint-limit loss of the diffusion actor on its action mean:

    loss += lambda_joint * L_joint(q0 + k_a * mu),   L_joint = mean_batch sum_j relu(q - q_hi)^2 + relu(q_lo - q)^2

with the same function (experiments/onepass_kit.joint_limit_loss), the same limits (onepass_agent.joint_info:
the env's dof soft limits = URDF limits) and the same weight (0.1) as BipedPPOOnePass. Nothing else changes, so
PPO-MLP vs. this run isolates the joint-limit loss, and this run vs. the diffusion PPO-only / full runs isolates the
actor architecture.

Registered in envs/common/torch_runner.py as algo name `a2c_continuous_joint_mlp`; envs/exp_onepass.sh
VARIANT=ppo_mlp_joint. Only option: params.config.onepass.lambda_joint. Written for rl_games 1.6.1 (pinned by
IsaacGymEnvs, used on the training machine): calc_gradients is that version's code with one marked addition, the same
copy OnePassAgent uses.
"""
import torch
from rl_games.algos_torch import a2c_continuous, torch_ext
from rl_games.common import common_losses

from experiments import onepass_kit as kit
from envs.common.onepass_agent import joint_info, _unwrap_task


class JointMLPAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        user = dict(params['config'].get('onepass', None) or {})
        unknown = set(user) - {'lambda_joint'}
        if unknown:
            raise ValueError(f'a2c_continuous_joint_mlp accepts only onepass.lambda_joint, got {sorted(unknown)}')
        self.lambda_joint = float(user.get('lambda_joint', 0.1))
        self.task = _unwrap_task(self.vec_env) if self.vec_env is not None else None
        q_def, k_a, lo, hi = joint_info(self.task, self.actions_num, self.ppo_device)
        self.q_default, self.action_scale = q_def, torch.tensor(k_a, dtype=torch.float, device=self.ppo_device)
        self.q_lower, self.q_upper = lo, hi
        self._jstats, self._aux_added = [], 0
        print(f'[joint_mlp] PPO-MLP + joint-limit loss: lambda_joint={self.lambda_joint}, k_a={k_a}\n'
              f'            q_default={q_def.tolist()}\n            lower={lo.tolist()}\n            upper={hi.tolist()}')

    def train_epoch(self):
        self._aux_added, self._jstats = 0, []
        out = super().train_epoch()
        if self._aux_added == 0:
            raise RuntimeError('[joint_mlp] the joint-limit loss reached no minibatch this iteration')
        if self.writer is not None and self._jstats:
            n = len(self._jstats)
            self.writer.add_scalar('onepass/loss_joint', sum(s[0] for s in self._jstats) / n, self.frame)
            self.writer.add_scalar('onepass/joint_active_frac', sum(s[1] for s in self._jstats) / n, self.frame)
            self.writer.add_scalar('onepass/aux_minibatches', float(self._aux_added), self.frame)
        return out

    def calc_gradients(self, input_dict):
        """rl_games 1.6.1 A2CAgent.calc_gradients, unchanged except for the block marked JOINT."""
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

            # ---- JOINT: M3 joint-limit loss on the predicted joint targets q0 + k_a * mu (as the diffusion actor)
            l_j, active = kit.joint_limit_loss(mu.float(), self.q_default, self.action_scale, self.q_lower, self.q_upper)
            loss = loss + self.lambda_joint * l_j
            self._jstats.append((float(l_j.detach()), float(active)))
            self._aux_added += 1
            # ---- end JOINT

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
