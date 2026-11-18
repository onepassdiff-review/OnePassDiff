from rl_games.torch_runner import Runner as _Runner
import torch
import torch._dynamo
# PyTorch 2.4.x: int_truediv on symbolic shapes crashes inside dynamo's
# insert_deferred_runtime_asserts. Fall back to eager on such failures.
torch._dynamo.config.suppress_errors = True
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import register_network
import os

# bug fix to load multi-gpu checkpoint
def safe_load(filename):
    return torch_ext.safe_filesystem_op(torch.load, filename, map_location='cuda:0')
torch_ext.safe_load = safe_load

class Runner(_Runner):

    def __init__(self, algo_observer=None):
        from envs.common.diffusion_policy import DiffusionActorCriticBuilder
        register_network('diffusion_actor_critic', DiffusionActorCriticBuilder)
        super().__init__(algo_observer)
        # OnePassDiff: PPO + M1/M2/M3 (algo name a2c_continuous_onepass). Needs
        # experiments/onepass_kit.py importable from the repository root.
        try:
            from envs.common.onepass_agent import OnePassAgent
            from rl_games.algos_torch import players
            self.algo_factory.register_builder(
                'a2c_continuous_onepass', lambda **kwargs: OnePassAgent(**kwargs))
            self.player_factory.register_builder(
                'a2c_continuous_onepass', lambda **kwargs: players.PpoPlayerContinuous(**kwargs))
        except ImportError as e:
            print(f'[torch_runner] a2c_continuous_onepass not available: {e}')
        # Control: PPO-MLP + the joint-limit loss (algo name a2c_continuous_joint_mlp).
        try:
            from envs.common.joint_mlp_agent import JointMLPAgent
            from rl_games.algos_torch import players
            self.algo_factory.register_builder(
                'a2c_continuous_joint_mlp', lambda **kwargs: JointMLPAgent(**kwargs))
            self.player_factory.register_builder(
                'a2c_continuous_joint_mlp', lambda **kwargs: players.PpoPlayerContinuous(**kwargs))
        except ImportError as e:
            print(f'[torch_runner] a2c_continuous_joint_mlp not available: {e}')

    def run(self, args):
        """Run either train/play depending on the args.

        Args:
            args (:obj:`dict`):  Args passed in as a dict obtained from a yaml file or some other config format.

        """
        if args['export']:
            self.run_export(args)
        elif args['play']:
            self.run_play(args)
        elif args['train']:
            self.run_train(args)
        else:
            self.run_train(args)

    def run_export(self, args):
        print("export not implemented")
        player = self.create_player()
        from rl_games.torch_runner import _restore, _override_sigma
        _restore(player, args)
        _override_sigma(player, args)

        import rl_games.algos_torch.flatten as flatten
        inputs = {
            'obs' : torch.zeros((1,) + player.obs_shape).to(player.device),
            'rnn_states' : player.states,
        }
        
        with torch.no_grad():
            adapter = flatten.TracingAdapter(ModelWrapper(player.model), inputs, allow_non_tensor=True)
            traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
            flattened_outputs = traced(*adapter.flattened_inputs)
            print(flattened_outputs)

        import onnx
        onnx_model_path = os.path.join(args['experiment_dir'], 'policy.onnx')
        torch.onnx.export(traced, *adapter.flattened_inputs, onnx_model_path, verbose=True, input_names=['obs'], output_names=['mu','log_std', 'value'])
        onnx_model = onnx.load(onnx_model_path)

        # Check that the model is well formed
        onnx.checker.check_model(onnx_model)

        import onnxruntime as ort
        import numpy as np
        from functools import partial
        ort_model = ort.InferenceSession(onnx_model_path)

        def get_action(self,obs,is_deterministic):
            mu, log_std, value = ort_model.run(
            None,
            {"obs": obs.cpu().numpy().astype(np.float32)},
            )
            
            return torch.tensor(mu).to(player.device)
        player.get_action = partial(get_action, player) # override get_action

        player.run()


class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        
        
    def forward(self,input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        #print(input_dict)
        #output_dict = self._model.a2c_network(input_dict)
        #input_dict['is_train'] = False
        #return output_dict['logits'], output_dict['values']
        return self._model.a2c_network(input_dict)