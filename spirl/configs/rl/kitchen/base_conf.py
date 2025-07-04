import os

from spirl.utility.general_utils import AttrDict
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.policies.mlp_policies import MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.rl.components.normalization import Normalizer
from spirl.configs.default_data_configs.kitchen import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'non-hierarchical RL experiments in kitchen env'

configuration = {
    'seed': 42,
    'agent': SACAgent,
    'environment': KitchenEnv,
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 280,
    'n_steps_per_epoch': 50000,
    'n_warmup_steps': 5e3,
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    n_layers=5,      # number of policy network layers
    nz_mid=256,
    max_action_range=1.,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
    output_dim=1,
    n_layers=2,      # number of policy network layers
    nz_mid=256,
    action_input=True,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

# Agent
agent_config = AttrDict(
    policy=MLPPolicy,
    policy_params=policy_params,
    critic=MLPCritic,
    critic_params=critic_params,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
    batch_size=256,
    log_video_caption=True,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
)

