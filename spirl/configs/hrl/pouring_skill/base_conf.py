import os
import copy

from spirl.utility.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.policies.mlp_policies import MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.envs.real_ur3 import RealUR3Env
from spirl.rl.components.sampler import HierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.skill_space_agent import SkillSpaceAgent
from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.rl.envs.real_ur3 import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': RealUR3Env,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 50,
    'max_rollout_len': 500,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 5e3,  # 5e3
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    batch_size=512,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    num_prior_net_layers=3,
    n_processing_layers=3,
    nz_vae=12,
    n_rollout_steps=10,
    model_download=False,
    state_cond_pred=False,
    weights_dir="weights",
    aux_pred_dim=len([]),         # C, 0 or 9
    state_cond=False,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=SkillPriorMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill_prior_learning/pouring_skill/hierarchical"),
))


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    nz_mid=256,
    n_layers=5,
    weights_dir="weights",
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=MLPPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=True,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

cfg = AttrDict()
cfg.extra = AttrDict()
cfg.extra.skill_uncertainty_plot = False
env_config = AttrDict(
    reward_norm=1.,
    img_debug=False,
    img_disp_delay=1,
    cfg=cfg,
)
