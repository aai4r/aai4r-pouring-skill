import os
import copy
import isaacgym
import torch
from init_conf import project_home_path
from utils.config import parse_sim_params
from task_rl.config import load_cfg
from isaacgym import gymutil

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.policies.mlp_policies import MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.envs.isaacgym_env import IsaacGymEnv, PouringWaterEnv
from spirl.rl.components.sampler import HierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.skill_space_agent import SkillSpaceAgent
from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.configs.default_data_configs.isaacgym_envs import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the isaacgym env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': PouringWaterEnv,
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
    n_processing_layers=5,
    nz_vae=10,
    n_rollout_steps=16,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=SkillPriorMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill_prior_learning/pouring_water/hierarchical"),
))


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    nz_mid=256,
    n_layers=5,
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


# IsaacGym Environment config setup
task_list = {"UR3_POURING": {"task": "DemoUR3Pouring", "config": "expert_ur3_pouring.yaml"}}
target = "UR3_POURING"

args = gymutil.parse_arguments(description="IsaacGym Task " + target)
args.task = task_list[target]['task']
args.device = args.sim_device_type
args.headless = False
args.test = False

# if torch.cuda.device_count() > 1:
assert torch.cuda.get_device_name(1)
args.compute_device_id = 1
args.device_id = 1
args.graphics_device_id = 1
args.rl_device = 'cuda:1'
args.sim_device = 'cuda:1'

cfg = load_cfg(cfg_file_name=task_list[target]['config'], des_path=[project_home_path, "task_rl"])
cfg["env"]["asset"]["assetRoot"] = os.path.join(project_home_path, "assets")

sim_params = parse_sim_params(args, cfg, None)
env_config = AttrDict(
    reward_norm=1.,
    args=args,
    cfg=cfg,
    sim_params=sim_params,
    img_debug=False,
    img_disp_delay=1
)

# # Environment
# env_config = AttrDict(
#     reward_norm=1.,
# )

