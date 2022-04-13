import os
import copy
import isaacgym
import torch
from init_conf import project_home_path

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.components.critic import SplitObsMLPCritic
from spirl.rl.envs.isaacgym_env import PouringWaterEnv
from spirl.rl.components.sampler import ACMultiImageAugmentedHierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.skill_space_agent import SkillSpaceAgent
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.configs.default_data_configs.isaacgym_envs import data_spec_img

from utils.config import parse_sim_params
from task_rl.config import load_cfg
from isaacgym import gymutil

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the isaacgym env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': PouringWaterEnv,
    'sampler': ACMultiImageAugmentedHierarchicalSampler,
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
sampler_config = AttrDict(
    n_frames=2,
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
    state_dim=data_spec_img.state_dim,
    action_dim=data_spec_img.n_actions,
    kl_div_weight=2e-4,
    n_input_frames=2,
    prior_input_res=data_spec_img.res,
    nz_vae=10,
    n_rollout_steps=16,
)
# nz_enc=128,
# nz_mid=128,
# n_processing_layers=5,

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=ImageClSPiRLMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/pouring_water_img/hierarchical"),
))


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec_img.state_dim,
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
    policy=ACLearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=hl_critic_params,
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=True,
)

# reduce replay capacity because we are training image-based, do not dump (too large)
from spirl.rl.components.replay_buffer import SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = ll_model_params.prior_input_res**2*3 * 2 + \
                                                             hl_agent_config.policy_params.action_dim   # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False


# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec_img

# IsaacGym Environment
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
    img_debug=True,
    img_disp_delay=1
)
