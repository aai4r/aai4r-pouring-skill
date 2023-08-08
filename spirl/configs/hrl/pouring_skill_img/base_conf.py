import os
import copy

from spirl.utility.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.components.critic import SplitObsMLPCritic
from spirl.rl.components.sampler import ACMultiImageAugmentedHierarchicalSampler, ACImageAugmentedSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from spirl.rl.agents.ac_agent import SACAgent
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.components.data_loader import GlobalSplitVideoDataset

from spirl.rl.envs.real_ur3_env import RealUR3Env
from spirl.rl.envs.isaacgym_env import PouringWaterEnv, IsaacGymEnv, args
from utils.config import parse_sim_params


data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    state_dim=47,
    n_actions=7,
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    env_name="pouring_skill_img",   # "pouring_skill_img"  # TODO, how it is affected?
    res=150,
    crop_rand_subseq=True,
)


current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': PouringWaterEnv,
    'sampler': ACImageAugmentedSampler,
    'data_dir': '.',
    'num_epochs': 50,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 1.5e3,  # 5e3
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
    capacity=1e4,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    batch_size=32,
    update_iteration=1,
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
    state_cond_pred=False,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    n_input_frames=1,
    nz_enc=256,
    nz_mid=256,
    nz_vae=7,
    n_processing_layers=3,
    num_prior_net_layers=3,
    cond_decode=True,
    use_pretrain=True,
    layer_freeze=-1,             # 5: freeze for skill train, -1: freeze all layers
    state_cond=True,
    state_cond_size=data_spec.state_dim,
    model_download=False,
    aux_pred_dim=len([]),     # gripper, bottle, cup position, set zero for only actions
    prior_input_res=data_spec.res,
    weights_dir="weights",
    recurrent_prior=False,   # D
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=ImageClSPiRLMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/pouring_skill_img/hierarchical"),
))


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    nz_mid=256,
    nz_enc=256,
    n_layers=3,
    policy_lr=3e-4,
    state_cond=ll_model_params.state_cond,
    state_cond_size=ll_model_params.state_cond_size,
    weights_dir=ll_model_params.weights_dir,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=3,  # number of policy network layer
    nz_mid=256,
    nz_enc=128,
    action_input=True,
    unused_obs_size=ll_model_params.prior_input_res ** 2 * 3 * ll_model_params.n_input_frames,
    critic_lr=3.0e-4,
    alpha_lr=2e-4,
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

from spirl.rl.components.replay_buffer import SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = ll_model_params.prior_input_res**2*3 * 2 + \
                                                             hl_agent_config.policy_params.action_dim   # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec

from task_rl.config import load_cfg
from isaacgym import gymapi
cfg = load_cfg(cfg_file_name="expert_ur3_pouring.yaml")

sim_params = parse_sim_params(args, cfg, None)

dev_num = 0
args.device_id = dev_num
args.rl_device = "cuda:{}".format(dev_num)
args.device = "cuda:{}".format(dev_num)
args.task = "DemoUR3Pouring"
args.physics_engine = gymapi.SIM_PHYSX
args.headless = False

env_config = AttrDict(
    reward_norm=1.,
    image_observation=True,
    img_debug=False,
    img_disp_delay=1,
    cfg=cfg,
    args=args,
    sim_params=sim_params
)
