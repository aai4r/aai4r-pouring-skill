from spirl.configs.hrl.pouring_water_img.spirl.conf import *
from spirl.rl.policies.cl_model_policies import ACClModelPolicy

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ImageClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/pouring_water_img/hierarchical_cl"),
    policy_model_epoch='142',  # default: latest
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
# by twkim, we try to finetune the skill decoder

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec_img.n_actions,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=3,  # number of critic network layer
    nz_mid=256,
    nz_enc=128,
    action_input=True,
    unused_obs_size=ll_model_params.prior_input_res ** 2 * 3 * ll_model_params.n_input_frames + hl_policy_params.action_dim,
    critic_lr=3.0e-4,
    alpha_lr=2e-4,
)

ll_agent_config = AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,                   # LL critic is not used since we are not finetuning LL
    critic_params=ll_critic_params,             # hl_critic_params
)

# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    prior_model_epoch=ll_policy_params.policy_model_epoch,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))

sampler_config = AttrDict(
    n_frames=2,
)

