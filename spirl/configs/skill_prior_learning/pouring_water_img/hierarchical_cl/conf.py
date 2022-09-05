import os

from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.models.skill_prior_mdl import SkillSpaceLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.isaacgym_envs import data_spec_img
from spirl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageClSPiRLMdl,
    'logger': SkillSpaceLogger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'pouring_water_img'),
    'epoch_cycles_train': 10,
    'num_epochs': 101,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec_img.state_dim,
    action_dim=data_spec_img.n_actions,
    aux_pred_dim=9,     # gripper, bottle, cup position, set zero for only actions
    aux_pred_index=[13, 14, 15, 23, 24, 25, 30, 31, 32],
    state_cond_pred=False,   # TODO  # robot state(joint, gripper) conditioned prediction
    n_rollout_steps=10,
    kl_div_weight=2e-4,
    prior_input_res=data_spec_img.res,
    n_input_frames=2,
    nz_vae=12,                  # skill embedding dim.
    nz_enc=256,                 # encoder output dim. (img -> nz_enc)
    nz_mid_prior=128,
    n_processing_layers=3,      # num_layers of skill decoder
    num_prior_net_layers=3,     # prior_net Predictor
    cond_decode=True,
    state_cond=True,
    state_cond_size=6,          # only joint values
    use_pretrain=True,
    weights_dir="weights",
)
model_config.weights_dir += "_pre" if model_config.use_pretrain else ""
model_config.weights_dir += "_st_cond" if model_config.state_cond else ""

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec_img
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames
