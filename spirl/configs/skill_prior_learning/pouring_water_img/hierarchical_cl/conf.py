import os

from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.models.skill_prior_mdl import SkillSpaceLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.isaacgym_envs import data_spec_img
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.utils.remote_server_utils import WeightNaming

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageClSPiRLMdl,
    'logger': SkillSpaceLogger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'pouring_water_img_vr'),
    'epoch_cycles_train': 10,
    'num_epochs': 300,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

# [13, 14, 15]: EE position
# [23, 24, 25]: bottle position
# [30, 31, 32]: cup position
# [40, 41, 42]: EE to bottle pos. difference
# [43, 44, 45, 46]: EE to bottle rot. difference
aux_pred_indices = [13, 14, 15, 23, 24, 25, 30, 31, 32, 40, 41, 42, 43, 44, 45, 46]
model_config = AttrDict(
    state_dim=data_spec_img.state_dim,
    action_dim=data_spec_img.n_actions,
    aux_pred_dim=len(aux_pred_indices),     # gripper, bottle, cup position, set zero for only actions
    aux_pred_index=aux_pred_indices,
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
    layer_freeze=5,             # 5: freeze for skill train, -1: freeze all layers for policy train
    recurrent_prior=True,
    weights_dir="weights",
)

WeightNaming.weights_name_convert(model_config)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec_img
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames
