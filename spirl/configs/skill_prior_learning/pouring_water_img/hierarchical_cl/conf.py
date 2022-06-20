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
    'epoch_cycles_train': 100,
    'num_epochs': 100,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec_img.state_dim,
    action_dim=data_spec_img.n_actions,
    state_cond_pred=False,   # TODO  # robot state(joint, gripper) conditioned prediction
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    prior_input_res=data_spec_img.res,
    n_input_frames=2,
    nz_vae=32,                  # skill embedding dim.
    nz_enc=256,                 # encoder output dim. (img -> nz_enc)
    n_processing_layers=3,      # num_layers of skill decoder
    num_prior_net_layers=2,     # prior_net Predictor
    cond_decode=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec_img
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames
