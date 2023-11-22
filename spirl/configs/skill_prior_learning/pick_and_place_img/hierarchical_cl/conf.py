import os

from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.components.logger import Logger
from spirl.utility.general_utils import AttrDict
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.rl.envs.real_ur3_env import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': ImageClSPiRLMdl,
    'logger': Logger,
    'data_dir': [os.path.join(os.environ["DATA_DIR"], 'pick_and_place_img')],
    'epoch_cycles_train': 100,
    'num_epochs': 100,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

data_spec.max_seq_len = 280

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    state_cond_pred=False,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=256,
    nz_mid=256,
    nz_vae=12,
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
    recurrent_prior=False,
    dropout=False,
    mc_dropout=False,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
