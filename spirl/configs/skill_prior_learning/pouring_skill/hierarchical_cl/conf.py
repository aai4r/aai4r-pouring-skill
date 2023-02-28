import os

from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.components.logger import Logger
from spirl.utility.general_utils import AttrDict
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.rl.envs.real_ur3_env import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': ClSPiRLMdl,
    'logger': Logger,
    'data_dir': [os.path.join(os.environ["DATA_DIR"], 'pouring_skill')],
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
    nz_enc=128,
    nz_mid=128,
    nz_vae=7,
    n_processing_layers=3,
    num_prior_net_layers=3,
    cond_decode=True,
    weights_dir="weights",
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
