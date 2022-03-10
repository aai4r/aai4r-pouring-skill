from spirl.utils.general_utils import AttrDict
from expert.data.isaacgym.src.isaacgym_data_loader import IsaacGymSequenceDataLoader


data_spec = AttrDict(
    dataset_class=IsaacGymSequenceDataLoader,
    n_actions=7,
    state_dim=24,
    env_name="pouring_water",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280
