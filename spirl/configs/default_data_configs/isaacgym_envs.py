from spirl.utils.general_utils import AttrDict
from task_rl.data.isaacgym.src.isaacgym_data_loader import IsaacGymSequenceDataLoader
from spirl.components.data_loader import GlobalSplitVideoDataset


data_spec = AttrDict(
    dataset_class=IsaacGymSequenceDataLoader,
    n_actions=7,
    state_dim=24,
    env_name="pouring_water",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280


data_spec_img = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    n_actions=7,
    state_dim=47,
    env_name="pouring_water_img",
    split=AttrDict(train=0.5, val=0.5, test=0.0),
    res=150,
    crop_rand_subseq=True,
)
data_spec_img.max_seq_len = 500
