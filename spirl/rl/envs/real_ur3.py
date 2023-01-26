from spirl.rl.components.environment import BaseEnvironment
from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset


data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    n_actions=6,
    state_dim=10,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    env_name="pouring_skill",
    res=128,
    crop_rand_subseq=True,
)


class RtdeUR3:
    def __init__(self):
        # using VR and its trigger for safety
        pass

    def step(self, action):
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def reset(self):
        obs = None
        return obs


class RealUR3Env(BaseEnvironment):
    def __init__(self, config):
        self.config = config
        self._env = RtdeUR3()

    def _default_hparams(self):
        pass

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs = None
        return obs

    def render(self, mode='rgb_array'):
        raise NotImplementedError

    def _postprocess_info(self):
        pass
