

from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class IsaacGymEnv(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "pouring_water",
        }))

    def _make_env(self, id):
        print("env id: ".format(id))

    def step(self, action):
        pass

    def reset(self):
        pass

    def get_episode_info(self):
        pass

    def _postprocess_info(self, info):
        pass
