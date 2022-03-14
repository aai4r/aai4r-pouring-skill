

from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import BaseEnvironment

from skill_rl.expert_ur3_pouring import DemoUR3Pouring
from tasks.base.vec_task import VecTaskPython


def parse_task_py(args, cfg, sim_params):
    device_id = args.device_id
    rl_device = args.rl_device

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        raise Exception("Unrecognized task!\n"
                        "Task should be one of ")
    env = VecTaskPython(task, rl_device)
    return task, env


class IsaacGymEnv(BaseEnvironment):
    def __init__(self, isaac_config):
        """
        required methods
        val_mode()  # OK!
        step()      # OK!
        reset()     # OK!
        render()
        get_episode_info()
        """
        self.config = isaac_config
        task, env = parse_task_py(args=self.config.args, cfg=self.config.cfg, sim_params=self.config.sim_params)
        self._env = env

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "pouring_water",
        }))

    def step(self, action):
        pass

    def reset(self):
        obs = self._env.reset()
        # obs = self._wrap_observation(obs)
        return obs

    def get_episode_info(self):
        pass

    def _postprocess_info(self, info):
        pass
