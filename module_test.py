import os
import sys
import imp

from spirl.rl.train import RLTrainer
from spirl.rl.components.params import get_args
from spirl.rl.envs.isaacgym_env import IsaacGymEnv
from spirl.rl.train import get_exp_dir, get_config_path
from spirl.utils.general_utils import AttrDict

# isaacgym modules
from isaacgym import gymutil
from skill_rl.config import load_cfg
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


def get_config(args):
    conf = AttrDict()

    # paths
    conf.exp_dir = get_exp_dir()
    conf.conf_path = get_config_path(args.path)

    # general and agent configs
    print('loading from the config file {}'.format(conf.conf_path))
    conf_module = imp.load_source('conf', conf.conf_path)
    conf.general = conf_module.configuration
    conf.agent = conf_module.agent_config
    conf.agent.device = "cude"

    # data config
    conf.data = conf_module.data_config

    # environment config
    conf.env = conf_module.env_config
    conf.env.device = "cuda"  # add device to env config as it directly returns tensors

    # sampler config
    conf.sampler = conf_module.sampler_config if hasattr(conf_module, 'sampler_config') else AttrDict({})

    # model loading config
    conf.ckpt_path = conf.agent.checkpt_path if 'checkpt_path' in conf.agent else None

    # load notes if there are any
    conf.notes = conf_module.notes

    return conf


def isaacgym_env_test():
    # spirl
    args = AttrDict(path="spirl/configs/hrl/pouring_water/spirl_cl")
    conf = get_config(args)

    # issacgym
    # Task name format: $ROBOT_TASK: $CONFIG
    task_list = {"UR3_POURING": {"task": "DemoUR3Pouring", "config": "expert_ur3_pouring.yaml"}}
    target = "UR3_POURING"
    print("Target Task: {}".format(target))
    args_i = gymutil.parse_arguments(description="IsaacGym Task " + target)
    cfg_i = load_cfg(cfg_file_name=task_list[target]['config'])

    env = IsaacGymEnv(config=conf)
    env.reset()


if __name__ == "__main__":
    print("***** Module test code *****")
    isaacgym_env_test()
