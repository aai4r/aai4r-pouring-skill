"""
Task demonstration by manual control
Candidates:
* Pouring Skill
* Pick & Place
* Etc.
"""
import time

from isaacgym import gymutil
from task_rl.config import load_cfg
from task_rl.expert_manager import ExpertManager
from task_rl.task_rl_trainer import SkillRLTrainer
from spirl.rl.components.params import get_args
from utils.config import parse_sim_params
from tasks.base.vec_task import VecTaskPython
from spirl.utility.general_utils import AttrDict


import os
import sys
import torch

from task_rl.expert_ur3_pouring import DemoUR3Pouring

# Task name format: $ROBOT_TASK: $CONFIG
task_list = {"UR3_POURING": {"task": "DemoUR3Pouring", "config": "expert_ur3_pouring.yaml"}}


def task_rl_run():
    print("Target Task: {}".format(target))

    """ 
        ** IsaacGym Params ** 
    """
    args = gymutil.parse_arguments(description="IsaacGym Task " + target)
    cfg = load_cfg(cfg_file_name=task_list[target]['config'])

    # sim_params, physics_engine, device_type, device_id, headless
    sim_params = parse_sim_params(args, cfg, None)
    print("args: ", args)
    print("cfg::: ", cfg)
    print("sim_params: ", sim_params)

    if torch.cuda.device_count() > 1:
        args.task = task_list[target]['task']
        args.device = args.sim_device_type
        args.compute_device_id = 1
        args.device_id = 1
        args.graphics_device_id = 1
        args.rl_device = 'cuda:1'
        args.sim_device = 'cuda:1'
        args.headless = False
        args.test = True

    isaac_config = AttrDict(args=args, cfg=cfg, sim_params=sim_params)

    """ 
        ** SPiRL Params **
    """
    os.environ["EXP_DIR"] = "../experiments"
    os.environ["DATA_DIR"] = "../data"

    # with multi-GPU env, using only single GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    task_name = "pouring_water_img"     # ['pouring_water', 'pouring_water_img']
    mode = "spirl_cl"
    sys.argv.append("--path=" + "../spirl/configs/hrl/{}/{}".format(task_name, mode))
    sys.argv.append("--seed={}".format(0))
    sys.argv.append("--prefix={}".format("SPIRL_" + task_name + "_seed0"))
    # sys.argv.append("--mode={}".format('val'))      # ['train'(default), 'val', 'rollout']
    # sys.argv.append("--resume={}".format('latest'))     # latest or number..

    train = SkillRLTrainer(args=get_args(), isaac_config=isaac_config)


if __name__ == '__main__':
    print("Task Demonstration Dataset")
    target = "UR3_POURING"
    task_rl_run()
