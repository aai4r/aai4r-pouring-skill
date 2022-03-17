"""
Task demonstration by manual control
Candidates:
* Pouring Skill
* Pick & Place
* Etc.
"""
import time

from isaacgym import gymutil
from skill_rl.config import load_cfg
from skill_rl.expert_manager import ExpertManager
from skill_rl.skill_rl_trainer import SkillRLTrainer
from spirl.rl.components.params import get_args
from utils.config import parse_sim_params
from tasks.base.vec_task import VecTaskPython
from spirl.utils.general_utils import AttrDict


import os
import sys
import torch

from skill_rl.expert_ur3_pouring import DemoUR3Pouring

# Task name format: $ROBOT_TASK: $CONFIG
task_list = {"UR3_POURING": {"task": "DemoUR3Pouring", "config": "expert_ur3_pouring.yaml"}}


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


def task_demonstration():
    print("Target Task: {}".format(target))
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

    task, env = parse_task_py(args=args, cfg=cfg, sim_params=sim_params)

    # frame params
    num_total_frames = cfg['expert']['num_total_frames']
    num_transitions_per_env = round(num_total_frames / env.num_envs + 0.51)
    print("===== Frame Info. =====")
    print("num_total_frames / num_envs: {} / {}".format(num_total_frames, env.num_envs))
    print("  ==> num_transition_per_env: {}".format(num_transitions_per_env))

    expert = ExpertManager(vec_env=env, num_transition_per_env=num_transitions_per_env, cfg=cfg, device=env.rl_device)
    expert.run(num_transitions_per_env=num_transitions_per_env)
    # skill_rl.load()


def task_rl_train():
    print("Target Task: {}".format(target))
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

    # task, env = parse_task_py(args=args, cfg=cfg, sim_params=sim_params)
    isaac_config = AttrDict(args=args, cfg=cfg, sim_params=sim_params)

    # # frame params
    # num_total_frames = cfg['expert']['num_total_frames']
    # num_transitions_per_env = round(num_total_frames / env.num_envs + 0.51)
    # print("===== Frame Info. =====")
    # print("num_total_frames / num_envs: {} / {}".format(num_total_frames, env.num_envs))
    # print("  ==> num_transition_per_env: {}".format(num_transitions_per_env))

    os.environ["EXP_DIR"] = "../experiments"
    os.environ["DATA_DIR"] = "../data"

    # with multi-GPU env, using only single GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    task_name = "pouring_water"
    mode = "spirl_cl"
    sys.argv.append("--path=" + "../spirl/configs/hrl/{}/{}".format(task_name, mode))
    sys.argv.append("--seed={}".format(0))
    sys.argv.append("--prefix={}".format("SPIRL_" + task_name + "_seed0"))

    train = SkillRLTrainer(args=get_args(), isaac_config=isaac_config)
    # train.run(num_transitions_per_env=num_transitions_per_env)
    # skill_rl.load()


if __name__ == '__main__':
    print("Task Demonstration Dataset")
    target = "UR3_POURING"

    # task_demonstration()
    task_rl_train()
