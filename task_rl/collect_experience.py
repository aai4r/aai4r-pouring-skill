"""
Collection of experience in isaacgym environment
"""

from isaacgym import gymutil
from task_rl.config import load_cfg
from task_rl.expert_manager import ExpertManager
from utils.config import parse_sim_params
from tasks.base.vec_task import VecTaskPython

import torch
import os

from task_rl.expert_ur3_pouring import DemoUR3Pouring

# Task name format: $ROBOT_TASK: $CONFIG
task_list = {"UR3_POURING": {"task": "DemoUR3Pouring", "config": "expert_ur3_pouring.yaml"},
             "Another_Task": {"task": None, "config": None}
             }


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
    if cfg['expert']['img_obs']:
        env.clip_obs = 255
    return task, env


def task_demonstration(task):
    print("Target Task: {}".format(task))
    args = gymutil.parse_arguments(description="IsaacGym Task " + task)
    cfg = load_cfg(cfg_file_name=task_list[task]['config'])
    sim_params = parse_sim_params(args, cfg, None)

    # param customization
    cfg['env']['numEnvs'] = 64
    cfg['expert']['num_total_frames'] = 200000
    cfg['expert']['save_data'] = True
    cfg['expert']['debug_cam'] = False
    cfg['expert']['img_obs'] = False

    if torch.cuda.device_count() > 1:
        args.task = task_list[task]['task']
        args.device = args.sim_device_type
        args.compute_device_id = 0
        args.device_id = 0
        args.graphics_device_id = 0
        args.rl_device = 'cuda:0'
        args.sim_device = 'cuda:0'
        args.headless = False
        args.test = True

    task, env = parse_task_py(args=args, cfg=cfg, sim_params=sim_params)

    # params
    cfg['device'] = 'cpu' if cfg['device_type'] == 'cpu' else cfg['device_type'] + ':' + str(cfg['device_id'])
    num_total_frames = cfg['expert']['num_total_frames']
    num_transitions_per_env = round(num_total_frames / env.num_envs + 0.51)
    print("===== Frame Info. =====")
    print("num_total_frames / num_envs: {} / {}".format(num_total_frames, env.num_envs))
    print("  ==> num_transition_per_env: {}".format(num_transitions_per_env))

    print("args: ", args)
    print("cfg::: ", cfg)
    print("sim_params: ", sim_params)

    expert = ExpertManager(vec_env=env, num_transition_per_env=num_transitions_per_env, cfg=cfg)
    expert.run(num_transitions_per_env=num_transitions_per_env)
    # task_rl.load()


if __name__ == '__main__':
    print("Task Demonstration Dataset")
    task = "UR3_POURING"
    task_demonstration(task=task)
