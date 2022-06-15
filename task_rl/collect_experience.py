"""
Collection of experience in isaacgym environment
"""
import numpy as np
from isaacgym import gymutil
from task_rl.config import load_cfg
from task_rl.expert_manager import ExpertManager
from utils.config import parse_sim_params
from tasks.base.vec_task import VecTaskPython

import torch
import subprocess as sp
import os

from task_rl.expert_ur3_pouring import DemoUR3Pouring

# Task name format: $ROBOT_TASK: $CONFIG
task_list = {"UR3_POURING": {"task": "DemoUR3Pouring", "config": "expert_ur3_pouring.yaml"},
             "UR3_POURING_IMG": {"task": "DemoUR3Pouring", "config": "expert_ur3_pouring.yaml"},
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


def get_gpu_free_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def task_demonstration(task):
    print("Target Task: {}".format(task))
    args = gymutil.parse_arguments(description="IsaacGym Task " + task)
    cfg = load_cfg(cfg_file_name=task_list[task]['config'])
    sim_params = parse_sim_params(args, cfg, None)

    # param customization
    cfg['env']['numEnvs'] = 32
    cfg['env']['enableDebugVis'] = False
    cfg['expert']['num_total_frames'] = 1500000
    cfg['expert']['desired_batch_size'] = 5 * (1000 * 1000 * 1000)  # GB
    cfg['expert']['save_data'] = True
    cfg['expert']['save_resume'] = True
    cfg['expert']['debug_cam'] = False
    cfg['expert']['img_obs'] = True

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

    # memory size check
    num_total_frames = cfg['expert']['num_total_frames']
    num_iter = 1
    if cfg['expert']['img_obs']:
        gpu_mems_mb = get_gpu_free_memory()
        target_gpu_free_mem_mb = gpu_mems_mb[args.device_id]

        h, w = cfg['env']['cam_height'], cfg['env']['cam_width']
        req_dominant_data_size_mb = (num_total_frames * (h * w * 3 * np.dtype(np.uint8).itemsize)) / (1000 * 1000)
        num_iter = int(np.ceil(req_dominant_data_size_mb / target_gpu_free_mem_mb))
        print("Required dominant dataset size (MB): ", req_dominant_data_size_mb)
        print("Target gpu free memory size (MB): ", target_gpu_free_mem_mb)
        print("Number of iterations: ", num_iter)

    num_transitions_per_env = round(num_total_frames / env.num_envs / num_iter + 0.51)
    print("===== Frame Info. =====")
    print("num_total_frames / num_envs / num_iter: {} / {} / {}".format(num_total_frames, env.num_envs, num_iter))
    print("  ==> num_transition_per_env: {}".format(num_transitions_per_env))

    print("args: ", args)
    print("cfg::: ", cfg)
    print("sim_params: ", sim_params)

    expert = ExpertManager(vec_env=env, num_transition_per_env=num_transitions_per_env, cfg=cfg)
    for i in range(num_iter):
        print("iter {}".format(i))
        expert.run(num_transitions_per_env=num_transitions_per_env)
    # task_rl.load()


if __name__ == '__main__':
    print("Task Demonstration Dataset")
    task = "UR3_POURING"
    task_demonstration(task=task)
