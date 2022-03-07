"""
Task demonstration by manual control
Candidates:
* Pouring Skill
* Pick & Place
* Etc.
"""

from isaacgym import gymutil
from expert.config import load_cfg
from expert.expert_manager import ExpertManager
from utils.config import parse_sim_params
from tasks.base.vec_task import VecTaskPython


import torch

from expert.expert_ur3_pouring import DemoUR3Pouring

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
    # expert.run(num_transitions_per_env=num_transitions_per_env)
    expert.load()


if __name__ == '__main__':
    print("Task Demonstration Dataset")
    target = "UR3_POURING"

    task_demonstration()
