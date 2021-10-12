import os

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_ppo import process_ppo

from tasks.mirobot import MirobotCube
from tasks.ur3_pouring import UR3Pouring

import torch


def enjoy(resume=None):
    if torch.cuda.device_count() > 1:
        args.compute_device_id = 1
        args.device_id = 1
        args.graphics_device_id = 1
        args.rl_device = 'cuda:1'
        args.sim_device = 'cuda:1'
        args.test = True
        args.num_envs = 64

        cfg['device_id'] = 1
        cfg['env']['numEnvs'] = 64

        if not resume:
            file_list = os.listdir(logdir)
            file_list = list(filter(lambda x: 'model_' in x, file_list))
            file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            latest_num = file_list[-1].split('_')[-1].split('.')[0]
        else:
            latest_num = resume

        cfg_train['resume'] = latest_num
        cfg_train['learn']['resume'] = latest_num
        cfg_train['learn']['print_log'] = False     # tensorboard log

    print("args: ", args)
    print("cfg: ", cfg)
    print("cfg_train: ", cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    ppo = process_ppo(args, env, cfg_train, logdir)

    ppo_iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        ppo_iterations = args.max_iterations

    ppo.run(num_learning_iterations=ppo_iterations, log_interval=cfg_train["learn"]["save_interval"])


def run_test():
    print("args: ", args)
    print("cfg: ", cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    print("env: ", env)


if __name__ == '__main__':
    set_np_formatting()

    # [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]
    # [MirobotCube, UR3Pouring]
    target_task = 'UR3Pouring'
    args = get_args(target_task=target_task, headless=False)

    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    enjoy()
    # run_test()

