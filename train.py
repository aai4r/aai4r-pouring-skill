# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_ppo import process_ppo


def train():
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    ppo = process_ppo(args, env, cfg_train, logdir)

    ppo_iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        ppo_iterations = args.max_iterations

    ppo.run(num_learning_iterations=ppo_iterations, log_interval=cfg_train["learn"]["save_interval"])


if __name__ == '__main__':
    set_np_formatting()

    # [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]
    # [MirobotCube]
    target_task = 'MirobotCube'
    args = get_args(target_task=target_task, headless=True)

    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
