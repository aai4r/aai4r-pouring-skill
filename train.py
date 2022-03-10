import os
import subprocess

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_ppo import process_ppo


def isaacgym_rl_train():
    set_np_formatting()

    # [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]
    # [MirobotCube, UR3Pouring]
    target_task = 'UR3Pouring'
    args = get_args(target_task=target_task, headless=True)

    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    task, env = parse_task(args, cfg, cfg_train, sim_params)
    ppo = process_ppo(args, env, cfg_train, logdir)

    ppo_iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        ppo_iterations = args.max_iterations

    ppo.run(num_learning_iterations=ppo_iterations, log_interval=cfg_train["learn"]["save_interval"])


def skill_train():
    print("***** main run code *****")
    # make folders if not exists
    folders = ['data', 'experiments']
    for fd in folders:
        if not os.path.exists(fd):
            os.mkdir(fd)

    # add env. variables to run
    os.environ["EXP_DIR"] = "./experiments"
    os.environ["DATA_DIR"] = "./data"

    # with multi-GPU env, using only single GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    task_name = "pouring_water"
    skill_mode = "hierarchical_cl"
    skillPriorCmd = ["python", "spirl/train.py",
                     "--path=spirl/configs/skill_prior_learning/{}/{}".format(task_name, skill_mode),
                     "--val_data_size={}".format(160)]

    rl_mode = "spirl_cl"
    spirlCmd = ["python3", "spirl/rl/train.py",
                "--path=spirl/configs/hrl/{}/{}".format(task_name, rl_mode),
                "--seed={}".format(0),
                "--prefix=SPIRL_kitchen_seed0",
                "--mode=val"]  # "train"(default) or "val"

    subprocess.call([" ".join(skillPriorCmd)], shell=True)


if __name__ == '__main__':
    # isaacgym_rl_train()
    skill_train()
