import sys

import isaacgym
import torch
import os
import imp
from tensorboardX import SummaryWriter

from spirl.rl.components.params import get_args
from spirl.train import set_seeds, make_path, datetime_str, save_config, get_exp_dir, save_checkpoint
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.utility.general_utils import AttrDict, ParamDict, AverageTimer, timing, pretty_print
from spirl.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks, mpi_sum, mpi_gather_experience
from spirl.rl.utils.wandb import WandBLogger
from spirl.rl.components.sampler import Sampler
from spirl.rl.components.replay_buffer import RolloutStorage, PouringSkillRolloutStorage

from spirl.train import ModelTrainer
from spirl.components.params import get_args as _get_args

WANDB_PROJECT_NAME = 'spirl_project'
WANDB_ENTITY_NAME = 'twkim0812'


class SkillTrainer(ModelTrainer):
    def __init__(self):
        self.set_params()
        _args = _get_args()
        _args.init_only = True
        super().__init__(args=_args)
        self.start_epoch = 0
        if args.resume or self.conf.ckpt_path is not None:
            self.start_epoch = self.resume(args.resume, self.conf.ckpt_path)

    def set_params(self):
        task_name = "pouring_skill"  # [block_stacking, kitchen, maze, office, pouring_water, pouring_water_img]
        mode = "hierarchical_cl"

        sys.argv.append("--path=" + "./configs/skill_prior_learning/{}/{}".format(task_name, mode))
        # sys.argv.append("--val_data_size={}".format(160))  # TODO, automatic.. batch_size < val_data_size < (total_data * val_ratio)
        # sys.argv.append("--resume={}".format('latest'))     # latest or number..

    def skill_train(self, num_epochs):
        for epoch in range(self.start_epoch, num_epochs):
            self.train_epoch(epoch)

            if not self.args.dont_save:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, os.path.join(self._hp.exp_path, 'weights' if not hasattr(self.conf.model,
                                                                            "weights_dir") else self.conf.model.weights_dir),
                    CheckpointHandler.get_ckpt_name(epoch))


class SharedAutonomyTrainer:
    """Sets up RL training loop, instantiates all components, runs training."""
    def __init__(self, args):
        self.skill_trainer = SkillTrainer()

        self.args = args
        self.setup_device()

        # set up params
        self.conf = self._get_config()
        update_with_mpi_config(self.conf)   # self.conf.mpi = AttrDict(is_chef=True)
        self._hp = self.default_hparams()
        self._hp.overwrite(self.conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'log')
        print('using log dir: ', log_dir)

        # set seeds, display, worker shutdown
        if args.seed != -1: self._hp.seed = args.seed   # override from command line if set
        set_seeds(self._hp.seed)
        os.environ["DISPLAY"] = ":1"
        set_shutdown_hooks()

        # set up logging
        if self.is_chef:
            print("Running base worker.")
            self.logger = self.setup_logging(self.conf, self.log_dir)
        else:
            print("Running worker {}, disabled logging.".format(self.conf.mpi.rank))
            self.logger = None

        # build env
        self.conf.env.seed = self._hp.seed
        if 'task_params' in self.conf.env: self.conf.env.task_params.seed=self._hp.seed
        if 'general' in self.conf: self.conf.general.seed=self._hp.seed

        self.env = self._hp.environment(self.conf.env)
        self.conf.agent.env_params = self.env      # (optional) set params from env for agent
        if self.is_chef:
            pretty_print(self.conf)

        # build agent (that holds actor, critic, exposes update method)
        self.conf.agent.num_workers = self.conf.mpi.num_workers
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, self.logger, self._hp.max_rollout_len)

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0
        if args.resume or self.conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, self.conf.ckpt_path)
            self._hp.n_warmup_steps = 1500     # (default: 0) no warmup if we reload from checkpoint!

        # skill train
        self.shared_autonomy_train()

    def default_hparams(self):
        default_dict = ParamDict({
            'seed': None,
            'agent': None,
            'data_dir': None,  # directory where dataset is in
            'environment': None,
            'sampler': Sampler,     # sampler type used
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'max_rollout_len': 1000,  # maximum length of the performed rollout
            'n_steps_per_update': 1,     # number of env steps collected per policy update
            'n_steps_per_epoch': 20000,       # number of env steps per epoch
            'log_output_per_epoch': 100,  # log the non-image/video outputs N times per epoch
            'log_images_per_epoch': 4,    # log images/videos N times per epoch
            'logging_target': '',    # where to log results to
            'n_warmup_steps': 0,    # steps of warmup experience collection before training
        })
        return default_dict

    def val(self):
        """Evaluate agent."""
        # val_rollout_storage = RolloutStorage()
        val_rollout_storage = PouringSkillRolloutStorage(conf=self.conf)
        with self.agent.val_mode():
            with torch.no_grad():
                with timing("Eval rollout time: "):
                    n_eval = 100 if self.args.mode == "val" else 10
                    for i in range(n_eval):  # WandBLogger.N_LOGGED_SAMPLES # for efficiency instead of self.args.n_val_samples
                        val_rollout_storage.append(self.sampler.sample_episode(is_train=False, render=True))
                        print("{} / {} val_rollout: {}".format(i, n_eval, val_rollout_storage.get()[-1].info[-1]))

        # import cv2
        # for i in range(len(val_rollout_storage.rollouts)):
        #     print("seq: ", i)
        #     for img in val_rollout_storage.rollouts[i].image:
        #         img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB)
        #         cv2.imshow('Image', img)
        #         cv2.waitKey()
        #         print("img type / dtype: {} / {}".format(type(img), img.dtype))
        #         print("img shape: {}".format(img.shape))

        if self.args.task_name:
            val_rollout_storage.task_stats()

        rollout_stats = val_rollout_storage.rollout_stats()
        if self.is_chef:
            with timing("Eval log time: "):
                self.agent.log_outputs(rollout_stats, val_rollout_storage,
                                       self.logger, log_images=True, step=self.global_step)
            print("Evaluation Avg_Reward: {}".format(rollout_stats.avg_reward))
        del val_rollout_storage

    def warmup_skill_train(self, epoch):
        self.skill_trainer.skill_train(num_epochs=epoch)

    def skill_deployment(self):
        # load updated weights

        n_total = 0
        with self.agent.val_mode():
            with torch.no_grad():
                while True:  # keep producing rollouts until we get a valid one
                    episode = self.sampler.sample_episode(is_train=False, render=True)
                    n_total += 1
                    if n_total % 10 == 0: print("n_total: ", n_total)

    def shared_autonomy_train(self):
        self.warmup_skill_train(epoch=3)

        print("agent model weight update!")
        self.agent.update_model_weights()

    def demo(self):
        """Run task demonstration"""
        print("Task demonstration: {}".format(self.conf.notes))
        rewards = 0
        n_total = 0
        with self.agent.val_mode():
            with torch.no_grad():
                while True:  # keep producing rollouts until we get a valid one
                    episode = self.sampler.sample_episode(is_train=False, render=True)
                    n_total += 1
                    if n_total % 10 == 0: print("n_total: ", n_total)
        # print("Rewards: {:d} / {:d} = {:.3f}%".format(n_success, n_total, float(n_success) / n_total * 100))

    def warmup(self):
        """Performs pre-training warmup experience collection with random policy."""
        if self.is_chef:
            print("Warmup data collection for {} steps...".format(self._hp.n_warmup_steps))
        with self.agent.rand_act_mode():
            self.sampler.init(is_train=True)
            warmup_experience_batch, _ = self.sampler.sample_batch(
                    batch_size=int(self._hp.n_warmup_steps / self.conf.mpi.num_workers))
            if self.use_multiple_workers:
                warmup_experience_batch = mpi_gather_experience(warmup_experience_batch)
        if self.is_chef:
            self.agent.add_experience(warmup_experience_batch)
            print("...Warmup done!")

    def _get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and agent configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration
        conf.agent = conf_module.agent_config
        conf.agent.device = self.device

        # data config
        conf.data = conf_module.data_config

        # environment config
        conf.env = conf_module.env_config
        conf.env.device = self.device       # add device to env config as it directly returns tensors

        # sampler config
        conf.sampler = conf_module.sampler_config if hasattr(conf_module, 'sampler_config') else AttrDict({})

        # model loading config
        conf.ckpt_path = conf.agent.checkpt_path if 'checkpt_path' in conf.agent else None

        # load notes if there are any
        if self.args.notes != '':
            conf.notes = self.args.notes
        else:
            try:
                conf.notes = conf_module.notes
            except:
                conf.notes = ''

        # load config overwrites
        if self.args.config_override != '':
            for override in self.args.config_override.split(','):
                key_str, value_str = override.split('=')
                keys = key_str.split('.')
                curr = conf
                for key in keys[:-1]:
                    curr = curr[key]
                curr[keys[-1]] = type(curr[keys[-1]])(value_str)

        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(conf.conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))

            # setup logger
            logger = None
            # if self.args.mode == 'train':
            exp_name = f"{os.path.basename(self.args.path)}_{self.args.prefix}" if self.args.prefix \
                else os.path.basename(self.args.path)
            if self._hp.logging_target == 'wandb':
                logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                     path=self._hp.exp_path, conf=conf)
            else:
                logger = SummaryWriter(log_dir)
                # raise NotImplementedError   # TODO implement alternative logging (e.g. TB)
            return logger

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        # TODO(karl): check whether that actually loads the optimizer too
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.agent,
                                           load_step=True, strict=self.args.strict_weight_loading)
        self.agent.load_state(self._hp.exp_path)
        self.agent.to(self.device)
        return start_epoch

    def print_train_update(self, epoch, agent_outputs, timers):
        print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none',
                                  self._hp.exp_path))
        print('Train Epoch: {} [It {}/{} ({:.0f}%)]'.format(
            epoch, self.global_step, self._hp.n_steps_per_epoch * self._hp.num_epochs,
                                     100. * self.global_step / (self._hp.n_steps_per_epoch * self._hp.num_epochs)))
        print('avg time for rollout: {:.2f}s, update: {:.2f}s, logs: {:.2f}s, total: {:.2f}s'
              .format(timers['rollout'].avg, timers['update'].avg, timers['log'].avg,
                      timers['rollout'].avg + timers['update'].avg + timers['log'].avg))
        togo_train_time = timers['batch'].avg * (self._hp.num_epochs * self._hp.n_steps_per_epoch - self.global_step) \
                          / self._hp.n_steps_per_update / 3600.
        print('ETA: {:.2f}h'.format(togo_train_time))

    @property
    def log_outputs_now(self):
        return self.n_update_steps % int((self._hp.n_steps_per_epoch / self._hp.n_steps_per_update)
                                       / self._hp.log_output_per_epoch) == 0 \
                    or self.log_images_now

    @property
    def log_images_now(self):
        return self.n_update_steps % int((self._hp.n_steps_per_epoch / self._hp.n_steps_per_update)
                                       / self._hp.log_images_per_epoch) == 0

    @property
    def is_chef(self):
        return self.conf.mpi.is_chef

    @property
    def use_multiple_workers(self):
        return self.conf.mpi.num_workers > 1


if __name__ == '__main__':
    # comment out following codes if you run this script directly
    os.environ["EXP_DIR"] = "../experiments"
    os.environ["DATA_DIR"] = "../dataset"

    # with multi-GPU env, using only single GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ["block_stacking", "kitchen", "office", "maze", "pouring_water", "pouring_water_img"]
    task_name = "pouring_skill"
    mode = "spirl_cl"

    args = get_args()
    args.path = os.path.join("./", "configs", "hrl", task_name, mode)
    args.seed = 0
    args.prefix = "{}".format("SPIRL_" + task_name + "_seed0")
    args.task_name = task_name
    args.n_val_samples = 100
    # args.resume = "latest"
    args.save_root = os.environ["DATA_DIR"]  # os.path.join(os.environ["DATA_DIR"], task_name)
    SharedAutonomyTrainer(args=args)
