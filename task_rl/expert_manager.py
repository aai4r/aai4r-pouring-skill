import os
import time
import datetime
import numpy as np
import h5py

from gym.spaces import Space
from task_rl.ExpertRolloutStorage import ExpertRolloutStorage

from task_rl.utils.rollout_utils import RolloutSaverIsaac
from spirl.utils.general_utils import AttrDict


class ExpertManager:
    def __init__(self, vec_env, num_transition_per_env, cfg, device):
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

        self.cfg = cfg
        self.task_name = self.cfg['task']['name']
        self.saver = RolloutSaverIsaac(save_dir=self.cfg['expert']['data_path'], task_name=self.task_name)

        self.device = device
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.vec_env = vec_env
        self.storage = ExpertRolloutStorage(self.vec_env.num_envs, num_transition_per_env, self.observation_space.shape,
                                            self.state_space.shape, self.action_space.shape, self.device)

    def save(self):
        np_observations = self.storage.observations.permute(1, 0, 2).reshape(-1, *self.storage.shapes.obs_shape).cpu().numpy()
        if self.storage.states.nelement() > 0:
            np_states = self.storage.states.permute(1, 0, 2).reshape(-1, *self.storage.shapes.states_shape).cpu().numpy()
        else:
            np_states = self.storage.states.cpu().numpy()
        np_actions = self.storage.actions.permute(1, 0, 2).reshape(-1, *self.storage.shapes.actions_shape).cpu().numpy()
        np_rewards = self.storage.rewards.permute(1, 0, 2).reshape(-1, 1).cpu().numpy()
        np_dones = self.storage.dones.permute(1, 0, 2).reshape(-1, 1).cpu().numpy()

        episode = AttrDict(
            observations=np_observations,
            states=np_states,
            actions=np_actions,
            rewards=np_rewards,
            dones=np_dones
        )
        self.saver.save_rollout_to_file(episode)

    def load(self):
        data_path = self.cfg['task_rl']['data_path']
        rollout_index = 0
        filename = "rollout_" + str(rollout_index) + '.h5'
        path = os.path.join(data_path, filename)
        print('path: ', path)
        with h5py.File(path, 'r') as f:
            data = AttrDict()

            key = 'traj{}'.format(0)
            # Fetch data into a dict
            for name in f[key].keys():
                if name in ['observations', 'actions', 'pad_mask']:
                    data[name] = f[key + '/' + name][()].astype(np.float32)
                    # print("{}: shape: {}, data: {}".format(name, data[name].shape, data[name]))
                    print("{}: shape: {}".format(name, data[name].shape))

    def run(self, num_transitions_per_env):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        # rollout task_rl demonstration
        start = time.time()
        for frame in range(num_transitions_per_env):
            if frame % 100 == 0:
                print("frames: {} / {}, elapsed: {}".format(
                    frame * self.vec_env.num_envs, num_transitions_per_env * self.vec_env.num_envs,
                    str(datetime.timedelta(seconds=int(time.time() - start)))))
            actions = self.vec_env.task.calc_expert_action()
            next_obs, rews, dones, infors = self.vec_env.step(actions)
            next_states = self.vec_env.get_state()
            self.storage.add_transitions(current_obs, current_states, actions, rews, dones)
            current_obs.copy_(next_obs)
            current_states.copy_(next_states)

        # posterior process like print info. save, etc.
        self.storage.info()
        if self.cfg['expert']['save_data']:
            self.save()
