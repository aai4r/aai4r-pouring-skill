import sys
import math
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler

from task_rl.utils.rollout_utils import RolloutSaverIsaac
from spirl.utils.general_utils import AttrDict


class ExpertRolloutStorage(RolloutSaverIsaac):

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, cfg, sampler='sequential'):
        super().__init__(save_dir=cfg['expert']['data_path'], task_name=cfg['task']['name'])

        self.cfg = cfg
        self.device = cfg['device']
        self.sampler = sampler
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.shapes = AttrDict(obs_shape=obs_shape, states_shape=states_shape, actions_shape=actions_shape,
                               rewards_shape=(1,), done_shape=(1,))

        _rollout_size = self.expected_rollout_size(print_info=True)
        SPLIT_SIZE = 50 * (1000 * 1000)  # MB
        self._n_split = math.ceil(_rollout_size.total / SPLIT_SIZE)
        split_list = [int(num_transitions_per_env / self._n_split)] * (self._n_split - 1)
        self.split_tr_list = split_list + [num_transitions_per_env - sum(split_list)]
        self.split_count = 0

        print("num split: {}".format(self._n_split))
        print("split tr: ", self.split_tr_list)
        self.step = 0

        self.init_rollout()
        self.split_count += 1

        self.summary = AttrDict(observations={"min": [], "max": [], "n_trans": [], "size": []},
                                states={"min": [], "max": [], "n_trans": [], "size": []},
                                rewards={"min": [], "max": [], "n_trans": [], "size": []},
                                actions={"min": [], "max": [], "n_trans": [], "size": []},
                                dones={"min": [], "max": [], "n_trans": [], "size": []})

    def add_transitions(self, observations, states, actions, rewards, dones):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1

        if self.step >= len(self.observations) - 1:
            self.collect_rollout_info()
            if self.cfg['expert']['save_data']:
                self.save()
                self.show_rollout_info()
            self.reset_rollout()
            self.step = 0

    def save(self):
        np_obs_dim = np.arange(len(self.observations.size()))[2:]
        np_observations = self.observations.permute(1, 0, *np_obs_dim).reshape(-1, *self.shapes.obs_shape).cpu().numpy()
        np_states = self.states.permute(1, 0, 2).cpu().numpy()
        if self.states.nelement() > 0:
            np_states = np_states.reshape(-1, *self.shapes.states_shape)
        np_actions = self.actions.permute(1, 0, 2).reshape(-1, *self.shapes.actions_shape).cpu().numpy()
        np_rewards = self.rewards.permute(1, 0, 2).reshape(-1, 1).cpu().numpy()
        np_dones = self.dones.permute(1, 0, 2).reshape(-1, 1).cpu().numpy()

        episode = AttrDict(
            observations=np_observations,
            states=np_states,
            actions=np_actions,
            rewards=np_rewards,
            dones=np_dones
        )
        self.save_rollout_to_file(episode)

    def init_rollout(self):
        obs_dtype = torch.uint8 if len(
            self.shapes.obs_shape) > 2 else torch.float32  # uint8 in case of image observation
        num_split_trans_per_env = self.split_tr_list[self.split_count]
        self.observations = torch.zeros(num_split_trans_per_env, self.num_envs, *self.shapes.obs_shape, device=self.device, dtype=obs_dtype)
        self.states = torch.zeros(num_split_trans_per_env, self.num_envs, *self.shapes.states_shape, device=self.device)
        self.rewards = torch.zeros(num_split_trans_per_env, self.num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_split_trans_per_env, self.num_envs, *self.shapes.actions_shape, device=self.device)
        self.dones = torch.zeros(num_split_trans_per_env, self.num_envs, 1, device=self.device).byte()

        self.rollout = AttrDict(observations=self.observations,
                                states=self.states,
                                rewards=self.rewards,
                                actions=self.actions,
                                dones=self.dones)

    def reset_rollout(self):
        if self._n_split < 2:
            print("No need to split the rollout...")
            return

        if self.split_count >= self._n_split:
            print("End of Rollout Split...")
            return

        self.init_rollout()
        self.split_count += 1
        self.step = 0

    def expected_rollout_size(self, print_info=False):
        expected_size = AttrDict()

        total_size = 0
        for key, val in self.shapes.items():
            _size = np.prod(val) * self.num_envs * self.num_transitions_per_env
            expected_size[key] = _size
            total_size += _size
        expected_size.total = total_size

        if print_info:
            print("----------------------------------------------")
            print("*** Rollout Memory Information ***")
            print("    Desired steps: {} steps".format(self.num_envs * self.num_transitions_per_env))
            print("    Expected Total Rollout Size: {:,} {}".format(*self.num_with_unit(expected_size.total)))
            print("----------------------------------------------")
        return expected_size

    def num_with_unit(self, input):
        unit_value = {"G.Byte": 1000000000, "M.Byte": 1000000, "K.Byte": 1000, "Byte": 1}
        for key, val in unit_value.items():
            if input >= val:
                return round(input / val), key

    def collect_rollout_info(self):
        for key, val in self.rollout.items():
            if val.nelement() <= 0: continue
            self.summary[key]['min'].append(val.min().item())
            self.summary[key]['max'].append(val.max().item())
            self.summary[key]['n_trans'].append(len(val))
            self.summary[key]['size'].append((val.element_size() * val.nelement()))

    def show_rollout_info(self):
        key_max_len = len(max(self.rollout.keys(), key=len))
        shp_max_val = max(list(map(lambda x: len(str(list(x.shape))), self.rollout.values())))
        dtype_max_len = max(list(map(lambda x: len(str(x.dtype)), self.rollout.values())))

        print("***** Expert Demo Rollout Storage Information *****")
        print("[Shape: (num_trans, num_envs, dim)]")
        total_size = 0
        for key, val in self.rollout.items():
            if val.nelement() <= 0: continue

            print("    {}{}, shape: {}{}, min/max: {}{:.3f}  / {}{:.3f}, datatype: {}{}, size: {:,} {}".format(
                key, ''.join([' ' for _ in range(key_max_len - len(key))]),
                list(val.shape), ''.join([' ' for _ in range(shp_max_val - len(str(list(val.shape))))]),
                ''.join([' ' if val.min() >= 0 else '']), val.min(), ''.join([' ' if val.max() >= 0 else '']), val.max(),
                val.dtype, ''.join([' ' for _ in range(dtype_max_len - len(str(val.dtype)))]),
                *self.num_with_unit(val.element_size() * val.nelement())))
            total_size += val.element_size() * val.nelement()
        print("    Total Dataset Size: {:,} {}".format(*self.num_with_unit(total_size)))

    def show_summary(self):
        key_max_len = len(max(self.rollout.keys(), key=len))
        shp_max_val = max(list(map(lambda x: len(str(list(x.shape))), self.rollout.values())))
        dtype_max_len = max(list(map(lambda x: len(str(x.dtype)), self.rollout.values())))

        print("*******************")
        print("***** Summary *****")
        print("*******************")
        total_size = 0
        for key, val in self.summary.items():
            _shape = [sum(val['n_trans'])] + list(self.rollout[key].shape)[1:]
            _min = min(val['min'])
            _max = max(val['max'])
            _dtype = self.rollout[key].dtype
            _size = sum(val['size'])
            total_size += _size
            print("    {}{},  shape: {}{},  min/max: {}{:.3f} / {}{:.3f},  datatype: {}{},  total size: {:,} {}".format(
                key, ''.join([' ' for _ in range(key_max_len - len(key))]),
                _shape, ''.join([' ' for _ in range(shp_max_val - len(str(_shape)))]),
                ''.join([' ' if _min >= 0 else '']), _min, ''.join([' ' if _max >= 0 else '']), _max,
                _dtype, ''.join([' ' for _ in range(dtype_max_len - len(str(_dtype)))]),
                *self.num_with_unit(_size)
            ))
        print("    Total Dataset Size: {:,} {}".format(*self.num_with_unit(total_size)))

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()