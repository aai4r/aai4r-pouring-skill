import sys
import math
import time
import datetime

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler

from task_rl.utils.rollout_utils import RolloutSaverIsaac
from spirl.utils.general_utils import AttrDict

dtype_to_byte = {torch.float32: 4, torch.float: 4, torch.uint8: 1}


class ExpertRolloutStorage(RolloutSaverIsaac):

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, cfg, sampler='sequential'):
        super().__init__(cfg=cfg)

        self.device = cfg['device']
        self.sampler = sampler
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.DESIRED_BATCH_SIZE = cfg['expert']['desired_batch_size'] if "desired_batch_size" in cfg['expert'] \
            else 2 * (1000 * 1000 * 1000)  # 2GB, default

        self.shapes = AttrDict(observations=obs_shape, states=states_shape, actions=actions_shape,
                               rewards=(1,), dones=(1,))

        obs_dtype = torch.uint8 if len(self.shapes.observations) > 2 else torch.float32  # uint8 for image obs
        self.dtypes = AttrDict(observations=obs_dtype, states=torch.float32, actions=torch.float32,
                               rewards=torch.float32, dones=torch.float32)

        # self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape,
        #                                 device=self.device, dtype=self.dtypes.observations)
        # self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        # self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        # self.step = 0
        self.reset_rollout()

        self.rollout = AttrDict(observations=self.observations,
                                states=self.states,
                                rewards=self.rewards,
                                actions=self.actions,
                                dones=self.dones)

        self.summary = AttrDict(observations={"min": [], "max": [], "n_trans": [], "size": [], "shape": [], "dtype": []},
                                states={"min": [], "max": [], "n_trans": [], "size": [], "shape": [], "dtype": []},
                                rewards={"min": [], "max": [], "n_trans": [], "size": [], "shape": [], "dtype": []},
                                actions={"min": [], "max": [], "n_trans": [], "size": [], "shape": [], "dtype": []},
                                dones={"min": [], "max": [], "n_trans": [], "size": [], "shape": [], "dtype": []},
                                )

    def add_transitions(self, observations, states, actions, rewards, dones):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1

    def save_batch(self):
        if not self.cfg['expert']['save_data']:
            return

        # (n_trans, n_env, *) --> (n_env, n_trans, *)
        np_obs_dim = np.arange(len(self.observations.size()))[2:]
        np_observations = self.observations.permute(1, 0, *np_obs_dim).cpu().numpy()
        np_states = self.states.permute(1, 0, 2).cpu().numpy()
        np_actions = self.actions.permute(1, 0, 2).cpu().numpy()
        np_rewards = self.rewards.permute(1, 0, 2).cpu().numpy()
        np_dones = self.dones.permute(1, 0, 2).cpu().numpy()

        trim_last = []
        for i_env in range(0, len(np_dones) - 1):
            ep_idx = np.where(np_dones[i_env] > 0)[0]
            if len(ep_idx) < 1:    # skip no terminal signal
                print("Too short episodes... Adjust the total frames or num_envs.")
                continue
            print("env_num: {}".format(i_env))

            ep_idx = np.append([-1], ep_idx)
            for i_episode in range(0, len(ep_idx) - 1):
                trim_last.append(np_dones.shape[1] - ep_idx[-1])   # n_trans - last_terminal
                print("    epi_idx: {},  trimed ep: {}".format(ep_idx, trim_last[-1]))

                start = ep_idx[i_episode] + 1
                end = ep_idx[i_episode + 1]
                ep_len = end - start
                print("    start: {}, end: {},   length: {}".format(start, end, ep_len))
                if ep_len < 50:     # skip too short episode
                    print("skipped ep_len: {}".format(ep_len))
                    continue

                skip_init_frames = 3    # to remove noisy frames during initialization
                episode = AttrDict(
                    observations=np_observations[i_env, start + skip_init_frames:end],
                    states=np_states[i_env, start + skip_init_frames:end],
                    actions=np_actions[i_env, start + skip_init_frames:end],
                    rewards=np_rewards[i_env, start + skip_init_frames:end],
                    dones=np_dones[i_env, start + skip_init_frames:end]
                )
                self.save_rollout_to_file(episode, obs_to_img_key=True)
                self.collect_rollout_statistics(episode)

                total_size = sum([sum(val['size']) for _, val in self.summary.items()])
                batch_thres = (total_size + self.pre_size) / (self.DESIRED_BATCH_SIZE * self.batch_count)
                if batch_thres > 1.0:
                    print("batch up!!")
                    self.batch_count += 1
                    self.counter = 0
        print("trim last lengths: ", trim_last)

    def save(self):
        np_obs_dim = np.arange(len(self.observations.size()))[2:]
        np_observations = self.observations.permute(1, 0, *np_obs_dim).reshape(-1, *self.shapes.observations).cpu().numpy()
        np_states = self.states.permute(1, 0, 2).cpu().numpy()
        if self.states.nelement() > 0:
            np_states = np_states.reshape(-1, *self.shapes.states)
        np_actions = self.actions.permute(1, 0, 2).reshape(-1, *self.shapes.actions).cpu().numpy()
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
        self.collect_rollout_statistics(episode)

    def reset_rollout(self):
        self.observations = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.shapes.observations,
                                        device=self.device, dtype=self.dtypes.observations)
        self.states = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.shapes.states, device=self.device)
        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.actions = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.shapes.actions, device=self.device)
        self.dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device).byte()

        self.rollout = AttrDict(observations=self.observations,
                                states=self.states,
                                rewards=self.rewards,
                                actions=self.actions,
                                dones=self.dones)
        self.step = 0

    def expected_rollout_size(self, print_info=False):
        expected_size = AttrDict()

        total_size = 0
        for key, val in self.shapes.items():
            _size = np.prod(val) * self.num_envs * self.num_transitions_per_env * dtype_to_byte[self.dtypes[key]]
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
        unit_value = {"T.Byte": 1000 ** 4, "G.Byte": 1000 ** 3, "M.Byte": 1000 ** 2, "K.Byte": 1000, "Byte": 1}
        for key, val in unit_value.items():
            if input >= val:
                return round(input / val, 1), key

    def get_accumulated_size(self):
        accumulated_size = 0
        for key, val in self.summary.items():
            accumulated_size += sum(val['size'])
        return accumulated_size

    def collect_rollout_statistics(self, episode):
        for key, val in episode.items():
            if val.size <= 0: continue
            self.summary[key]['min'].append(val.min().item())
            self.summary[key]['max'].append(val.max().item())
            self.summary[key]['n_trans'].append(len(val))
            self.summary[key]['size'].append((val.itemsize * val.size))
            self.summary[key]['shape'] = [sum(self.summary[key]['n_trans'])] + list(val.shape)
            if not self.summary[key]['dtype']:
                self.summary[key]['dtype'].append(val.dtype)

    def show_rollout_info(self):
        key_max_len = len(max(self.rollout.keys(), key=len))
        shp_max_len = max(list(map(lambda x: len(str(x['shape'])), self.summary.values())))
        dtype_max_len = max(list(map(lambda x: len(str(x['dtype'])), self.summary.values())))

        print("***** Expert Demo Rollout Storage Information *****")
        print("[Shape: (num_trans, num_envs, dim)]")
        total_size = 0
        for key, val in self.rollout.items():
            if val.nelement() <= 0: continue

            print("    {}{}, shape: {}{}, min/max: {}{:.3f}  / {}{:.3f}, datatype: {}{}, size: {:,} {}".format(
                key, ''.join([' ' for _ in range(key_max_len - len(key))]),
                list(val.shape), ''.join([' ' for _ in range(shp_max_len - len(str(list(val.shape))))]),
                ''.join([' ' if val.min() >= 0 else '']), val.min(), ''.join([' ' if val.max() >= 0 else '']), val.max(),
                val.dtype, ''.join([' ' for _ in range(dtype_max_len - len(str(val.dtype)))]),
                *self.num_with_unit(val.element_size() * val.nelement())))
            total_size += val.element_size() * val.nelement()
        print("    Total Dataset Size: {:,} {}".format(*self.num_with_unit(total_size)))

    def show_summary(self, start_time):
        key_max_len = len(max(self.summary.keys(), key=len))
        shp_max_len = max(list(map(lambda x: len(str(x['shape'])), self.summary.values())))
        dtype_max_len = max(list(map(lambda x: len(str(x['dtype'])), self.summary.values())))

        print("*******************")
        print("***** Summary *****")
        print("*******************")
        total_size = 0
        for key, val in self.summary.items():
            _shape = [sum(val['n_trans'])] + list(self.shapes[key])
            _min = min(val['min'])
            _max = max(val['max'])
            _dtype = self.dtypes[key]
            _size = sum(val['size'])
            total_size += _size
            print("    {}{},  shape: {}{},  min/max: {}{:.3f} / {}{:.3f},  datatype: {}{},  total size: {:,} {}".format(
                key, ''.join([' ' for _ in range(key_max_len - len(key))]),
                _shape, ''.join([' ' for _ in range(shp_max_len - len(str(_shape)))]),
                ''.join([' ' if _min >= 0 else '']), _min, ''.join([' ' if _max >= 0 else '']), _max,
                _dtype, ''.join([' ' for _ in range(dtype_max_len - len(str(_dtype)))]),
                *self.num_with_unit(_size)
            ))
        print("    Total Dataset Size: {:,} {}".format(*self.num_with_unit(total_size)))
        print("    Total Elapsed Time for Collection: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()