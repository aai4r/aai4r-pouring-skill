import sys
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler

from spirl.utils.general_utils import AttrDict


class ExpertRolloutStorage:

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):

        self.device = device
        self.sampler = sampler

        # Core
        obs_dtype = torch.uint8 if len(obs_shape) > 2 else torch.float32    # uint8 in case of image observation
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device, dtype=obs_dtype)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.rollout = AttrDict(observations=self.observations,
                                states=self.states,
                                rewards=self.rewards,
                                actions=self.actions,
                                dones=self.dones)

        self.expected_rollout_size(print_info=True)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.shapes = AttrDict(obs_shape=obs_shape, states_shape=states_shape, actions_shape=actions_shape)

        self.step = 0

    def add_transitions(self, observations, states, actions, rewards, dones):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1

    def clear(self):
        self.step = 0

    def expected_rollout_size(self, print_info=False):
        rollout_size = AttrDict()

        total_size = 0
        for key, val in self.rollout.items():
            if val.nelement() <= 0: continue
            _size = val.element_size() * val.nelement()
            rollout_size[key] = _size
            total_size += _size
        rollout_size.total = total_size

        if print_info:
            print("*** Rollout Memory Information ***")
            print("    Desired steps: {}".format(self.rollout.actions.shape[:2].numel()))
            print("    Expected Total Rollout Size: {:,} {}".format(*self.num_unit(rollout_size.total)))
        return rollout_size

    def num_unit(self, input):
        unit_value = {"G.Byte": 1000000000, "M.Byte": 1000000, "K.Byte": 1000, "Byte": 1}
        for key, val in unit_value.items():
            if input >= val:
                return round(input / val), key

    def info(self):
        self.expected_rollout_size()    # TODO

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
                *self.num_unit(val.element_size() * val.nelement())))
            total_size += val.element_size() * val.nelement()
        print("    Total Dataset Size: {:,} {}".format(*self.num_unit(total_size)))

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()