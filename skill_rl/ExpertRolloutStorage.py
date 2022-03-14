import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler

from spirl.utils.general_utils import AttrDict


class ExpertRolloutStorage:

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

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

    def info(self):
        print("***** Expert Demo Rollout Storage Information *****")
        print("[Shape: (num_trans, num_envs, dim)]")
        print("    observations: {}".format(self.observations.shape))
        print("    states:       {}".format(self.states.shape))
        print("    rewards:      {}".format(self.rewards.shape))
        print("    actions:      {}".format(self.actions.shape))
        print("    dones:        {}".format(self.dones.shape))

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()