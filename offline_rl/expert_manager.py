import torch
from gym.spaces import Space
from offline_rl.ExpertRolloutStorage import ExpertRolloutStorage


class ExpertManager:
    def __init__(self, vec_env, num_transition_per_env, device):
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

        self.device = device
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.vec_env = vec_env
        self.storage = ExpertRolloutStorage(self.vec_env.num_envs, num_transition_per_env, self.observation_space.shape,
                                            self.state_space.shape, self.action_space.shape, self.device)

    def save(self):
        pass

    def load(self):
        pass

    def run(self, num_transitions_per_env):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        # rollout expert demonstration
        for it in range(0, num_transitions_per_env):
            print("it: ", it)
            # actions = torch.rand((self.vec_env.num_envs,) + self.action_space.shape)
            actions = self.vec_env.task.calc_expert_action()
            next_obs, rews, dones, infors = self.vec_env.step(actions)
            next_states = self.vec_env.get_state()
            self.storage.add_transitions(current_obs, current_states, actions, rews, dones)
            current_obs.copy_(next_obs)
            current_states.copy_(next_states)