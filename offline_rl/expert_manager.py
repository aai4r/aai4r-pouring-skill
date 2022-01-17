import torch
from gym.spaces import Space


class ExpertManager:
    def __init__(self, vec_env):
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.vec_env = vec_env


    def save(self):
        pass

    def load(self):
        pass

    def run(self, num_expert_iterations):
        current_obs = self.vec_env.reset()
        current_state = self.vec_env.get_state()

        for it in range(0, num_expert_iterations):
            print("it: ", it, self.action_space.shape)
            rand_act = torch.rand((self.vec_env.num_envs,) + self.action_space.shape)
            next_obs, rews, dones, infors = self.vec_env.step(rand_act)
            print("rand act: ", rand_act)

