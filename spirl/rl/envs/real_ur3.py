from spirl.rl.components.environment import BaseEnvironment
from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset

from vr_teleop.tasks.real_ur3_robotiq85 import BaseRTDE, UR3ControlMode
from vr_teleop.tasks.base import VRWrapper

import numpy as np


data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    state_dim=10,
    n_actions=6,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    env_name="pouring_skill",
    res=128,
    crop_rand_subseq=True,
)


class RtdeUR3(BaseRTDE, UR3ControlMode):
    def __init__(self):
        self.init_vr()  # TODO, for safe test..
        BaseRTDE.__init__(self, HOST="192.168.0.75")
        UR3ControlMode.__init__(self, init_mode="forward")

        # using VR and its trigger for safety
        self.num_states = 10
        self.num_acts = 6

    def init_vr(self):
        self.vr = VRWrapper(device="cpu", rot_d=(-89.9, 0.0, 89.9))

    def get_obs(self, np_type=True):
        q = self.get_actual_q()
        g = self.grip_one_hot_state()
        s = self.cont_mode_one_hot_state()
        obs = np.array(q + g + s) if np_type else q + g + s
        return obs

    def step(self, action):
        obs = self.get_obs()
        print("action: ", action)

        while True:
            start_t = self.init_period()
            cont_status = self.vr.get_controller_status()
            if cont_status["btn_reset_pose"]:
                obs = self.reset()
                break

            if cont_status["btn_trigger"]:
                print("VR trigger on!")
                actual_q = self.get_actual_q()
                goal_pose = self.goal_pose(des_pos=action[:3], des_rot=action[3:])
                goal_q = self.get_inverse_kinematics(tcp_pose=goal_pose)
                diff_q = (np.array(goal_q) - np.array(actual_q)) * 0.5
                self.speed_j(list(diff_q), self.acc, self.dt)
                self.wait_period(start_t)
                break
            self.speed_stop()

        reward = 0
        done = False
        info = ""
        return obs, reward, done, info

    def reset(self):
        self.speed_stop()
        self.move_j(self.iposes)
        return self.get_obs()


class RealUR3Env(BaseEnvironment):
    def __init__(self, config):
        self.config = config
        self._env = RtdeUR3()

    def _default_hparams(self):
        pass

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self._env.reset()

    def render(self, mode='rgb_array'):
        return np.random.rand(128, 128, 3)
        # raise NotImplementedError

    def _postprocess_info(self):
        pass
