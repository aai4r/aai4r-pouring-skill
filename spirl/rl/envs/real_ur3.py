from spirl.rl.components.environment import BaseEnvironment
from spirl.utility.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset

from vr_teleop.tasks.real_ur3_robotiq85 import BaseRTDE, UR3ControlMode
from vr_teleop.tasks.base import VRWrapper

import numpy as np


data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    state_dim=20,
    n_actions=9,
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
        joint = self.get_actual_q()
        tcp_pos, tcp_aa = self.get_actual_tcp_pos_ori()
        tcp_quat = self.quat_from_tcp_axis_angle(tcp_aa, tolist=True)
        # TODO, target_diff is temporal state...
        target_diff = (np.array([0.5196, -0.1044, 0.088]) - np.array(tcp_pos)).tolist()
        g_one_hot = self.grip_one_hot_state()
        cm_one_hot = self.cont_mode_one_hot_state()
        obs = joint + tcp_pos + tcp_quat + target_diff + g_one_hot + cm_one_hot
        return np.array(obs) if np_type else obs

    @staticmethod
    def arg_max_one_hot(list1d):
        one_hot = [0] * len(list1d)
        arg_max = max(range(len(list1d)), key=lambda i: list1d[i])
        one_hot[arg_max] = 1
        return one_hot

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
                act_pos, act_quat, grip = action[:3], action[3:7], action[7:]
                grip_onehot = self.arg_max_one_hot(list1d=grip)
                self.move_grip_on_off(self.grip_onehot_to_bool(grip_onehot))

                actual_tcp_pos, actual_tcp_ori = self.get_actual_tcp_pos_ori()
                actual_q = self.get_actual_q()
                des_pos = np.array(actual_tcp_pos) + np.array(act_pos)
                des_rot = self.goal_axis_angle_from_act_quat(act_quat=act_quat, actual_tcp_aa=actual_tcp_ori)

                goal_pose = self.goal_pose(des_pos=des_pos, des_rot=des_rot)
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
        _pose = self.add_noise_angle(inputs=self.iposes)
        self.move_j(_pose.tolist())
        self.move_grip_on_off(grip_action=False)
        self.set_rpy_base(self.get_actual_tcp_pose())
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
