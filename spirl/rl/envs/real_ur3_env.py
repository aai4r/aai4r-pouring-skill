import copy

from spirl.rl.components.environment import BaseEnvironment
from spirl.utility.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset

from vr_teleop.tasks.real_ur3_robotiq85 import BaseRTDE, UR3ControlMode
from vr_teleop.tasks.base import VRWrapper
from vr_teleop.tasks.lib_modules import RealSense

from utils.utils import quaternion_real_last, quaternion_real_first, get_euler_xyz
from utils.torch_jit_utils import quat_mul, quat_conjugate
from pytorch3d import transforms as tr
from vr_teleop.tasks.rollout_manager import RolloutManager, RolloutManagerExpand, RobotState, RobotState2
import torch
import numpy as np


data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    state_dim=16,
    n_actions=8,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    env_name="pouring_skill_img",
    res=150,
    crop_rand_subseq=True,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RtdeUR3(BaseRTDE, UR3ControlMode):
    def __init__(self):
        self.init_vr()  # TODO, for safe test..
        BaseRTDE.__init__(self, HOST="192.168.0.75")
        UR3ControlMode.__init__(self, init_mode="forward")

        # shared autonomy control params
        self.user_control_authority = False
        self.rollout = RolloutManager(task_name="pouring_skill")
        self.collect_demo = True

        # using VR and its trigger for safety
        self.num_states = data_spec.state_dim
        self.num_acts = data_spec.n_actions

    def init_vr(self):
        self.vr = VRWrapper(device="cpu", rot_d=(-89.9, 0.0, 89.9))

    def get_robot_state(self):
        tcp_pos, tcp_aa = self.get_actual_tcp_pos_ori()
        target_diff = np.array([0.5196, -0.1044, 0.088]) - np.array(tcp_pos)
        state = RobotState2(joint=self.get_actual_q(),
                            ee_pos=tcp_pos,
                            ee_quat=self.quat_from_tcp_axis_angle(tcp_aa),
                            target_diff=target_diff.tolist(),
                            # gripper_one_hot=self.grip_one_hot_state(),
                            gripper_pos=[self.gripper.gripper_to_mm_normalize()],
                            control_mode_one_hot=self.cont_mode_one_hot_state())
        return state

    def record_frame(self, state, action_pos, action_quat, action_grip, done):
        """
        :param state:
        :param action_pos:      Relative positional diff. (vel), list type
        :param action_quat:     Rotation as a quaternion (real-last), list type
        :param action_grip:     Gripper ON/OFF one-hot vector, list type
        :param done:            Scalar value of 0 or 1
        :return:
        """
        info = str({"gripper": self.grip_on, "control_mode": self.CONTROL_MODE})
        action = action_pos + action_quat + action_grip
        self.rollout.append(state=state, action=action, done=done, info=info)

    def get_obs(self, np_type=True):
        joint = self.get_actual_q()
        tcp_pos, tcp_aa = self.get_actual_tcp_pos_ori()
        tcp_quat = self.quat_from_tcp_axis_angle(tcp_aa, tolist=True)
        # TODO, target_diff is temporal state...
        # target_diff = (np.array([0.5196, -0.1044, 0.088]) - np.array(tcp_pos)).tolist()
        # g_one_hot = self.grip_one_hot_state()
        g_pos = self.grip_pos(normalize=True, list_type=True)
        cont_mode_one_hot = self.cont_mode_one_hot_state()
        obs = joint + tcp_pos + tcp_quat + g_pos + cont_mode_one_hot
        return np.array(obs) if np_type else obs

    @staticmethod
    def arg_max_one_hot(list1d):
        one_hot = [0] * len(list1d)
        arg_max = max(range(len(list1d)), key=lambda i: list1d[i])
        one_hot[arg_max] = 1
        return one_hot

    def step_sa(self, action):  # shared autonomy step
        cont_status = self.vr.get_controller_status()
        if cont_status["btn_trigger"]:
            print("VR Trigger On!")
            self.user_control_authority = True
            self.speed_stop()

        while self.user_control_authority:
            start_t = self.init_period()
            cont_status = self.vr.get_controller_status()

            if cont_status["btn_reset_pose"]:
                self.speed_stop()
                self.record_frame(state=self.get_robot_state(),
                                  action_pos=[0., 0., 0.],
                                  action_quat=[0., 0., 0., 1.],
                                  action_grip=[self.gripper.gripper_to_mm_normalize()],
                                  done=1)

                self.rollout.show_rollout_summary()
                self.rollout.save_to_file()
                self.rollout.reset()

                obs = self.reset()
                reward, done, info = 0, True, ""
                return obs, reward, done, info

            diff_j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if cont_status["btn_trigger"]:
                state = self.get_robot_state()
                # if cont_status["btn_gripper"]:
                #     self.move_grip_on_off_toggle()

                if cont_status["btn_grip"]:
                    self.gripper.grasping_by_hold(step=-10.0)
                else:
                    self.gripper.grasping_by_hold(step=10.0)
                gripper_action_norm = self.gripper.get_gripper_action(normalize=True)

                vr_curr_pos_vel, vr_curr_quat = cont_status["lin_vel"], cont_status["pose_quat"]
                act_pos = list(vr_curr_pos_vel)

                actual_tcp_pos, actual_tcp_ori_aa = self.get_actual_tcp_pos_ori()

                actual_tcp_ori_quat = tr.axis_angle_to_quaternion(torch.tensor(actual_tcp_ori_aa))
                actual_tcp_ori_quat = quaternion_real_last(actual_tcp_ori_quat)
                conj = quat_conjugate(actual_tcp_ori_quat)
                act_quat = quat_mul(torch.tensor(vr_curr_quat), conj)

                if self.collect_demo:
                    self.record_frame(state=state,
                                      action_pos=act_pos,
                                      action_quat=act_quat.tolist(),
                                      action_grip=[gripper_action_norm],
                                      done=0)

                d_pos = np.array(actual_tcp_pos) + np.array(act_pos)
                d_rot = self.goal_axis_angle_from_act_quat(act_quat=act_quat, actual_tcp_aa=actual_tcp_ori_aa)
                goal_pose = self.goal_pose(des_pos=d_pos, des_rot=d_rot)  # limit handling

                goal_j = self.get_inverse_kinematics(tcp_pose=goal_pose)
                diff_j = (np.array(goal_j) - np.array(self.get_actual_q())) * 1.0
            self.speed_j(list(diff_j), self.acc, self.dt)
            self.wait_period(start_t)

        return self._step(action=action)

    def step(self, action):     # trigger based step
        print("action: ", action)

        while True:
            cont_status = self.vr.get_controller_status()
            if cont_status["btn_reset_pose"]:
                obs = self.reset()
                reward, done, info = 0, True, ""
                return obs, reward, done, info

            if cont_status["btn_trigger"]:
                print("VR trigger on!")
                start_t = self.init_period()
                goal_pose = self.goal_pose_from_action(action=action)
                goal_q = self.get_inverse_kinematics(tcp_pose=goal_pose)
                diff_q = (np.array(goal_q) - np.array(self.get_actual_q())) * 1.0
                self.speed_j(list(diff_q), self.acc, self.dt)
                self.wait_period(start_t)
                break
            self.speed_stop()

        obs = self.get_obs()
        reward = 0
        done = False
        info = ""
        return obs, reward, done, info

    def _step(self, action):    # natural step
        start_t = self.init_period()
        goal_pose = self.goal_pose_from_action(action=action)
        goal_q = self.get_inverse_kinematics(tcp_pose=goal_pose)
        diff_q = (np.array(goal_q) - np.array(self.get_actual_q())) * 1.0
        self.speed_j(list(diff_q), self.acc, self.dt)
        self.wait_period(start_t)

        obs = self.get_obs()    # next state / observation
        reward = 0
        done = False
        info = ""
        return obs, reward, done, info

    def goal_pose_from_action(self, action):
        act_pos, act_quat, grip = action[:3], action[3:7], action[7:]

        if len(grip) == 2:
            grip_onehot = self.arg_max_one_hot(list1d=grip)
            print("grip: {}, grip_onehot: {}".format(grip, grip_onehot))
            self.move_grip_on_off(self.grip_onehot_to_bool(grip_onehot))
        elif len(grip) == 1:
            self.gripper.rq_move_mm_norm(grip[0] * 1.0)
        else:
            raise NotImplementedError

        actual_tcp_pos, actual_tcp_ori = self.get_actual_tcp_pos_ori()
        des_pos = np.array(actual_tcp_pos) + np.array(act_pos)
        des_rot = self.goal_axis_angle_from_act_quat(act_quat=act_quat, actual_tcp_aa=actual_tcp_ori)

        return self.goal_pose(des_pos=des_pos, des_rot=des_rot)

    def reset(self):
        self.speed_stop()
        _pose = self.add_noise_angle(inputs=self.iposes)
        self.move_j(_pose.tolist())
        # self.move_grip_on_off(grip_action=False)
        self.gripper.rq_move_mm_norm(1.)
        print("[Reset] current CONT MODE: ", self.CONTROL_MODE, self.rpy_base)

        self.user_control_authority = False
        return self.get_obs()

    def render(self):
        raise NotImplementedError


from vr_teleop.tasks.lib_modules import visualize
import cv2


class ImageRtdeUR3(RtdeUR3):
    def __init__(self):
        super().__init__()
        self.cam = RealSense()
        self.config = AttrDict(crop_h=460, crop_w=460, resize_h=150, resize_w=150)
        self.rollout = RolloutManagerExpand(task_name="pouring_skill_img")    # TODO, task_name param

    def get_robot_state(self):
        tcp_pos, tcp_aa = self.get_actual_tcp_pos_ori()
        state = RobotState2(joint=self.get_actual_q(),
                            ee_pos=tcp_pos,
                            ee_quat=self.quat_from_tcp_axis_angle(tcp_aa),
                            # gripper_one_hot=self.grip_one_hot_state(),
                            gripper_pos=[self.gripper.gripper_to_mm_normalize()],
                            control_mode_one_hot=self.cont_mode_one_hot_state())
        return state

    def record_frame(self, image, state, action_pos, action_quat, action_grip, done):
        info = str({"gripper": self.grip_on, "control_mode": self.CONTROL_MODE})
        action = action_pos + action_quat + action_grip
        self.rollout.append(image=image, state=state, action=action, done=done, info=info)

    def pre_processing(self, color_image):
        """
        * crop and resize
        :param color_image:
        :return:
        """
        ih, iw = color_image.shape[:2]
        crop_h = crop_w = min(iw, ih)
        resize_h, resize_w = self.config.resize_h, self.config.resize_w

        y, x = (np.random.rand(2) * np.array([ih - crop_h, iw - crop_w])).astype(np.int16)

        cropped_img = color_image[y:y + crop_h, x:x + crop_w]
        resized_img = cv2.resize(cropped_img, dsize=(resize_h, resize_w), interpolation=cv2.INTER_AREA)
        out = resized_img
        return out

    def render(self, mode='rgb_array'):
        depth, color = self.cam.get_np_images()

        color = self.pre_processing(color)
        visualize(depth_image=depth, color_image=color, disp_name="RealSense D435")
        color = (color / 255.0).astype(np.float32)
        return color

    def step_sa(self, action):  # shared autonomy step
        cont_status = self.vr.get_controller_status()
        if cont_status["btn_trigger"]:
            print("VR Trigger On!")
            self.user_control_authority = True
            self.speed_stop()

        while self.user_control_authority:
            start_t = self.init_period()
            cont_status = self.vr.get_controller_status()

            if cont_status["btn_reset_pose"]:
                self.speed_stop()
                depth, color = self.cam.get_np_images()
                self.record_frame(image=copy.deepcopy(color),
                                  state=self.get_robot_state(),
                                  action_pos=[0., 0., 0.],
                                  action_quat=[0., 0., 0., 1.],
                                  action_grip=[self.gripper.gripper_to_mm_normalize()],
                                  done=1)

                obs = self.reset()
                reward, done, info = 0, True, ""

                self.rollout.show_rollout_summary()
                self.rollout.save_to_file()
                self.rollout.reset()
                return obs, reward, done, info

            diff_j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if cont_status["btn_trigger"]:
                depth, color = self.cam.get_np_images()
                visualize(depth_image=depth, color_image=color, disp_name="RealSense D435")
                state = self.get_robot_state()

                if cont_status["btn_grip"]:
                    self.gripper.grasping_by_hold(step=-10.0)
                else:
                    self.gripper.grasping_by_hold(step=10.0)
                gripper_action_norm = self.gripper.get_gripper_action(normalize=True)

                vr_curr_pos_vel, vr_curr_quat = cont_status["lin_vel"], cont_status["pose_quat"]
                act_pos = list(vr_curr_pos_vel)

                actual_tcp_pos, actual_tcp_ori_aa = self.get_actual_tcp_pos_ori()

                actual_tcp_ori_quat = tr.axis_angle_to_quaternion(torch.tensor(actual_tcp_ori_aa))
                actual_tcp_ori_quat = quaternion_real_last(actual_tcp_ori_quat)
                conj = quat_conjugate(actual_tcp_ori_quat)
                act_quat = quat_mul(torch.tensor(vr_curr_quat), conj)

                if self.collect_demo:
                    self.record_frame(image=copy.deepcopy(color),
                                      state=state,
                                      action_pos=act_pos,
                                      action_quat=act_quat.tolist(),
                                      action_grip=[gripper_action_norm],
                                      done=0)

                d_pos = np.array(actual_tcp_pos) + np.array(act_pos)
                d_rot = self.goal_axis_angle_from_act_quat(act_quat=act_quat, actual_tcp_aa=actual_tcp_ori_aa)
                goal_pose = self.goal_pose(des_pos=d_pos, des_rot=d_rot)  # limit handling

                goal_j = self.get_inverse_kinematics(tcp_pose=goal_pose)
                diff_j = (np.array(goal_j) - np.array(self.get_actual_q())) * 1.0
            self.speed_j(list(diff_j), self.acc, self.dt)
            self.wait_period(start_t)

        return self._step(action=action)


class RealUR3Env(BaseEnvironment):
    def __init__(self, config):
        self.config = config
        self._env = ImageRtdeUR3() if self.config.image_observation else RtdeUR3()

    def _default_hparams(self):
        pass

    def step(self, action):
        # obs, reward, done, info = self._env.step(action)
        obs, reward, done, info = self._env.step_sa(action)
        return obs, reward, done, info

    def reset(self):
        return self._env.reset()

    def render(self, mode='rgb_array'):
        return self._env.render()

    def _postprocess_info(self):
        pass
