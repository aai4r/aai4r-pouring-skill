from vr_teleop.tasks.lib_modules import noisy
from vr_teleop.tasks.lib_modules import visualize

import os
import time

import h5py
import random
import numpy as np

import dataclasses
from dataclasses import dataclass

import torch.cuda

from dataset.rollout_dataset import BatchRolloutFolder, get_ordered_file_list
from spirl.utility.general_utils import AttrDict

"""
written by twkim
Rollout / demonstration dataset management

"""


@dataclass
class RobotState:
    joint: list = None
    ee_pos: list = None
    ee_quat: list = None
    target_diff: list = None
    gripper_one_hot: list = None
    control_mode_one_hot: list = None

    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    def item_vec(self):
        return self.joint + self.ee_pos + self.ee_quat + self.target_diff + \
               self.gripper_one_hot + self.control_mode_one_hot

    def import_state_from(self, np_state1d):
        assert type(np_state1d) == np.ndarray
        self.joint = np_state1d[:6].tolist()
        self.ee_pos = np_state1d[6:9].tolist()
        self.ee_quat = np_state1d[9:13].tolist()
        self.target_diff = np_state1d[13:16].tolist()   # TODO, temp target position difference!!
        self.gripper_one_hot = np_state1d[16:18].tolist()
        self.control_mode_one_hot = np_state1d[18:20].tolist()

    @staticmethod
    def random_data(n_joint, n_cont_mode):
        _joint = [random.randint(-100, 100) / 200.0 for _ in range(n_joint)]
        _ee_pos = [random.randint(-100, 100) / 200.0 for _ in range(3)]
        _ee_quat = [random.randint(-100, 100) / 200.0 for _ in range(4)]
        grip_on = random.randint(-100, 100) > 0
        _gripper = [int(grip_on), int(not grip_on)]
        _control_mode = [0] * n_cont_mode
        _control_mode[random.randint(0, n_cont_mode - 1)] = 1
        return _joint + _ee_pos + _ee_quat + _gripper + _control_mode


@dataclass
class RobotState2(RobotState):
    gripper_pos: list = None  # 1-dim

    def item_vec(self):
        return self.joint + self.ee_pos + self.ee_quat + self.gripper_pos + self.control_mode_one_hot

    def import_state_from(self, np_state1d):
        assert type(np_state1d) == np.ndarray
        self.joint = np_state1d[:6].tolist()
        self.ee_pos = np_state1d[6:9].tolist()
        self.ee_quat = np_state1d[9:13].tolist()
        self.gripper_pos = np_state1d[13:14].tolist()
        self.control_mode_one_hot = np_state1d[14:16].tolist()

    def gen_random_data(self, n_joint, n_cont_mode):
        _joint = [random.randint(-100, 100) / 200.0 for _ in range(n_joint)]
        _ee_pos = [random.randint(-100, 100) / 200.0 for _ in range(3)]
        _ee_quat = [random.randint(-100, 100) / 200.0 for _ in range(4)]
        _gripper = [random.randint(0, 100) / 100.0]
        _control_mode = [0] * n_cont_mode
        _control_mode[random.randint(0, n_cont_mode - 1)] = 1
        self.joint = _joint
        self.ee_pos = _ee_pos
        self.ee_quat = _ee_quat
        self.gripper_pos = _gripper
        self.control_mode_one_hot = _control_mode


class RolloutManager(BatchRolloutFolder):
    """
    RolloutManager is only for state based skill learning.
    """
    def __init__(self, task_name, root_dir=None, task_desc=""):
        super().__init__(task_name=task_name, root_dir=root_dir)
        self.robot_state_class = RobotState2
        self._states = []    # joint, gripper, etc.
        self._actions = []
        self._dones = []
        self._info = []

        self.attr_list = ['state', 'action', 'done', 'info', 'pad_mask']
        self.episode_count = 0

    def isempty(self):
        return not (bool(self._states) and bool(self._actions) and bool(self._dones) and bool(self._info))

    def append(self, state, action, done, info):
        assert type(state) is self.robot_state_class
        self._states.append(state)
        self._actions.append(action)
        self._dones.append(done)
        self._info.append(info)

    def get(self, index):
        assert 0 <= index < self.len()
        return None, self._states[index], self._actions[index], self._dones[index], self._info[index]

    def len(self):
        assert len(self._states) == len(self._actions) == len(self._dones) # == len(self._info)
        return len(self._states)

    def reset(self):
        self._states = []
        self._actions = []
        self._dones = []
        self._info = []

    def to_np_rollout(self):
        np_rollout = AttrDict()
        _st = []
        [_st.append(d.item_vec()) for d in self._states]
        np_rollout.states = np.array(_st)
        np_rollout.actions = np.array(self._actions)
        np_rollout.dones = np.array(self._dones)
        np_rollout.info = self._info    # raw data (dict str)
        return np_rollout

    def show_rollout_summary(self):
        print("====================================")
        print("Current rollout dataset info.")
        print("Rollout length: ", self.len())
        idx = 0
        sample_state, sample_action, sample_done, sample_info = self.get(idx)

        print("* STEP: [{}]".format(idx))
        print("    state * {} dim with {}".format(sum([len(i) if i is not None else 0 for i in sample_state]), sample_state))
        print("    action * {} dim with {}".format(len(sample_action), sample_action))
        print("    done * {} dim with {}".format(len([sample_done]), sample_done))
        print("    info: ", sample_info)

    def save_to_file(self):
        np_episode_dict = self.to_np_rollout()
        save_path = self.get_final_save_path(self.batch_index)

        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        traj = f.create_group("traj0")
        traj.create_dataset("states", data=np_episode_dict.states)
        traj.create_dataset("actions", data=np_episode_dict.actions)
        traj.create_dataset("info", data=np_episode_dict.info)

        terminals = np_episode_dict.dones
        if np.sum(terminals) == 0: terminals[-1] = True
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj.create_dataset("pad_mask", data=pad_mask)
        f.close()
        print("save to ", save_path)

    def load_from_file(self, batch_idx, rollout_idx):
        self.reset()
        load_path = self.get_final_load_path(batch_index=batch_idx, rollout_num=rollout_idx)
        with h5py.File(load_path, 'r') as f:
            key = 'traj{}'.format(0)
            print("f: ", f[key])
            for name in f[key].keys():
                if name == 'states':
                    temp = f[key + '/' + name][()].astype(np.float32)
                    for i in range(len(temp)):
                        robot_state = self.robot_state_class()
                        robot_state.import_state_from(np_state1d=temp[i])
                        self._states.append(robot_state)
                    # self._states = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'actions':
                    self._actions = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'pad_mask':
                    self._dones = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'info':
                    temp = f[key + '/' + name][()]
                    self._info = f[key + '/' + name][()]
                else:
                    raise ValueError("{}: Unexpected rollout element...".format(name))
        print("Load complete!")


class RolloutManagerExpand(RolloutManager):
    """
        RolloutManagerExpand is an expanded version of RolloutManager,
        where it includes image observations (or depth image is also added later).
    """
    def __init__(self, task_name, root_dir=None, task_desc=""):
        super().__init__(task_name=task_name, root_dir=root_dir, task_desc=task_desc)
        self._images = []
        self._extra = []    # additional data
        self.curr_load_path = ""

    def isempty(self):
        return not (bool(self._images) and bool(self._states) and bool(self._actions)
                    and bool(self._dones) and bool(self._info) and bool(self._extra))

    def append(self, image, state, action, done, info, extra=None):
        super().append(state, action, done, info)
        self._images.append(image)
        self._extra.append(extra)

    def get(self, index):
        assert 0 <= index < self.len()
        return self._images[index], self._states[index], self._actions[index], self._dones[index], self._info[index]

    def get_extra(self, index):
        assert 0 <= index < self.len()
        return self._extra[index]

    def len(self):
        assert len(self._images) == len(self._states) == len(self._actions) == len(self._dones) == len(self._extra)
        return len(self._states)

    def reset(self):
        super().reset()
        self._images = []
        self._extra = []

    def to_np_rollout(self):
        np_rollout = super().to_np_rollout()
        c = None
        for obs in self._images:
            c = np.expand_dims(obs, axis=0) if c is None else np.concatenate((c, np.expand_dims(obs, axis=0)), axis=0)
        np_rollout.images = c
        np_rollout.extra = np.array(self._extra)
        return np_rollout

    def show_rollout_summary(self):
        print("====================================")
        print("Current rollout dataset info.")
        print("Current Load path: ", self.curr_load_path)
        print("Rollout length: ", self.len())
        idx = 0
        sample_obs, sample_state, sample_action, sample_done, sample_info = self.get(idx)
        sample_extra = self.get_extra(idx)

        print("* STEP: [{}]".format(idx))
        print("    observation shape {}".format(sample_obs.shape))
        print("    state * {} dim with {}".format(sum([len(i) if i is not None else 0 for i in sample_state]), sample_state))
        print("    action * {} dim with {}".format(len(sample_action), sample_action))
        print("    done * {} dim with {}".format(len([sample_done]), sample_done))
        print("    info: ", sample_info)
        print("    extra * {} dim with {}".format(len([sample_extra]), sample_extra))

    def save_to_file(self):
        np_episode_dict = self.to_np_rollout()
        save_path = self.get_final_save_path(self.batch_index)

        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        traj = f.create_group("traj0")
        traj.create_dataset("images", data=np_episode_dict.images)
        traj.create_dataset("states", data=np_episode_dict.states)
        traj.create_dataset("actions", data=np_episode_dict.actions)
        traj.create_dataset("info", data=np_episode_dict.info)
        traj.create_dataset("extra", data=np_episode_dict.extra)

        terminals = np_episode_dict.dones
        if np.sum(terminals) == 0: terminals[-1] = True
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj.create_dataset("pad_mask", data=pad_mask)
        f.close()
        print("save to ", save_path)

    def load_from_file(self, batch_idx, rollout_idx):
        self.reset()
        load_path = self.get_final_load_path(batch_index=batch_idx, rollout_num=rollout_idx)
        self.curr_load_path = load_path
        with h5py.File(load_path, 'r') as f:
            key = 'traj{}'.format(0)
            print("f: ", f[key])
            for name in f[key].keys():
                if name == 'images':
                    # TODO, uint8 --> float32
                    img = f[key + '/' + name][()].astype(np.uint8)
                    print("img shape: ", img.shape)
                    for i in range(len(img)):
                        # img = f[key + '/' + name][()].astype(np.float32)
                        # img /= 255.0
                        self._images.append(img[i])
                elif name == 'states':
                    temp = f[key + '/' + name][()].astype(np.float32)
                    for i in range(len(temp)):
                        robot_state = self.robot_state_class()
                        robot_state.import_state_from(np_state1d=temp[i])
                        self._states.append(robot_state)
                    # self._states = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'actions':
                    self._actions = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'pad_mask':
                    self._dones = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'info':
                    # temp = f[key + '/' + name][()]
                    self._info = f[key + '/' + name][()]
                elif name == 'extra':
                    self._extra = f[key + '/' + name][()].astype(np.float32).tolist()
                else:
                    raise ValueError("{}: Unexpected rollout element...".format(name))
        print("Load complete!")


from torch import nn
from torchvision import models, transforms
import cv2


class VideoDatasetCompressor(RolloutManagerExpand):
    def __init__(self, task_name, root_dir=None):
        super().__init__(task_name=task_name, root_dir=root_dir)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        modules = models.resnet18(pretrained=True).to(self.device)
        modules = list(modules.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False
        self.resnet18.eval()
        self._features = []
        self.aux_tag = '_f'

        self.tr = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

        self.config = AttrDict(crop_h=460, crop_w=460, resize_h=224, resize_w=224)

    def pre_processing(self, color_image):
        """
        In:
        1) random crop
        2) resize
        3) photometric distortion
            - zoom
            - brightness
            - noises(gaussian, salt-pepper, poisson, speckle)
            - candidates(affine transformation, rotation, )
        :param resize_w:
        :param resize_h:
        :param crop_w: crop width size
        :param crop_h: crop height size
        :param color_image:
        :return:
        """
        ih, iw = color_image.shape[:2]
        crop_h, crop_w = self.config.crop_h, self.config.crop_w
        resize_h, resize_w = self.config.resize_h, self.config.resize_w
        y, x = (np.random.rand(2) * np.array([ih - crop_h, iw - crop_w])).astype(np.int16)

        zoom_pix = 50
        zoom = np.random.randint(0, zoom_pix)
        cropped_img = color_image[y:y+crop_h-zoom, x:x+crop_w-zoom]
        resized_img = cv2.resize(cropped_img, dsize=(resize_h, resize_w), interpolation=cv2.INTER_AREA)

        brightness = 50
        resized_img = cv2.convertScaleAbs(resized_img, resized_img, 1, np.random.randint(-brightness, brightness))
        noisy_img = noisy(image=resized_img, noise_type='s&p', random_noise=False)
        out = noisy_img
        return out

    @staticmethod
    def np_img_to_tensor(np_img, device):
        """
        In: np_img(height, width, channel), [0, 255], uint8
        :param device: cpu, cuda:0, etc.
        :param np_img:
        :return: tensor(channel, height, width), [0, 1], float32
        """
        assert len(np_img.shape) == 3
        img_tensor = torch.tensor(np_img.transpose(2, 0, 1), device=device, dtype=torch.float32)
        return img_tensor / 255.0 if np_img.dtype == np.uint8 else img_tensor

    @staticmethod
    def tensor_img_to_np(tensor_img):
        """
        In: tensor_img(channel, height, width), [0, 1], float32
        :param tensor_img:
        :return: np(height, width, channel), [0, 255], uint8
        """
        assert ((tensor_img.dtype == torch.float16) or
                (tensor_img.dtype == torch.float32) or
                tensor_img.dtype == torch.float64) and len(tensor_img.shape) == 3
        np_img = tensor_img.cpu().numpy().transpose(1, 2, 0)
        return (np_img * 255.0).astype(np.uint8)

    def to_np_rollout(self):
        if not self._features: raise IndexError("Empty features...")
        np_rollout = RolloutManager.to_np_rollout(self)
        np_rollout.features = np.array(self._features)
        return np_rollout

    def reset(self):
        super().reset()
        self._features = []

    def compressed(self):
        """
        Compress each of image observation to feature vector by CNN-like models
        and save it to file
        :param batch_idx:
        :param rollout_idx:
        :return:
        """
        obs_stack_tensor = torch.zeros(self.len(), 3, self.config.resize_h, self.config.resize_w, device=self.device)
        for i in range(self.len()):
            obs, state, action, done, info = self.get(i)
            obs_np = self.pre_processing(obs)
            obs_tensor = self.np_img_to_tensor(np_img=obs_np, device=self.device).unsqueeze(0)
            obs_stack_tensor[i] = obs_tensor
            if visualize(depth_image=np.zeros(obs_np.shape), color_image=obs_np, delay=1) == 27:
                break
        obs_stack_tensor = self.tr(obs_stack_tensor)

        # # check the visual result of tr
        # for i in range(len(obs_stack_tensor)):
        #     obs_tensor = obs_stack_tensor[i]
        #     obs_np = self.tensor_img_to_np(obs_tensor)
        #     if self.visualize(depth_image=np.zeros(obs_np.shape), color_image=obs_np, delay=0) == 27:
        #         break
        f = self.resnet18(obs_stack_tensor)
        f = f.view(len(f), -1)
        return f.cpu().numpy()

    def featurization(self, batch_idx, rollout_idx, n_augments=3):
        self.load_from_file(batch_idx=batch_idx, rollout_idx=rollout_idx)
        for i in range(n_augments):
            features = self.compressed()
            self._features = features.tolist()
            self.save_to_file_f()
        self.reset()

    def featurization_all(self, n_augments=3):
        batches = self.get_batch_folders()
        print(batches)
        for b in batches:
            b_idx = int(b[5:])
            rollout_list = self.get_rollout_list(b_idx)
            for r in rollout_list:
                r_idx = r[len('rollout_'):r.find('.')]
                self.featurization(batch_idx=b_idx, rollout_idx=r_idx, n_augments=n_augments)

    def get_final_save_path_f(self, batch_index):
        batch_dir = self.batch_name + "{}".format(batch_index)
        task_dir = os.path.join(self.root_dir, self.task_name + self.aux_tag)
        task_batch_dir = os.path.join(task_dir, batch_dir)
        if not os.path.exists(task_batch_dir): os.makedirs(task_batch_dir)

        rollout_list = get_ordered_file_list(path=task_batch_dir, included_ext=['h5'])
        next_idx = (lambda x: int(x[x.find('_') + 1:x.find('.')]))(rollout_list[-1]) + 1 if len(rollout_list) > 0 else 0
        save_path = os.path.join(task_batch_dir, "rollout_{}.h5".format(next_idx))
        return save_path

    def get_final_load_path_f(self, batch_idx, rollout_num):
        batch_dir = self.batch_name + "{}".format(batch_idx)
        task_batch_dir = os.path.join(self.root_dir, self.task_name + self.aux_tag, batch_dir)
        if not os.path.exists(task_batch_dir):
            raise OSError("{} not exists".format(task_batch_dir))
        load_path = os.path.join(task_batch_dir, "rollout_{}.h5".format(rollout_num))
        return load_path

    def save_to_file_f(self):
        np_episode_dict = self.to_np_rollout()
        save_path = self.get_final_save_path_f(batch_index=self.batch_index)

        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        traj = f.create_group("traj0")
        traj.create_dataset("features", data=np_episode_dict.features)
        traj.create_dataset("states", data=np_episode_dict.states)
        traj.create_dataset("actions", data=np_episode_dict.actions)
        traj.create_dataset("info", data=np_episode_dict.info)

        terminals = np_episode_dict.dones
        if np.sum(terminals) == 0: terminals[-1] = True
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj.create_dataset("pad_mask", data=pad_mask)
        f.close()
        print("save to ", save_path)

    def load_from_file_f(self, batch_idx, rollout_idx):
        self.reset()
        load_path = self.get_final_load_path_f(batch_idx=batch_idx, rollout_num=rollout_idx)
        with h5py.File(load_path, 'r') as f:
            key = 'traj{}'.format(0)
            for name in f[key].keys():
                if name == 'features':
                    self._features = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'states':
                    temp = f[key + '/' + name][()].astype(np.float32)
                    for i in range(len(temp)):
                        robot_state = self.robot_state_class()
                        robot_state.import_state_from(np_state1d=temp[i])
                        self._states.append(robot_state)
                    # self._states = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'actions':
                    self._actions = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'pad_mask':
                    self._dones = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'info':
                    self._info = f[key + '/' + name][()]
                else:
                    raise ValueError("{}: Unexpected rollout element...".format(name))
        print("Load complete!")


if __name__ == "__main__":
    task = "pouring_skill_img"
    # roll = RolloutManagerExpand(task_name=task)
    # rollouts = roll.get_rollout_list(batch_idx=2)
    # for i in range(len(rollouts)):
    #     roll.load_from_file(batch_idx=2, rollout_idx=i)
    #     np_rollout = roll.to_np_rollout()
    #     print(i, np_rollout.images.shape)
    #     np_rollout.dones[-1] = 1.
    #     print("dones: ", np_rollout.dones)
    #     # visualize(depth_image=np.zeros(np_rollout.images[0].shape), color_image=np_rollout.images[0], delay=0)
    #     roll.save_to_file()
    # print(np_rollout.observations.shape, np_rollout.observations)
    # exit()

    vdc = VideoDatasetCompressor(task_name=task)
    batches = vdc.get_batch_folders()
    # print(batches)
    # for b in batches:
    #     b_idx = int(b[5:])
    #     rollout_list = vdc.get_rollout_list(b_idx)
    #     for r in rollout_list:
    #         r_idx = r[len('rollout_'):r.find('.')]
    #         print(rollout_list)

    vdc.load_from_file(batch_idx=1, rollout_idx=43)
    for i in range(vdc.len()):
        obs, state, action, done, info = vdc.get(i)
        visualize(depth_image=np.zeros(obs.shape), color_image=obs)
    # np_rollout = vdc.to_np_rollout()
    # print(np_rollout.features.shape)
    # vdc.featurization_all(n_augments=1)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.torch_jit_utils import quat_apply, to_torch
from vr_teleop.tasks.lib_modules import UR3ControlMode
from utils.utilities import quaternion_real_first, quaternion_real_last
from pytorch3d import transforms as tr


class CoordViz:
    def __init__(self, elev=30, azim=-60):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.view_init(elev=elev, azim=azim)

        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.view_init(elev=elev, azim=azim)

        self.dr = 1.5   # drawing range
        self.bl = 0.5   # basis length
        self.origin = [0, 0, 0]
        self.basis_x = [self.bl, 0, 0]
        self.basis_y = [0, self.bl, 0]
        self.basis_z = [0, 0, self.bl]

        self.draw_basis()
        self.set_viz_form()

    def set_viz_form(self):
        self.ax1.set_xticks([-self.dr, 0, self.dr])
        self.ax1.set_yticks([-self.dr, 0, self.dr])
        self.ax1.set_zticks([-self.dr, 0, self.dr])
        self.ax1.set_xlabel('X-Axis'), self.ax1.set_ylabel('Y-Axis'), self.ax1.set_zlabel('Z-Axis')
        self.ax1.set_title('Source Trajectory')

        self.ax2.set_xticks([-self.dr, 0, self.dr])
        self.ax2.set_yticks([-self.dr, 0, self.dr])
        self.ax2.set_zticks([-self.dr, 0, self.dr])
        self.ax2.set_xlabel('X-Axis'), self.ax2.set_ylabel('Y-Axis'), self.ax2.set_zlabel('Z-Axis')
        self.ax2.set_title('Constrained Trajectory')

    def draw_basis(self):
        self.draw_line_left(p1=self.origin, p2=self.basis_x, color='r')
        self.draw_line_left(p1=self.origin, p2=self.basis_y, color='g')
        self.draw_line_left(p1=self.origin, p2=self.basis_z, color='b')

        self.draw_line_right(p1=self.origin, p2=self.basis_x, color='r')
        self.draw_line_right(p1=self.origin, p2=self.basis_y, color='g')
        self.draw_line_right(p1=self.origin, p2=self.basis_z, color='b')

    def draw_line_left(self, p1, p2, color='black'):
        return self.ax1.plot(xs=[p1[0], p2[0]],
                             ys=[p1[1], p2[1]],
                             zs=[p1[2], p2[2]], color=color)

    def draw_line_right(self, p1, p2, color='black'):
        return self.ax2.plot(xs=[p1[0], p2[0]],
                             ys=[p1[1], p2[1]],
                             zs=[p1[2], p2[2]], color=color)

    def refresh(self):
        self.set_viz_form()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def show(self):
        self.set_viz_form()
        plt.show()
        Axes3D.plot()


class RotCoordViz(CoordViz):
    def __init__(self, task_name, conf_mode, rot_mode):
        assert rot_mode in ['alpha', 'beta', 'gamma']
        elev, azim = conf_mode[rot_mode].elev, conf_mode[rot_mode].azim
        super().__init__(elev=elev, azim=azim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rollout = RolloutManagerExpand(task_name)
        self.rollout.load_from_file(batch_idx=1, rollout_idx=3)

    def quat_to_mat(self, q):
        _q = q if torch.is_tensor(q) else torch.tensor(q, device=self.device)
        assert len(_q.shape) == 1   # [x, y, z, w]
        px = quat_apply(_q, to_torch([1, 0, 0], device=self.device, dtype=torch.float32)).cpu().numpy()
        py = quat_apply(_q, to_torch([0, 1, 0], device=self.device, dtype=torch.float32)).cpu().numpy()
        pz = quat_apply(_q, to_torch([0, 0, 1], device=self.device, dtype=torch.float32)).cpu().numpy()
        return np.stack((px, py, pz), axis=0)

    def draw_coord_to_left(self, mat):
        px, py, pz = mat[0], mat[1], mat[2]
        self.draw_line_left(p1=self.origin, p2=px, color='r')
        self.draw_line_left(p1=self.origin, p2=py, color='g')
        self.draw_line_left(p1=self.origin, p2=pz, color='b')

    def draw_coord_to_right(self, mat):
        px, py, pz = mat[0], mat[1], mat[2]
        self.draw_line_right(p1=self.origin, p2=px, color='b')  # r
        self.draw_line_right(p1=self.origin, p2=py, color='r')  # g
        self.draw_line_right(p1=self.origin, p2=pz, color='g')  # b


class RotCoordVizRealTime(CoordViz, UR3ControlMode):
    def __init__(self, task_name, elev=30, azim=-145):
        CoordViz.__init__(self, elev=elev, azim=azim)
        UR3ControlMode.__init__(self, init_mode="forward")
        self.l1, self.l2, self.l3 = None, None, None
        self.r1, self.r2, self.r3 = None, None, None
        plt.ion()
        plt.show()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rollout = RolloutManagerExpand(task_name)

    def get_constrained_quat(self, q):
        _q = quaternion_real_first(torch.tensor(q, device="cpu", dtype=torch.float32))
        aa = tr.quaternion_to_axis_angle(_q)
        des_pose = self.goal_pose(des_pos=[0, 0, 0], des_rot=aa)
        aa = torch.tensor(des_pose[3:], device="cpu", dtype=torch.float32)
        _q = tr.axis_angle_to_quaternion(aa)
        _q = quaternion_real_last(_q)
        return _q

    def draw(self, rq, cq):
        raw_mat = self.quat_to_mat(rq)  # raw quaternion
        px, py, pz = raw_mat[0], raw_mat[1], raw_mat[2]
        if self.l1 is self.l2 is self.l3 is None:
            self.l1,  = self.draw_line_left(p1=self.origin, p2=px, color='r')
            self.l2,  = self.draw_line_left(p1=self.origin, p2=py, color='g')
            self.l3,  = self.draw_line_left(p1=self.origin, p2=pz, color='b')
        else:
            # x-axis
            vx = np.stack((self.origin, px))
            self.l1.set_data(vx[:, 0], vx[:, 1])
            self.l1.set_3d_properties(vx[:, 2])

            # y-axis
            vy = np.stack((self.origin, py))
            self.l2.set_data(vy[:, 0], vy[:, 1])
            self.l2.set_3d_properties(vy[:, 2])

            # z-axis
            vz = np.stack((self.origin, pz))
            self.l3.set_data(vz[:, 0], vz[:, 1])
            self.l3.set_3d_properties(vz[:, 2])

        cont_mat = self.quat_to_mat(cq.to(self.device))     # constrained quaternion
        _px, _py, _pz = cont_mat[0], cont_mat[1], cont_mat[2]
        if self.r1 is self.r2 is self.r3 is None:
            self.r1,  = self.draw_line_right(p1=self.origin, p2=_px, color='r')
            self.r2,  = self.draw_line_right(p1=self.origin, p2=_py, color='g')
            self.r3,  = self.draw_line_right(p1=self.origin, p2=_pz, color='b')
        else:
            # x-axis
            vx = np.stack((self.origin, _px))
            self.r1.set_data(vx[:, 0], vx[:, 1])
            self.r1.set_3d_properties(vx[:, 2])

            # y-axis
            vy = np.stack((self.origin, _py))
            self.r2.set_data(vy[:, 0], vy[:, 1])
            self.r2.set_3d_properties(vy[:, 2])

            # z-axis
            vz = np.stack((self.origin, _pz))
            self.r3.set_data(vz[:, 0], vz[:, 1])
            self.r3.set_3d_properties(vz[:, 2])

        self.refresh()

    def refresh(self):
        self.set_viz_form()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def record_frame(self, observation, state, action_pos, action_quat, action_grip, action_mode, done, extra=None):
        info = str({"gripper": self.grip_on, "control_mode": self.CONTROL_MODE})
        action = action_pos + action_quat + action_grip + action_mode
        self.rollout.append(image=observation, state=state, action=action, done=done, info=info, extra=extra)

    def quat_to_mat(self, q):
        _q = q if torch.is_tensor(q) else torch.tensor(q, device=self.device, dtype=torch.float32)
        assert len(_q.shape) == 1   # [x, y, z, w]
        px = quat_apply(_q, to_torch([1, 0, 0], device=self.device, dtype=torch.float32)).cpu().numpy()
        py = quat_apply(_q, to_torch([0, 1, 0], device=self.device, dtype=torch.float32)).cpu().numpy()
        pz = quat_apply(_q, to_torch([0, 0, 1], device=self.device, dtype=torch.float32)).cpu().numpy()
        return np.stack((px, py, pz), axis=0)

    def draw_coord_to_left(self, mat):
        px, py, pz = mat[0], mat[1], mat[2]
        self.draw_line_left(p1=self.origin, p2=px, color='r')
        self.draw_line_left(p1=self.origin, p2=py, color='g')
        self.draw_line_left(p1=self.origin, p2=pz, color='b')

    def draw_coord_to_right(self, mat):
        px, py, pz = mat[0], mat[1], mat[2]
        self.draw_line_right(p1=self.origin, p2=px, color='r')
        self.draw_line_right(p1=self.origin, p2=py, color='g')
        self.draw_line_right(p1=self.origin, p2=pz, color='b')