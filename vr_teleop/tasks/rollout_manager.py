from lib_modules import noisy
import os
import time

import h5py
import random
import numpy as np

import dataclasses
from dataclasses import dataclass

import torch.cuda

from dataset.rollout_dataset import BatchRolloutFolder
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
        self._observations = []

    def isempty(self):
        return not (bool(self._observations) and bool(self._states) and bool(self._actions)
                    and bool(self._dones) and bool(self._info))

    def append(self, observation, state, action, done, info):
        super().append(state, action, done, info)
        self._observations.append(observation)

    def get(self, index):
        assert 0 <= index < self.len()
        return self._observations[index], self._states[index], self._actions[index], self._dones[index], self._info[index]

    def len(self):
        assert len(self._observations) == len(self._states) == len(self._actions) == len(self._dones) # == len(self._info)
        return len(self._states)

    def reset(self):
        super().reset()
        self._observations = []

    def to_np_rollout(self):
        np_rollout = super().to_np_rollout()
        c = None
        for obs in self._observations:
            c = np.expand_dims(obs, axis=0) if c is None else np.concatenate((c, np.expand_dims(obs, axis=0)), axis=0)
        np_rollout.observations = c
        return np_rollout

    def show_rollout_summary(self):
        print("====================================")
        print("Current rollout dataset info.")
        print("Rollout length: ", self.len())
        idx = 0
        sample_obs, sample_state, sample_action, sample_done, sample_info = self.get(idx)

        print("* STEP: [{}]".format(idx))
        print("    observation shape {}".format(sample_obs.shape))
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
        traj.create_dataset("observations", data=np_episode_dict.observations)
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
                if name == 'observations':
                    # TODO, uint8 --> float32
                    img = f[key + '/' + name][()].astype(np.uint8)
                    print("img shape: ", img.shape)
                    for i in range(len(img)):
                        # img = f[key + '/' + name][()].astype(np.float32)
                        # img /= 255.0
                        self._observations.append(img[i])
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

        self.tr = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

        self.config = AttrDict(crop_h=460, crop_w=460, resize_h=150, resize_w=150)

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
    def visualize(depth_image, color_image, delay=1):
        if depth_image is None or color_image is None:
            print("Can't get a frame....")
            return cv2.waitKey(delay)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        return cv2.waitKey(delay)

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

    def compressed(self, batch_idx, rollout_idx):
        """
        Compress each of image observation to feature vector by CNN-like models
        and save it to file
        :param batch_idx:
        :param rollout_idx:
        :return:
        """

        self.load_from_file(batch_idx=batch_idx, rollout_idx=rollout_idx)
        obs_stack_tensor = torch.zeros(self.len(), 3, self.config.resize_h, self.config.resize_w, device=self.device)
        start = time.time()
        for i in range(self.len()):
            obs, state, action, done, info = self.get(i)
            obs_np = self.pre_processing(obs)
            obs_tensor = self.np_img_to_tensor(np_img=obs_np, device=self.device).unsqueeze(0)
            obs_stack_tensor[i] = obs_tensor
            if self.visualize(depth_image=np.zeros(obs_np.shape), color_image=obs_np, delay=1) == 27:
                break
        print("elapsed: {}".format(time.time() - start))
        print("obs_stack_tensor: ", obs_stack_tensor.shape)
        obs_stack_tensor = self.tr(obs_stack_tensor)

        # # check the visual result of tr
        # for i in range(len(obs_stack_tensor)):
        #     obs_tensor = obs_stack_tensor[i]
        #     obs_np = self.tensor_img_to_np(obs_tensor)
        #     if self.visualize(depth_image=np.zeros(obs_np.shape), color_image=obs_np, delay=0) == 27:
        #         break
        start = time.time()
        f = self.resnet18(obs_stack_tensor)
        f = f.view(len(f), -1)
        print("feature stack tensor, elapsed: {}, shape: {}, value: {}, "
              .format(time.time() - start, f.shape, f))
        return f.cpu().numpy()


if __name__ == "__main__":
    # test code for rollout file check
    task = "pouring_skill_img"
    vdc = VideoDatasetCompressor(task_name=task)
    f = vdc.compressed(batch_idx=1, rollout_idx=0)
    print("f: ", f.shape, type(f), f)
