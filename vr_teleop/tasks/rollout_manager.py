import os
import h5py
import random
import numpy as np

import dataclasses
from dataclasses import dataclass
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


class RolloutManager(BatchRolloutFolder):
    def __init__(self, task_name, root_dir=None, task_desc=""):
        super().__init__(task_name=task_name, root_dir=root_dir)
        self._states = []    # joint, gripper, etc.
        self._actions = []
        self._dones = []
        self._info = []

        self.attr_list = ['state', 'action', 'done', 'info', 'pad_mask']
        self.episode_count = 0

    def append(self, state, action, done, info):
        assert type(state) is RobotState
        self._states.append(state)
        self._actions.append(action)
        self._dones.append(done)
        self._info.append(info)

    def get(self, index):
        assert 0 <= index < self.len()
        return self._states[index], self._actions[index], self._dones[index], self._info[index]

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
        print("    state * {} dim with {}".format(sum([len(i) for i in sample_state]), sample_state))
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
                        rs = RobotState()
                        rs.import_state_from(np_state1d=temp[i])
                        self._states.append(rs)
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


if __name__ == "__main__":
    # test code for rollout file check
    task = "pouring_skill"
    roll = RolloutManager(task_name=task)
    roll.load_from_file(batch_idx=1, rollout_idx=6)
    for i in range(roll.len()):
        state, action, done, info = roll.get(i)
        print("state: ", state)

