import os
import h5py
import random
import numpy as np

import dataclasses
from dataclasses import dataclass
from dataset.rollout_dataset import BatchRolloutFolder
from spirl.utils.general_utils import AttrDict

"""
written by twkim
Rollout / demonstration dataset management

"""


@dataclass
class RobotState:
    joint: list = None
    gripper: list = None
    control_mode: list = None

    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    def item_vec(self):
        return self.joint + self.gripper + self.control_mode

    @staticmethod
    def random_data(n_joint, n_cont_mode):
        _joint = [random.randint(-100, 100) / 200.0 for _ in range(n_joint)]
        grip_on = random.randint(-100, 100) > 0
        _gripper = [int(grip_on), int(not grip_on)]
        _control_mode = [0] * n_cont_mode
        _control_mode[random.randint(0, n_cont_mode - 1)] = 1
        return RobotState(joint=_joint, gripper=_gripper, control_mode=_control_mode)


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
        self._info.append(str(info))

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
        np_rollout.info = np.array(self._info)
        return np_rollout

    def show_current_rollout_info(self):
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
        traj = f.create_group("traj0")
        traj.create_dataset("states", data=np_episode_dict.states)
        traj.create_dataset("actions", data=np_episode_dict.actions)
        traj.create_dataset("info", data="np_episode_dict.info")    # TODO

        terminals = np_episode_dict.dones
        if np.sum(terminals) == 0: terminals[-1] = True
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj.create_dataset("pad_mask", data=pad_mask)
        f.close()
        print("save to ", save_path)

    def load_from_file(self):
        self.reset()
        load_path = self.get_final_load_path(batch_index=self.batch_index, rollout_num=self.rollout_idx)
        with h5py.File(load_path, 'r') as f:
            key = 'traj{}'.format(0)
            print("f: ", f[key])
            for name in f[key].keys():
                if name == 'states':
                    temp = f[key + '/' + name][()].astype(np.float32)
                    for i in range(len(temp)):
                        self._states.append(RobotState(joint=temp[i, :6].tolist(),
                                                       gripper=temp[i, 6:8].tolist(),
                                                       control_mode=temp[i, 8:].tolist()))
                    # self._states = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'actions':
                    self._actions = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'pad_mask':
                    self._dones = f[key + '/' + name][()].astype(np.float32).tolist()
                elif name == 'info':
                    temp = f[key + '/' + name][()]
                    print("info temp ", temp, type(temp))
                    self._info = f[key + '/' + name][()]
                else:
                    raise ValueError("{}: Unexpected rollout element...".format(name))
        print("Load complete!")
