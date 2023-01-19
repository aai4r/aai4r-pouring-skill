import os
import h5py
import random
import dataclasses
from dataclasses import dataclass

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

    @staticmethod
    def random_data(n_joint, n_cont_mode):
        _joint = [random.randint(-100, 100) / 200.0 for _ in range(n_joint)]
        grip_on = random.randint(-100, 100) > 0
        _gripper = [int(grip_on), int(not grip_on)]
        _control_mode = [0] * n_cont_mode
        _control_mode[random.randint(0, n_cont_mode - 1)] = 1
        return RobotState(joint=_joint, gripper=_gripper, control_mode=_control_mode)


class RolloutManager:
    def __init__(self, root_dir=os.path.dirname(os.getcwd()), save_folder="demo_dataset"):
        self._states = []    # joint, gripper, etc.
        self._actions = []
        self._info = []

        self.save_dir = os.path.join(root_dir, save_folder)
        self.episode_count = 0

    def append(self, state, action, info):
        assert type(state) is RobotState
        self._states.append(state)
        self._actions.append(action)
        self._info.append(info)

    def get(self, index):
        assert 0 <= index < self.len()
        return self._states[index], self._actions[index], self._info[index]

    def len(self):
        assert len(self._states) == len(self._actions) == len(self._info)
        return len(self._states)

    def reset(self):
        self._states = []
        self._actions = []
        self._info = []

    def show_current_rollout_info(self):
        print("====================================")
        print("Current rollout dataset info.")
        print("Rollout length: ", self.len())
        idx = 0
        sample_state, sample_action, sample_info = self.get(idx)
        print("* STEP: [{}]".format(idx))
        print("    state * {} dim with {}".format(sum([len(i) for i in sample_state]), sample_state))
        print("    action * {} dim with {}".format(len(sample_action), sample_action))
        print("    info: ", sample_info)

    def save_to_file(self):
        print("save dir: ", self.save_dir)
        pass

    def load_from_file(self):
        pass

