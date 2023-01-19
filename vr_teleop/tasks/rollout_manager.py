from dataclasses import dataclass

"""
written by twkim
Rollout / demonstration dataset management

"""


@dataclass
class RobotState:
    joint: float = None
    gripper: bool = None
    control_mode: str = None


class RolloutManager:
    def __init__(self, info=""):
        self._states = []    # joint, gripper, etc.
        self._actions = []
        self._info = info

    def append(self, state, action):
        assert type(state) is RobotState
        self._states.append(state)
        self._actions.append(action)

    def get(self, index):
        assert 0 <= index < self.len()
        return self._states[index], self._actions[index]

    def len(self):
        assert len(self._states) == len(self._actions)
        return len(self._states)

    def reset(self):
        self._states = []
        self._actions = []

    def show_current_rollout_info(self):
        print("====================================")
        print("Current rollout dataset info.")
        print("Rollout length: ", self.len())
        sample_state, sample_action = self.get(0)
        print("")

    def save_to_file(self):
        pass

    def load_from_file(self):
        pass

