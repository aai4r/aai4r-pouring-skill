import time
import numpy as np
from vr_teleop.tasks.rollout_manager import RolloutManagerExpand
from vr_teleop.tasks.lib_modules import visualize


class RolloutManipulation:
    def __init__(self, task_name):
        self.rollout = RolloutManagerExpand(task_name=task_name)

    def check(self):
        self.rollout.load_from_file(batch_idx=1, rollout_idx=4)
        for i in range(self.rollout.len()):
            obs, state, action, done, info = self.rollout.get(i)
            cont_mode = ""
            if state.control_mode_one_hot == [1.0, 0.0]:
                cont_mode = "forward"
            elif state.control_mode_one_hot == [0.0, 1.0]:
                cont_mode = "downward"
            print("control mode: [{}], action: {}, done: {}".format(cont_mode, action[-1], np.argmax(done)))

    def yield_dataset(self, src_batch_idx, target_batch_idx):
        n_files = len(self.rollout.get_rollout_list(batch_idx=src_batch_idx))
        for j in range(n_files):
            self.rollout.reset()
            self.rollout.load_from_file(batch_idx=src_batch_idx, rollout_idx=j)
            self.rollout.show_rollout_summary()
            for i in range(self.rollout.len()):
                yield j, i, self.rollout.get(i)

    def extract_features(self):
        pass

    def conf_action(self, src_batch_idx, target_batch_idx):
        for x in self.yield_dataset(src_batch_idx, target_batch_idx):
            j, i = x[:2]
            obs, state, action, done, info = x[-1]
            visualize(np.zeros(obs.shape), obs)
            self.rollout._dones[i] = float(not done)
            if state.control_mode_one_hot == [0.0, 1.0]:  # downward
                action[-1] = -1.0
            elif state.control_mode_one_hot == [1.0, 0.0]:  # forward
                action[-1] = 1.0
            time.sleep(0.001)
        self.rollout.save_to_file(batch_idx=target_batch_idx)


if __name__ == "__main__":
    tasks = ["pouring_skill_img", "pick_and_place_img", "multi_skill_img"]
    rm = RolloutManipulation(task_name="multi_skill_img")
    rm.conf_action(src_batch_idx=1, target_batch_idx=2)
    # rm.check()