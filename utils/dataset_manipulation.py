import time
import numpy as np
import cv2
from vr_teleop.tasks.rollout_manager import RolloutManagerExpand
from vr_teleop.tasks.lib_modules import visualize


class RolloutManipulation:
    def __init__(self, task_name):
        self.rollout = RolloutManagerExpand(task_name=task_name)

    def check(self, src_batch_idx):
        for x in self.yield_dataset(src_batch_idx):
            j, i = x[:2]
            obs, state, action, done, info = x[-1]
            visualize(np.zeros(obs.shape), obs)
            print("obs shape: ", obs.shape)

    def yield_dataset(self, src_batch_idx, target_batch_idx=None):
        n_files = len(self.rollout.get_rollout_list(batch_idx=src_batch_idx))
        for j in range(n_files):
            if j > 0 and target_batch_idx is not None:
                self.rollout.save_to_file(batch_idx=target_batch_idx)
            self.rollout.reset()
            self.rollout.load_from_file(batch_idx=src_batch_idx, rollout_idx=j)
            self.rollout.show_rollout_summary()
            for i in range(self.rollout.len()):
                yield j, i, self.rollout.get(i)

    def extract_features(self, n_features=2000):
        N = 500
        im_size_per_frame = 640 * 480 * 3
        print("N frames: {}".format(N))
        print("img size per frame: {:,}".format(im_size_per_frame))
        print("Total episode size: {:,}".format(im_size_per_frame * N))
        print("Size with {} sampled features: {:,}".format(n_features, 512 * n_features * N))

    def manipulate(self, src_batch_idx, target_batch_idx, height, width):
        for x in self.yield_dataset(src_batch_idx, target_batch_idx=target_batch_idx):
            j, i = x[:2]
            obs, state, action, done, info = x[-1]

            # image resize
            obs = cv2.resize(obs, dsize=(width, height), interpolation=cv2.INTER_AREA)

            # conf action redefinition
            if state.control_mode_one_hot == [0.0, 1.0]:  # downward
                action[-1] = -1.0
            elif state.control_mode_one_hot == [1.0, 0.0]:  # forward
                action[-1] = 1.0

            self.rollout.replace(index=i, image=obs, action=action, done=float(not done))
            visualize(np.zeros(obs.shape), obs)


if __name__ == "__main__":
    tasks = ["pouring_skill_img", "pick_and_place_img", "multi_skill_img"]
    rm = RolloutManipulation(task_name="multi_skill_img")
    rm.manipulate(src_batch_idx=1, target_batch_idx=2, height=240, width=320)
    # rm.conf_action(src_batch_idx=1, target_batch_idx=2)
    # rm.check(src_batch_idx=2)

