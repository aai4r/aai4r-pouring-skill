"""
* Dataset management classes
Author: twkim

Folder Structure
root: (dataset)
-- task1
---- dataset_description.txt
------ (task name, and so on.. )
---- batch1
------ rollout0.h5py
"""

import os
import sys
import numpy as np

from vr_teleop.tasks.rollout_manager import RobotState


class BatchRolloutRepo(object):
    def __init__(self, root_dir, task_name, dataset_desc=""):
        self.root_dir = root_dir
        self.task_name = task_name
        self.task_dir = os.path.join(self.root_dir, self.task_name)

        self.cut_off_len = 50   # Does NOT save the noisy rollout episode less than cut_off
        self.batch_index = self.get_last_batch_num() + 1    # batch start from 1

    def get_last_batch_num(self):     # return batch num 0 if the path does not exist
        if not os.path.exists(self.task_dir): return 0
        folders = os.listdir(self.task_dir)
        latest_num = 0
        if folders:
            _batch = sorted(folders, key=lambda x: int(x[len('batch'):]))[-1]  # assume folder name 'batch'
            latest_num = int(_batch[len('batch'):])
        return latest_num

    def save_rollout_to_file(self, episode):
        if len(episode.state) < self.cut_off_len:
            print("Invalid episode length... {} < {}".format(len(episode.state), self.cut_off_len))
            return
        batch_dir = ""
        # if not os.path.exists(path):
        #     os.makedirs(path)


if __name__ == "__main__":
    # test code
    d = BatchRolloutRepo(root_dir=os.getcwd(), task_name="pourwing_water")
    print("root ", d.task_dir)
    for _ in range(10):
        epi = RobotState().random_data(n_joint=6, n_cont_mode=2)
        print(epi)
    # print(d.get_last_batch_num())

