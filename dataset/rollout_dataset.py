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
from dataclasses import dataclass


def get_ordered_file_list(path, included_ext):
    file_names = [fn for fn in os.listdir(path)
                  if any(fn.endswith(ext) for ext in included_ext)]
    rollout_list = sorted(file_names, key=lambda x: int(x[x.find('_') + 1:x.find('.')]))
    return rollout_list


@dataclass
class MinMax:
    min: int
    max: int


class BatchRolloutFolder(object):
    """
        Manages batch folder structures such as indexing of batch folder, rollout file, etc.
    """
    def __init__(self, task_name, root_dir, dataset_desc=""):
        self.task_name = task_name
        self.root_dir = os.path.dirname(os.path.abspath(__file__)) if root_dir is None else root_dir
        self.task_dir = os.path.join(self.root_dir, self.task_name)

        self.batch_index_range = MinMax(1, 10000)
        self.rollout_idx_range = MinMax(0, 1000)
        self.batch_index = 1    # batch start from 1
        self.rollout_idx = 0
        self.batch_name = 'batch'

    def get_batch_folders(self):
        return [] if not os.path.exists(self.task_dir) else \
            [batch for batch in os.listdir(self.task_dir) if 'batch' in batch]

    def get_rollout_list(self, batch_idx):
        batches = self.get_batch_folders()
        if not batches: return batches
        rollout_list = []
        for i, b in enumerate(batches):
            if int(b[5:]) == batch_idx:
                rollout_list = get_ordered_file_list(path=os.path.join(self.task_dir, batches[i]),
                                                     included_ext=['h5'])
        return rollout_list

    def get_last_batch_idx(self):     # return batch num 0 if the path does not exist
        if not os.path.exists(self.task_dir): return 0
        folders = os.listdir(self.task_dir)
        latest_num = 0
        if folders:
            _batch = sorted(folders, key=lambda x: int(x[len(self.batch_name):]))[-1]  # assume folder name 'batch'
            latest_num = int(_batch[len(self.batch_name):])
        return latest_num

    def save_path_verify(self, batch_index):
        """
            Check the batch-level folder check and create the folder if not exists
            :return: task-batch folder path
        """
        batch_dir = self.batch_name + "{}".format(batch_index)
        task_batch_dir = os.path.join(self.task_dir, batch_dir)
        if not os.path.exists(task_batch_dir): os.makedirs(task_batch_dir)
        return task_batch_dir

    def load_path_verify(self, batch_index):
        """
            Everything is same except for creating the folder
            :param batch_index:
            :return: task-batch dir
        """
        batch_dir = self.batch_name + "{}".format(batch_index)
        task_batch_dir = os.path.join(self.task_dir, batch_dir)
        if not os.path.exists(task_batch_dir):
            raise OSError("{} not exists".format(task_batch_dir))
        return task_batch_dir

    def get_final_save_path(self, batch_index):
        task_batch_dir = self.save_path_verify(batch_index=batch_index)
        rollout_list = get_ordered_file_list(path=task_batch_dir, included_ext=['h5'])
        next_idx = (lambda x: int(x[x.find('_') + 1:x.find('.')]))(rollout_list[-1]) + 1 if len(rollout_list) > 0 else 0
        save_path = os.path.join(task_batch_dir, "rollout_{}.h5".format(next_idx))
        return save_path

    def get_final_load_path(self, batch_index, rollout_num):
        task_batch_dir = self.load_path_verify(batch_index=batch_index)
        load_path = os.path.join(task_batch_dir, "rollout_{}.h5".format(rollout_num))
        return load_path

    def inc_batch_index(self):
        self.batch_index = min(self.batch_index + 1, self.batch_index_range.max)

    def dec_batch_index(self):
        self.batch_index = max(self.batch_index - 1, self.batch_index_range.min)

    def inc_rollout_idx(self):
        self.rollout_idx = min(self.rollout_idx + 1, self.rollout_idx_range.max)

    def dec_rollout_idx(self):
        self.rollout_idx = max(self.rollout_idx - 1, self.rollout_idx_range.min)

    def is_batch_full(self):
        """
            Determine by capacity or etc.
        """
        raise NotImplementedError


if __name__ == "__main__":
    # test code
    task_name = "pouring_water"
    # d = BatchRolloutRepo(root_dir=os.getcwd(), task_name=task_name)
    #
    # dataset = RolloutManager()
    # for _ in range(100):
    #     _state = RobotState().random_data(n_joint=6, n_cont_mode=2)
    #     _action = list(np.random.rand(6) * 2 - 1.0)
    #     _done = [0]
    #     dataset.append(state=_state, action=_action, done=_done, info="sample info")
    # print(dataset.show_current_rollout_info())
    # np_roll = dataset.to_np_rollout()
    # # d.save_rollout_to_file(episode=np_roll)
    # d.load_rollout_from_file(batch_idx=1)

