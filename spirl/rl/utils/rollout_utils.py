import os
import cv2
import h5py
import numpy as np


class RolloutRepository(object):
    """Saves rollout episodes to a target directory."""
    def __init__(self, root_dir, task_name, batch_start_num=None):
        self.root_dir = root_dir
        self.task_name = task_name

        if (batch_start_num is not None) and (not type(batch_start_num) == int):
            raise TypeError("Invalid type, batch_start_num should be integer type")

        self.batch_count = self.get_last_batch_num() + 1 if batch_start_num is None else batch_start_num
        self.epi_count = 0
        self.cut_off_len = 50       # Doesn't save the data less than cut off length
        self.front_trim_len = 3     # trim first three frames due to noisy inputs of previous episode

    def get_last_batch_num(self):     # return batch num 0 if the path does not exist
        path = os.path.join(self.root_dir, self.task_name)
        if not os.path.exists(path): return 0
        folders = os.listdir(path)
        latest_num = 0
        if folders:
            _batch = sorted(folders, key=lambda x: int(x[len('batch'):]))[-1]  # assume folder name 'batch'
            latest_num = int(_batch[len('batch'):])
        return latest_num

    def save_rollout_to_file(self, episode):
        """Saves an episode to the next file index of the target folder."""
        if len(episode.done) < self.cut_off_len:
            print("Invalid episode length... {} < {}".format(len(episode.done), self.cut_off_len))
            return

        # front trim
        for key in episode: episode[key] = episode[key][self.front_trim_len:]

        batch_folder = "batch{}".format(self.batch_count) if self.batch_count is not None else ""
        task_batch_path = os.path.join(self.root_dir, self.task_name, batch_folder)
        if not os.path.exists(task_batch_path):
            os.makedirs(task_batch_path)
        save_path = os.path.join(task_batch_path, "rollout_{}.h5".format(self.epi_count))

        # save rollout to file
        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        # store trajectory info in traj0 group
        traj_data = f.create_group("traj0")
        traj_data.create_dataset("states", data=np.array(episode.state))
        traj_data.create_dataset("images", data=np.array([x * 255.0 for x in episode.image], dtype=np.uint8))
        traj_data.create_dataset("actions", data=np.array(episode.action))

        terminals = np.array(episode.done)
        if np.sum(terminals) == 0:
            terminals[-1] = True

        # build pad-mask that indicates how long sequence is
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj_data.create_dataset("pad_mask", data=pad_mask)

        f.close()

        self.epi_count += 1     # TODO, batch up count

    def _resize_video(self, images, dim=64):
        """Resize a video in numpy array form to target dimension."""
        ret = np.zeros((images.shape[0], dim, dim, 3))

        for i in range(images.shape[0]):
            ret[i] = cv2.resize(images[i], dsize=(dim, dim),
                                interpolation=cv2.INTER_CUBIC)

        return ret.astype(np.uint8)

    def reset(self):
        """Resets episode counter."""
        self.epi_count = 0

    def print_info(self):
        print("===============================")
        print("Rollout Repository Information")
        print("===============================")
        print("  * root_dir: {}".format(self.root_dir))
        print("  * task_name: {}".format(self.task_name))
        print("  * batches: {}".format(self.batch_count))
        print("")
