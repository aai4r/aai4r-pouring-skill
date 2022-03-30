import os
import h5py
import numpy as np

from spirl.rl.utils.rollout_utils import RolloutSaver


class RolloutSaverIsaac(RolloutSaver):
    def __init__(self, save_dir, task_name):
        super().__init__(save_dir=save_dir)
        self.task_name = task_name
        self.batch_count = 1

    def save_rollout_to_file(self, episode):
        """Saves an episode to the next file index of the target folder."""
        _obs_shape = episode.observations.shape
        _opt_path = "batch{}".format(self.batch_count) if len(_obs_shape) > 2 else ""
        path = os.path.join(self.save_dir, self.task_name, _opt_path)
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, "rollout_{}.h5".format(self.counter))

        # save rollout to file
        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        # store trajectory info in traj0 group
        traj_data = f.create_group("traj0")
        for key, val in episode.items():
            traj_data.create_dataset(key, data=np.array(episode[key], dtype=episode[key].dtype))

        terminals = np.array(episode.dones)
        if np.sum(terminals) == 0:
            terminals[-1] = True

        # build pad-mask that indicates how long sequence is
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj_data.create_dataset("pad_mask", data=pad_mask)

        f.close()

        self.counter += 1
        print("Rollout is stored in {}".format(save_path))
