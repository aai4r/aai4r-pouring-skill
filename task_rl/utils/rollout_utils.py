import os
import h5py
import numpy as np

from spirl.rl.utils.rollout_utils import RolloutRepository


class RolloutSaverIsaac(RolloutRepository):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(root_dir=cfg['expert']['data_path'], task_name=cfg['task']['name'])
        self.save_resume = cfg['expert']['save_resume']
        self.pre_size = 0

        if self.save_resume:
            self.check_batch_resume()

    def check_batch_resume(self):
        path = os.path.join(self.root_dir, self.task_name)
        if not os.path.exists(path):
            return

        # size check
        self.pre_size = self.get_dir_size(path)
        folders = os.listdir(path)
        if folders:
            latest_num = self.get_last_batch_num()
            self.batch_count = int(latest_num) + 1
            print("Dataset pre-size: {}    Current batch count : {}".format(self.pre_size, self.batch_count))

    def get_dir_size(self, path='.'):
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self.get_dir_size(entry.path)
        return total

    def save_rollout_to_file(self, episode, obs_to_img_key=False):
        """Saves an episode to the next file index of the target folder."""
        _obs_shape = episode.observations.shape
        _opt_path = "batch{}".format(self.batch_count) if len(_obs_shape) > 2 else ""
        path = os.path.join(self.root_dir, self.task_name, _opt_path)
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, "rollout_{}.h5".format(self.epi_count))

        # save rollout to file
        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        # store trajectory info in traj0 group
        traj_data = f.create_group("traj0")
        for key, val in episode.items():
            if key in ['rewards', 'dones']:     # dataset skip
                continue
            _key = 'images' if key in ['observations'] and obs_to_img_key else key  # save 'observations' as 'images'
            traj_data.create_dataset(_key, data=np.array(episode[key], dtype=episode[key].dtype))

        terminals = np.array(episode.dones)
        if np.sum(terminals) == 0:
            terminals[-1] = True

        # build pad-mask that indicates how long sequence is
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj_data.create_dataset("pad_mask", data=pad_mask)

        f.close()

        self.epi_count += 1
        print("Rollout is stored in {}".format(save_path))
