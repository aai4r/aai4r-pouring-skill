import os
import h5py
import torch
import numpy as np
import cv2

from spirl.utils.general_utils import AttrDict
np.set_printoptions(precision=3)


class DatasetUtil:
    def __init__(self, data_path, task_name):
        self.data_path = data_path
        self.task_name = task_name

        # init path and folder
        self.path = os.path.join(self.data_path, self.task_name)
        print("path: {}".format(self.path))

        self.folder_list = sorted(os.listdir(self.path), key=lambda x: int(x[5:]))
        print("batch folder list: {}".format(self.folder_list))

    def rollout_play(self):
        exit_flag = False
        for batch_idx, folder in enumerate(self.folder_list):
            if exit_flag: break
            # print("idx: {},  batch folder: {}".format(batch_idx + 1, folder))
            path = os.path.join(self.data_path, self.task_name, "batch{}".format(batch_idx + 1))
            rollout_list = sorted(os.listdir(path), key=lambda x: int(x[x.find('_') + 1:x.find('.')]))

            for rollout in rollout_list:
                if exit_flag: break
                _path = os.path.join(path, rollout)
                with h5py.File(_path, 'r') as f:
                    data = AttrDict()
                    key = 'traj{}'.format(0)

                    # Fetch data into a dict
                    for name in f[key].keys():
                        if name in ['actions', 'states', 'rewards', 'terminals', 'pad_mask']:
                            data[name] = f[key + '/' + name][()].astype(np.float32)
                            # print("{}: shape: {}, data: {}".format(name, data[name].shape, data[name]))
                        elif name in ['observations']:
                            data[name] = f[key + '/' + name][()].astype(np.uint8)
                        elif name in ['images']:
                            data[name] = f[key + '/' + name][()].astype(np.uint8)
                        elif name in ['dones']:
                            data[name] = f[key + '/' + name][()].astype(np.float32)

                        print("{}: shape: {}".format(name, data[name].shape))

                    step = 0
                    for img, st in zip(data.images, data.states):
                        if exit_flag: break
                        print("    rollout: {} step: {} / {}, img, shape: {}, min/max: {}/{}  type: {}".format(
                            int(rollout[rollout.find('_')+1:rollout.find('.')]),
                            step, len(data.images), img.shape, img.min(), img.max(), img.dtype))
                        print("        dof_pos: {}".format(st[:6]))
                        print("        grip_pos: {}".format(st[6:7]))
                        print("        bottle_pos: {}".format(st[7:10]))
                        print("        bottle_rot: {}".format(st[10:14]))
                        print("        cup_pos: {}".format(st[14:17]))
                        print("        cup_rot: {}".format(st[17:21]))
                        print("        liq_rot: {}".format(st[21:24]))
                        step += 1
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow(task_name, img)
                        k = cv2.waitKey(0)
                        if k == 27:
                            exit_flag = True

    def statistics(self):
        frames = {}
        rollout_count = []
        keys = []
        shapes = {}
        for batch_idx, folder in enumerate(self.folder_list):
            # print("idx: {},  batch folder: {}".format(batch_idx + 1, folder))
            path = os.path.join(self.data_path, self.task_name, "batch{}".format(batch_idx + 1))
            rollout_list = sorted(os.listdir(path), key=lambda x: int(x[x.find('_')+1:x.find('.')]))
            # print("rollout lists: ", rollout_list)
            rollout_count.append(len(rollout_list))

            for rollout in rollout_list:
                _path = os.path.join(path, rollout)
                _rolloutID = os.path.join("batch{}".format(batch_idx + 1), rollout)
                with h5py.File(_path, 'r') as f:
                    key = 'traj{}'.format(0)
                    keys = list(f[key].keys())

                    for name in f[key].keys():
                        if name not in shapes:
                            data = f[key + '/' + name][()]
                            shapes[name] = data.shape[1:]   # remove batch shape

                        if _rolloutID not in frames:
                            data = f[key + '/' + name][()].astype(np.float32)
                            frames[_rolloutID] = (len(data))

                        # if name in ['actions']:
                        #     data = f[key + '/' + name][()].astype(np.float32)
                        #     frames.append(len(data))
                        #     keys = list(f[key].keys())
                        #     break

        print("*******************")
        print("***** SUMMARY *****")
        print("*******************")
        print("Dataset Keys: {}".format(keys))
        print("Shapes: {}".format(shapes))
        print("rollout: # of batches: {}, total: {:,}, \nrollout counts for each batch folder: {}".
              format(len(rollout_count), sum(rollout_count), rollout_count))
        print("frames, len: {}, sum: {:,}, frame length: {}".
              format(len(frames.values()), sum(frames.values()), frames.values()))


if __name__ == '__main__':
    data_path = "../data"
    task_name = "pouring_water_img"      # block_stacking, pouring_water_img

    du = DatasetUtil(data_path=data_path, task_name=task_name)
    # du.statistics()
    du.rollout_play()
