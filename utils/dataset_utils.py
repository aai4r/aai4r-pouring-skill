import os
import h5py
import torch
import numpy as np
import cv2

from spirl.utils.general_utils import AttrDict


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
            # print("idx: {},  batch folder: {}".format(batch_idx + 1, folder))
            path = os.path.join(self.data_path, self.task_name, "batch{}".format(batch_idx + 1))
            rollout_list = sorted(os.listdir(path), key=lambda x: int(x[x.find('_') + 1:x.find('.')]))

            for rollout in rollout_list:
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
                    for img in data.images:

                        print("    rollout: {} step: {} / {}, img, shape: {}, min/max: {}/{}  type: {}".format(
                            int(rollout[rollout.find('_')+1:rollout.find('.')]),
                            step, len(data.images), img.shape, img.min(), img.max(), img.dtype))
                        step += 1
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow(task_name, img)
                        k = cv2.waitKey(0)
                        if k == 27:
                            exit_flag = True
                            break

                    if exit_flag:
                        break

    def statistics(self):
        frames = []
        rollout_count = []
        for batch_idx, folder in enumerate(self.folder_list):
            # print("idx: {},  batch folder: {}".format(batch_idx + 1, folder))
            path = os.path.join(self.data_path, self.task_name, "batch{}".format(batch_idx + 1))
            rollout_list = sorted(os.listdir(path), key=lambda x: int(x[x.find('_')+1:x.find('.')]))
            # print("rollout lists: ", rollout_list)
            rollout_count.append(len(rollout_list))

            for rollout in rollout_list:
                _path = os.path.join(path, rollout)
                with h5py.File(_path, 'r') as f:
                    key = 'traj{}'.format(0)
                    # print("keys: ", f[key].keys())

                    for name in f[key].keys():
                        if name in ['states']:
                            data = f[key + '/' + name][()].astype(np.float32)
                            frames.append(len(data))
                            break

        print("*******************")
        print("***** SUMMARY *****")
        print("*******************")
        print("rollout count, len: {}, total: {:,}, values: {}".format(len(rollout_count), sum(rollout_count), rollout_count))
        print("frames, len: {}, sum: {:,}, values: {}".format(len(frames), sum(frames), frames))


def rollout_play():
    data_path = './data'
    task_name = 'pouring_water_img'    # block_stacking, pouring_water_img

    batch_index = 1
    exit_flat = False
    for i in range(0, 36):
        rollout_index = i
        filename = "rollout_" + str(rollout_index) + '.h5'
        path = os.path.join(data_path, task_name, "batch{}".format(batch_index), filename)
        print('path: ', path)
        with h5py.File(path, 'r') as f:
            data = AttrDict()

            key = 'traj{}'.format(0)
            print("keys: ", f[key].keys())
            # key list of 'block_stacking': [actions, images, pad_mask, states]
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
            for img in data.images:
                print("    rollout: {} step: {} / {}, img, shape: {}, min/max: {}/{}  type: {}".format(
                    rollout_index, step, len(data.images), img.shape, img.min(), img.max(), img.dtype))
                step += 1
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(task_name, img)
                k = cv2.waitKey(0)
                if k == 27:
                    exit_flat = True
                    break

            if exit_flat:
                break


def save_test():
    num_env = 32
    num_trans = 300
    obs_shape = (12, )
    obs = torch.rand(num_trans, num_env, *obs_shape)
    done = torch.zeros(num_trans, num_env, 1)

    obs = obs.permute(1, 0, 2).cpu().numpy()
    done = done.permute(1, 0, 2).cpu().numpy()
    done[0, [7, 30, 120, 230, 281]] = 1

    print("obs shape: {}, done shape: {}".format(obs.shape, done.shape))

    epi_idx = np.where(done[0] > 0)[0]
    epi_idx = np.append([-1], epi_idx)
    print("epi_idx: ", epi_idx)
    for i in range(0, len(epi_idx) - 1):
        start = epi_idx[i] + 1
        end = epi_idx[i+1]
        print("start: {}, end: {},   length: {}".format(start, end, end - start))
        print("obs:: ", len(obs[0, start:end]))


def dataset_statistics():
    data_path = './data'
    task_name = 'pouring_water_img'  # block_stacking, pouring_water_img
    path = os.path.join(data_path, task_name)
    print("path: {}".format(path))

    folder_list = sorted(os.listdir(path), key=lambda x : int(x[5:]))   # batch x
    print("folder list: {}".format(folder_list))

    rollout_count = []
    for batch in folder_list:
        sub_path = os.path.join(path, batch)
        rollouts = os.listdir(sub_path)
        rollout_count.append(len(rollouts))

    print("Number of rollout data: {}".format(rollout_count))
    print("Total rollout: {}".format(sum(rollout_count)))

    # count the total frames
    frames = []
    for batch_idx, rout in enumerate(rollout_count):
        print("idx: {},  rollout count: {}".format(batch_idx + 1, rout))
        path = os.path.join(data_path, task_name, "batch{}".format(batch_idx + 1))
        file_list = os.listdir(path)
        print("file names: ", file_list)

        for file in file_list:
            _path = os.path.join(path, file)
            with h5py.File(_path, 'r') as f:
                key = 'traj{}'.format(0)
                # print("keys: ", f[key].keys())
                for name in f[key].keys():
                    if name in ['states']:
                        data = f[key + '/' + name][()].astype(np.float32)
                        # print("data shape: ", data.shape)
                        frames.append(len(data))

    print("frames, len: {}, sum: {:,}, values: {}".format(len(frames), sum(frames), frames))


if __name__ == '__main__':
    data_path = "../data"
    task_name = "pouring_water_img"      # block_stacking, pouring_water_img
    # dataset_statistics()
    # rollout_play()
    # save_test()

    du = DatasetUtil(data_path=data_path, task_name=task_name)
    # du.statistics()
    du.rollout_play()

