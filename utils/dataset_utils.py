import os
import h5py
import torch
import numpy as np
import cv2
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
plt.ion()

from spirl.utility.general_utils import AttrDict
np.set_printoptions(precision=3)

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow


class SkillDatasetManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Skill Dataset Manager GUI")
        self.statusBar().showMessage('Ready')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()


class DatasetUtil:
    def __init__(self, data_path, task_name, plot_state):
        self.data_path = data_path
        self.task_name = task_name

        # init path and folder
        self.path = os.path.join(self.data_path, self.task_name)
        print("path: {}".format(self.path))

        self.folder_list = sorted(os.listdir(self.path), key=lambda x: int(x[5:]))
        print("batch folder list: {}".format(self.folder_list))

        self.plot_state = plot_state
        if plot_state:
            self.init_plot()

        # list to be excluded in dataset
        self.exclusion_list = []    # rollout_#.h5@batch%

    def init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.reset_plot()

    def reset_plot(self):
        self.ax.clear()
        fontlabel = {"fontsize": "large", "color": "gray", "fontweight": "bold"}
        self.ax.set_xlabel("X", fontdict=fontlabel, labelpad=16)
        self.ax.set_ylabel("Y", fontdict=fontlabel, labelpad=16)
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([-0.5, 0.5])
        self.ax.view_init(elev=30, azim=0)
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        self.px = np.array([])
        self.py = np.array([])
        self.pz = np.array([])

    def plot3d(self, p, label='', color=''):
        self.px = np.append(self.px, p[0])
        self.py = np.append(self.py, p[1])
        self.pz = np.append(self.pz, p[2])
        self.ax.plot(self.px, self.py, self.pz, label=label, color='red')
        if self.ax.get_legend() is None:
            self.ax.legend()

        # self.ax.scatter(p[0], p[1], p[2], color="green")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def append_exclusion(self, batch_name, rollout_name):
        integrated = rollout_name + '@' + batch_name
        if not integrated in self.exclusion_list:
            self.exclusion_list.append(integrated)

    def print_exclusion_list(self):
        if not self.exclusion_list:
            print("Empty exclusion list...")
            return

        for string in self.exclusion_list:
            idx = string.find("@")
            if idx < 0: raise ValueError("Invalid string... '@' is missing")
            rollout_name, batch_name = string[:idx], string[idx + 1:]
            print("{} / {}".format(batch_name, rollout_name))

    def print_state_details(self, image, state, action):
        print("    image, shape: {}, min/max: {}/{}, type: {}".format(image.shape, image.min(), image.max(), image.dtype))
        print("    state, shape: {}".format(state.shape))
        print("    action: joint={}, grip={}".format(action[:6], action[6:]))
        print("        dof_pos: {}".format(state[:6]))
        print("        grip_pos: {}".format(state[6:7]))
        offset = 6
        print("        grasp_pos: {}".format(state[7 + offset:10 + offset]))
        print("        grasp_rot: {}".format(state[10 + offset:14 + offset]))
        print("        cup-bottle_tip_diff: {}".format(state[14 + offset:17 + offset]))
        print("        bottle_pos: {}".format(state[17 + offset:20 + offset]))
        print("        bottle_rot: {}".format(state[20 + offset:24 + offset]))
        print("        cup_pos: {}".format(state[24 + offset:27 + offset]))
        print("        cup_rot: {}".format(state[27 + offset:31 + offset]))
        print("        liq_pos: {}".format(state[31 + offset:]))

    def rollout_play(self):
        exit_flag = False
        batch_skip = False
        for batch_idx, folder in enumerate(self.folder_list):
            if exit_flag: break
            # print("idx: {},  batch folder: {}".format(batch_idx + 1, folder))
            path = os.path.join(self.data_path, self.task_name, folder)

            included_extensions = ['h5']
            file_names = [fn for fn in os.listdir(path)
                          if any(fn.endswith(ext) for ext in included_extensions)]
            rollout_list = sorted(file_names, key=lambda x: int(x[x.find('_') + 1:x.find('.')]))

            for rollout in rollout_list:
                if exit_flag: break
                _path = os.path.join(path, rollout)
                # print("path: ", _path)
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
                    # print("pad_mask: {}".format(data.pad_mask))

                    print("{} / {},    {} / {}".format(folder, len(self.folder_list), rollout, len(rollout_list)))
                    step, n_frames = 0, len(data.states)
                    while step < n_frames:
                        img, st, a = data.images[step], data.states[step], data.actions[step]
                        if exit_flag: break
                        print("    step: {} / {}, ".format(step, len(data.images)))
                        # self.print_state_details(image=img, state=st, action=a)
                        if self.plot_state:
                            offset = 6
                            self.plot3d(p=st[7+offset:10+offset], label="grasp_pos")

                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow(task_name + " r: skip rollout, b: skip batch", img)
                        k = cv2.waitKey(0)
                        if k == 27:
                            exit_flag = True
                        elif k == ord('r'):
                            print("skip rollout data")
                            break
                        elif k == ord('b'):
                            print("skip batch")
                            batch_skip = True
                            break
                        elif k == ord('s'):
                            print("step jump")
                            step += 4
                        elif k == ord('m'):
                            print("mark as an exclusive item")
                            step -= 1
                            self.append_exclusion(batch_name=folder, rollout_name=rollout)
                        step += 1

                    if self.plot_state:
                        self.reset_plot()

                if batch_skip:
                    batch_skip = False
                    break

        print("end of loop")
        self.print_exclusion_list()

    def statistics(self):
        frames = {}
        rollout_count = []
        keys = []
        shapes = {}
        data_range = {}
        for batch_idx, folder in enumerate(self.folder_list):
            # print("idx: {},  batch folder: {}".format(batch_idx + 1, folder))
            path = os.path.join(self.data_path, self.task_name, "batch{}".format(batch_idx + 1))
            rollout_list = sorted(os.listdir(path), key=lambda x: int(x[x.find('_')+1:x.find('.')]))
            # print("rollout lists: ", rollout_list)
            rollout_count.append(len(rollout_list))

            print("batch{}...........................".format(batch_idx + 1))
            for rollout in rollout_list:
                _path = os.path.join(path, rollout)
                _rolloutID = os.path.join("batch{}".format(batch_idx + 1), rollout)
                with h5py.File(_path, 'r') as f:
                    key = 'traj{}'.format(0)
                    keys = list(f[key].keys())

                    for name in f[key].keys():  # actions, states, ....
                        data = f[key + '/' + name][()]
                        _min, _max = data.min(), data.max()
                        _min_key, _max_key = name + '_min', name + '_max'
                        data_range[_min_key] = min(data_range[_min_key], _min) if hasattr(data_range, _min_key) else _min
                        data_range[_max_key] = min(data_range[_max_key], _max) if hasattr(data_range, _max_key) else _max

                        if _rolloutID not in frames:
                            frames[_rolloutID] = (len(data.astype(np.float32)))
                            if name not in shapes: shapes[name] = data.shape[1:]  # remove batch shape

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
        print("frames, episodes: {}, sum: {:,}, frame length: {}".
              format(len(frames.values()), sum(frames.values()), frames.values()))
        print("Dataset range: {}".format(data_range))


if __name__ == '__main__':
    data_path = "../data"
    task_name = "pouring_water_img"      # block_stacking, pouring_water_img, office_TA

    # du = DatasetUtil(data_path=data_path, task_name=task_name, plot_state=True)
    # du.statistics()
    # du.rollout_play()

    app = QApplication(sys.argv)
    ex = SkillDatasetManagerGUI()
    sys.exit(app.exec_())
