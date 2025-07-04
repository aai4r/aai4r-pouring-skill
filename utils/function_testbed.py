import time

from utilities import TaskPathManager
import torch


def function_test():
    device = "cuda:0"
    num_env = 10
    num_task_steps = 5
    task = TaskPathManager(num_env=num_env, num_task_steps=num_task_steps, device=device)
    task.print_task_status()

    pos = torch.ones(num_env, 3, device=device)
    rot = torch.ones(num_env, 4, device=device)
    grip = torch.ones(num_env, 1, device=device)
    for i in range(5):
        task.push_task_pose(env_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pos=pos, rot=rot, grip=grip)
        pos += 1
        rot += 1
        grip += 1
    task.print_task_status()
    pos, rot, grip = task.get_desired_pose()
    print("get pos: \n{}, rot: \n{}, grip: \n{}".format(pos, rot, grip))
    print("shapes, pos: {}, rot: {}, grip: {}".format(pos.shape, rot.shape, grip.shape))

    task.update_step_by_checking_arrive(torch.rand_like(pos), torch.rand_like(rot), torch.rand_like(grip))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch_jit_utils import quat_apply, to_torch
from vr_teleop.tasks.rollout_manager import RolloutManagerExpand
from spirl.utility.general_utils import AttrDict
from utilities import Rx, Ry, Rz, deg2rad


# TODO!
class CoordViz:
    def __init__(self, elev=30, azim=-60):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.view_init(elev=elev, azim=azim)
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.view_init(elev=elev, azim=azim)

        self.dr = 1.5   # drawing range
        self.bl = 0.5   # basis length
        self.origin = [0, 0, 0]
        self.basis_x = [self.bl, 0, 0]
        self.basis_y = [0, self.bl, 0]
        self.basis_z = [0, 0, self.bl]

        self.draw_basis()
        self.set_viz_form()

    def set_viz_form(self, labels=None):
        if labels is None:
            labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
        self.ax1.set_xticks([-self.dr, 0, self.dr] if labels[0] else [])
        self.ax1.set_yticks([-self.dr, 0, self.dr] if labels[1] else [])
        self.ax1.set_zticks([-self.dr, 0, self.dr] if labels[2] else [])
        self.ax1.set_xlabel(labels[0]), self.ax1.set_ylabel(labels[1]), self.ax1.set_zlabel(labels[2])
        self.ax1.set_title('Source Trajectory')

        self.ax2.set_xticks([-self.dr, 0, self.dr] if labels[0] else [])
        self.ax2.set_yticks([-self.dr, 0, self.dr] if labels[1] else [])
        self.ax2.set_zticks([-self.dr, 0, self.dr] if labels[2] else [])
        self.ax2.set_xlabel(labels[0]), self.ax2.set_ylabel(labels[1]), self.ax2.set_zlabel(labels[2])
        self.ax2.set_title('Constrained Trajectory')

    def draw_basis(self):
        self.draw_line_left(p1=self.origin, p2=self.basis_x, color='r')
        self.draw_line_left(p1=self.origin, p2=self.basis_y, color='g')
        self.draw_line_left(p1=self.origin, p2=self.basis_z, color='b')

        self.draw_line_right(p1=self.origin, p2=self.basis_x, color='r')
        self.draw_line_right(p1=self.origin, p2=self.basis_y, color='g')
        self.draw_line_right(p1=self.origin, p2=self.basis_z, color='b')

    def draw_line_left(self, p1, p2, color='black'):
        self.ax1.plot(xs=[p1[0], p2[0]],
                      ys=[p1[1], p2[1]],
                      zs=[p1[2], p2[2]], color=color)

    def draw_line_right(self, p1, p2, color='black'):
        self.ax2.plot(xs=[p1[0], p2[0]],
                      ys=[p1[1], p2[1]],
                      zs=[p1[2], p2[2]], color=color)

    def refresh(self, dt=0.1, labels=None):
        self.set_viz_form(labels)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(dt)

    def show(self):
        plt.show()
        Axes3D.plot()


class RotCoordViz(CoordViz):
    def __init__(self, conf_mode):
        self.conf_mode = conf_mode
        rot_mode = self.conf_mode.rot_mode
        assert rot_mode in ['alpha', 'beta', 'gamma']
        elev, azim = conf_mode[rot_mode].elev, conf_mode[rot_mode].azim
        batch_idx, rollout_idx = conf_mode.batch_idx, conf_mode[rot_mode].rollout_idx
        super().__init__(elev=elev, azim=azim)

        self.fig.canvas.manager.set_window_title(self.conf_mode.mode + ", " + self.conf_mode.rot_mode)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rollout = RolloutManagerExpand(self.conf_mode.task_name)
        self.rollout.load_from_file(batch_idx=batch_idx, rollout_idx=rollout_idx)

    def quat_to_mat(self, q):
        _q = q if torch.is_tensor(q) else torch.tensor(q, device=self.device)
        assert len(_q.shape) == 1   # [x, y, z, w]
        px = quat_apply(_q, to_torch([1, 0, 0], device=self.device, dtype=torch.float32)).cpu().numpy()
        py = quat_apply(_q, to_torch([0, 1, 0], device=self.device, dtype=torch.float32)).cpu().numpy()
        pz = quat_apply(_q, to_torch([0, 0, 1], device=self.device, dtype=torch.float32)).cpu().numpy()
        return np.stack((px, py, pz), axis=0)

    def draw_coord_to_left(self, mat):
        px, py, pz = mat[0], mat[1], mat[2]
        self.draw_line_left(p1=self.origin, p2=px, color='r')
        self.draw_line_left(p1=self.origin, p2=py, color='g')
        self.draw_line_left(p1=self.origin, p2=pz, color='b')

    def draw_coord_to_right(self, mat):
        px, py, pz = mat[0], mat[1], mat[2]
        self.draw_line_right(p1=self.origin, p2=px, color='r')
        self.draw_line_right(p1=self.origin, p2=py, color='g')
        self.draw_line_right(p1=self.origin, p2=pz, color='b')


def coord_viz():
    print("Coordinate Viz!")
    fwd = AttrDict(mode="forward",
                   task_name="pouring_constraint",
                   rot_mode="beta",
                   batch_idx=1,
                   alpha=AttrDict(elev=0, azim=180, rollout_idx=6, labels=['', 'Y-Axis', 'Z-Axis']),
                   beta=AttrDict(elev=0, azim=90, rollout_idx=7, labels=['X-Axis', '', 'Z-Axis']),
                   gamma=AttrDict(elev=90, azim=180, rollout_idx=8, labels=['X-Axis', 'Y-Axis', '']))

    dwn = AttrDict(mode="downward",
                   task_name="pick_and_place_constraint",
                   rot_mode="gamma",
                   batch_idx=1,
                   alpha=AttrDict(elev=0, azim=180, rollout_idx=6, labels=['', 'Y-Axis', 'Z-Axis']),
                   beta=AttrDict(elev=0, azim=-90, rollout_idx=7, labels=['X-Axis', '', 'Z-Axis']),
                   gamma=AttrDict(elev=90, azim=180, rollout_idx=8, labels=['X-Axis', 'Y-Axis', '']))

    cv = RotCoordViz(conf_mode=dwn)   # elev=30, azim=145
    for i in range(len(cv.rollout._actions)-1):
        # print("quat_target: ", cv.rollout._actions[i][3:7])
        q_source = cv.rollout._actions[i][3:7]
        q_target = cv.rollout._extra[i][:]

        mat_source = cv.quat_to_mat(q_source)
        mat_target = cv.quat_to_mat(q_target)

        cv.draw_coord_to_left(mat_source)
        cv.draw_coord_to_right(mat_target)  # order change..
        # cv.draw_coord_to_right(mat_target[:, [2, 0, 1]])    # order change..
        # break
        cv.refresh(dt=0.01, labels=cv.conf_mode[cv.conf_mode.rot_mode].labels)
    cv.show()


from vr_teleop.tasks.lib_modules import BaseRTDE


def grip_test():
    rt = BaseRTDE("192.168.0.75")
    rt.move_grip_to(0)


if __name__ == '__main__':
    # function_test()
    coord_viz()
    # grip_test()
