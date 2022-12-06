import copy
import os.path
import time

import numpy as np
import sys

from utils.torch_jit_utils import *
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import rtde_control
import rtde_receive
from vr_teleop.gripper.robotiq_gripper_control import RobotiqGripper

from spirl.utils.general_utils import AttrDict
from pytorch3d import transforms as tr


def rad2deg(rad): return rad * (180.0 / np.pi)
def deg2rad(deg): return deg * (np.pi / 180.0)


class RealUR3:
    def __init__(self):
        self.HOST = "192.168.0.75"

        self.rtde_c = rtde_control.RTDEControlInterface(self.HOST)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.HOST)

        self.default_control_params = AttrDict(speed=0.25, acceleration=1.2, blend=0.099)
        self.lim_ax = AttrDict(x_max=0.53, x_min=0.38,
                               y_max=0.2, y_min=-0.2,
                               z_max=0.3, z_min=0.07,
                               rx_max=deg2rad(135.0), rx_min=deg2rad(-135.0),
                               ry_max=deg2rad(20.0), ry_min=deg2rad(-5.0),
                               rz_max=deg2rad(40.0), rz_min=deg2rad(-40.0))

        # direction variable will be removed...
        self.v_ax = AttrDict(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)   # desired velocity of each axis
        self.spd_limit = 0.1

        # for geometry calc.
        self.x_axis = torch.tensor([1.0, 0.0, 0.0])
        self.y_axis = torch.tensor([0.0, 1.0, 0.0])
        self.z_axis = torch.tensor([0.0, 0.0, 1.0])

        self.xy_plane = torch.stack([self.x_axis, self.y_axis], dim=1)
        self.xz_plane = torch.stack([self.x_axis, self.z_axis], dim=1)
        self.yz_plane = torch.stack([self.y_axis, self.z_axis], dim=1)

        # self.Pxy = torch.mm(self.xy_plane.T, self.xy_plane).inverse()
        self.Pxy = (self.xy_plane @ (self.xy_plane.T @ self.xy_plane).inverse()) @ self.xy_plane.T
        self.Pxz = (self.xz_plane @ (self.xz_plane.T @ self.xz_plane).inverse()) @ self.xz_plane.T
        self.Pyz = (self.yz_plane @ (self.yz_plane.T @ self.yz_plane).inverse()) @ self.yz_plane.T

    def init_vr(self):
        pass

    def limit_check(self, tcp_p):
        # x-axis limit
        if tcp_p[0] >= self.lim_ax.x_max and self.v_ax.x > 0: self.v_ax.x *= -1   # 0.0
        if tcp_p[0] <= self.lim_ax.x_min and self.v_ax.x < 0: self.v_ax.x *= -1   # 0.0

        # y-axis limit
        if tcp_p[1] >= self.lim_ax.y_max and self.v_ax.y > 0: self.v_ax.y *= -1   # 0.0
        if tcp_p[1] <= self.lim_ax.y_min and self.v_ax.y < 0: self.v_ax.y *= -1   # 0.0

        # z-axis limit
        if tcp_p[2] >= self.lim_ax.z_max and self.v_ax.z > 0: self.v_ax.z *= -1  # 0.0
        if tcp_p[2] <= self.lim_ax.z_min and self.v_ax.z < 0: self.v_ax.z *= -1  # 0.0

        _q = tr.axis_angle_to_quaternion(torch.tensor(tcp_p[3:]))     # [w, x, y, z]
        q = torch.cat((_q[1:], _q[0].unsqueeze(0)))     # [x, y, z, w]

        qx_axis = tf_vector(q, self.x_axis)      # refer TCP coordinate
        qy_axis = tf_vector(q, self.y_axis)
        qz_axis = tf_vector(q, self.z_axis)

        # pitch
        v_xz = self.Pxz @ qz_axis
        dot_xz = (v_xz @ self.x_axis) / (v_xz.norm() * self.x_axis.norm())
        rad_xz = torch.acos(dot_xz)
        rad_xz *= 1.0 if v_xz[2] < 0 else -1.0

        # yaw
        v_xy = self.Pxy @ qz_axis
        dot_xy = (v_xy @ self.x_axis) / (v_xy.norm() * self.x_axis.norm())
        rad_xy = torch.acos(dot_xy)
        rad_xy *= 1.0 if v_xy[1] > 0 else -1.0

        # roll
        v_yz = self.Pyz @ qy_axis
        dot_yz = (v_yz @ self.z_axis) / (v_yz.norm() * self.z_axis.norm())
        rad_yz = torch.acos(dot_yz)
        rad_yz *= 1.0 if v_yz[1] < 0 else -1.0

        if rad_xz >= self.lim_ax.ry_max and self.v_ax.ry > 0: self.v_ax.ry *= -1  # 0.0
        if rad_xz < self.lim_ax.ry_min and self.v_ax.ry < 0: self.v_ax.ry *= -1  # 0.0

        if rad_xy >= self.lim_ax.rz_max and self.v_ax.rz > 0: self.v_ax.rz *= -1  # 0.0
        if rad_xy < self.lim_ax.rz_min and self.v_ax.rz < 0: self.v_ax.rz *= -1  # 0.0

        if rad_yz >= self.lim_ax.rx_max and self.v_ax.rx > 0: self.v_ax.rx *= -1  # 0.0
        if rad_yz < self.lim_ax.rx_min and self.v_ax.rx < 0: self.v_ax.rx *= -1  # 0.0

    def set_velJ(self):
        pass

    def set_velL(self, vel):
        assert len(vel) == len(self.v_ax)
        self.v_ax.x = vel[0]
        self.v_ax.y = vel[1]
        self.v_ax.z = vel[2]
        self.v_ax.rx = vel[3]
        self.v_ax.ry = vel[4]
        self.v_ax.rz = vel[5]

    def run_vr_teleop(self):
        try:
            # Move to initial joint position
            init_joint = [deg2rad(8.5), deg2rad(-110), deg2rad(-115),
                          deg2rad(-135), deg2rad(-82), deg2rad(0)]
            self.rtde_c.moveJ(init_joint)

            # determine limit positions in each axis
            # [x, y, z, roll, pitch, yaw]
            self.set_velL(vel=[0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
            while True:
                actual_tcp_p = self.rtde_r.getActualTCPPose()
                print("actual_tcp_p: ", actual_tcp_p)

                # get velocity command from VR

                self.limit_check(tcp_p=actual_tcp_p)
                self.rtde_c.speedL(xd=list(self.v_ax.values()), acceleration=1.2)
        finally:
            print("end of control... ")
            self.rtde_c.speedL(xd=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], acceleration=1.2)
            time.sleep(1)
            if not hasattr(self, "rtde_c"): return
            if self.rtde_c:
                self.rtde_c.stopScript()


if __name__ == "__main__":
    u = RealUR3()
    u.run_vr_teleop()
