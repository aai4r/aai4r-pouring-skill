import os.path
import time
import sys

from utils.torch_jit_utils import *
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import rtde_control
import rtde_receive
from vr_teleop.gripper.robotiq_gripper_control import RobotiqGripper

from spirl.utils.general_utils import AttrDict
from utils.utils import euler_to_mat3d, orientation_error
from pytorch3d import transforms as tr
from base import VRWrapper


def rad2deg(rad): return rad * (180.0 / np.pi)
def deg2rad(deg): return deg * (np.pi / 180.0)


def quat_to_real_last(q_real_first):
    return torch.cat((q_real_first[1:], q_real_first[0].unsqueeze(0)))     # [x, y, z, w]


class RealUR3:
    def __init__(self):
        self.init_vr()
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

        self.v_ax = AttrDict(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)   # desired velocity of each axis
        self.spd_X_limit = 0.08
        self.spd_R_limit = 0.1
        self.ap = 0.9   # low-pass filter

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
        self.vr = VRWrapper(device="cpu", rot_d=(89.99, -89.99, 0.0))

    def limit_check(self, tcp_p):
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

        lim_flag = AttrDict.fromkeys(self.lim_ax, False)

        # x-axis limit
        if tcp_p[0] >= self.lim_ax.x_max and self.v_ax.x > 0: lim_flag.x_max = True  # self.v_ax.x *= -1  # 0.0
        if tcp_p[0] <= self.lim_ax.x_min and self.v_ax.x < 0: lim_flag.x_min = True  # self.v_ax.x *= -1  # 0.0

        # y-axis limit
        if tcp_p[1] >= self.lim_ax.y_max and self.v_ax.y > 0: lim_flag.y_max = True  # self.v_ax.y *= -1  # 0.0
        if tcp_p[1] <= self.lim_ax.y_min and self.v_ax.y < 0: lim_flag.y_min = True  # self.v_ax.y *= -1  # 0.0

        # z-axis limit
        if tcp_p[2] >= self.lim_ax.z_max and self.v_ax.z > 0: lim_flag.z_max = True  # self.v_ax.z *= -1  # 0.0
        if tcp_p[2] <= self.lim_ax.z_min and self.v_ax.z < 0: lim_flag.z_min = True  # self.v_ax.z *= -1  # 0.0

        # roll limit
        if rad_yz >= self.lim_ax.rx_max and self.v_ax.rx > 0: lim_flag.rx_max = True  # self.v_ax.rx *= -1  # 0.0
        if rad_yz < self.lim_ax.rx_min and self.v_ax.rx < 0: lim_flag.rx_min = True  # self.v_ax.rx *= -1  # 0.0

        # pitch limit
        if rad_xz >= self.lim_ax.ry_max and self.v_ax.ry > 0: lim_flag.ry_max = True  # self.v_ax.ry *= -1  # 0.0
        if rad_xz < self.lim_ax.ry_min and self.v_ax.ry < 0: lim_flag.ry_min = True  # self.v_ax.ry *= -1  # 0.0

        # yaw limit
        if rad_xy >= self.lim_ax.rz_max and self.v_ax.rz > 0: lim_flag.rz_max = True  # self.v_ax.rz *= -1  # 0.0
        if rad_xy < self.lim_ax.rz_min and self.v_ax.rz < 0: lim_flag.rz_min = True  # self.v_ax.rz *= -1  # 0.0
        return lim_flag

    def workspace_verify(self):
        try:
            # Move to initial joint position
            init_joint = [deg2rad(8.5), deg2rad(-110), deg2rad(-115),
                          deg2rad(-135), deg2rad(-82), deg2rad(0)]
            self.rtde_c.moveJ(init_joint)
            init_tcp_p = self.rtde_r.getActualTCPPose()

            # determine limit positions in each axis
            # [x, y, z, roll, pitch, yaw]
            check_cnt = AttrDict(x=0, y=0, z=0, rx=0, ry=0, rz=0)
            test_vel = AttrDict(x=0.05, y=0.05, z=0.05, rx=0.2, ry=0.1, rz=0.1)
            for key, val in test_vel.items():
                _vel = AttrDict.fromkeys(test_vel, 0.0)
                _vel[key] = val
                self.set_velL(vel=list(_vel.values()))

                while True:
                    actual_tcp_p = self.rtde_r.getActualTCPPose()
                    # print("actual_tcp_p: ", actual_tcp_p)

                    lim_flag = self.limit_check(tcp_p=actual_tcp_p)
                    if lim_flag.x_max or lim_flag.x_min:
                        self.v_ax.x *= -1.0
                        check_cnt.x += 1
                    if lim_flag.y_max or lim_flag.y_min:
                        self.v_ax.y *= -1.0
                        check_cnt.y += 1
                    if lim_flag.z_max or lim_flag.z_min:
                        self.v_ax.z *= -1.0
                        check_cnt.z += 1
                    if lim_flag.rx_max or lim_flag.rx_min:
                        self.v_ax.rx *= -1.0
                        check_cnt.rx += 1
                    if lim_flag.ry_max or lim_flag.ry_min:
                        self.v_ax.ry *= -1.0
                        check_cnt.ry += 1
                    if lim_flag.rz_max or lim_flag.rz_min:
                        self.v_ax.rz *= -1.0
                        check_cnt.rz += 1

                    self.rtde_c.speedL(xd=list(self.v_ax.values()), acceleration=1.2)

                    p_err = np.linalg.norm(np.array(init_tcp_p)[:3] - np.array(actual_tcp_p)[:3])
                    r_err = np.linalg.norm(np.array(init_tcp_p)[3:] - np.array(actual_tcp_p)[3:])
                    if check_cnt[key] >= 2 and p_err < 0.02 and r_err < 0.03:
                        break
        finally:
            print("end of control... ")
            if not hasattr(self, "rtde_c"): return
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()

    def set_velJ(self):
        pass

    def set_velL(self, v):
        assert len(v) == len(self.v_ax)
        # apply speed limit
        _x = v[0] if v[0] == 0 else min(abs(v[0]), self.spd_X_limit) * (1.0 if v[0] > 0 else -1.0)
        _y = v[1] if v[1] == 0 else min(abs(v[1]), self.spd_X_limit) * (1.0 if v[1] > 0 else -1.0)
        _z = v[2] if v[2] == 0 else min(abs(v[2]), self.spd_X_limit) * (1.0 if v[2] > 0 else -1.0)

        _rx = v[3] if v[3] == 0 else min(abs(v[3]), self.spd_R_limit) * (1.0 if v[3] > 0 else -1.0)
        _ry = v[4] if v[4] == 0 else min(abs(v[4]), self.spd_R_limit) * (1.0 if v[4] > 0 else -1.0)
        _rz = v[5] if v[5] == 0 else min(abs(v[5]), self.spd_R_limit) * (1.0 if v[5] > 0 else -1.0)

        # low-pass filter
        self.v_ax.x = _x * self.ap + self.v_ax.x * (1.0 - self.ap)
        self.v_ax.y = _y * self.ap + self.v_ax.y * (1.0 - self.ap)
        self.v_ax.z = _z * self.ap + self.v_ax.z * (1.0 - self.ap)
        self.v_ax.rx = _rx * self.ap + self.v_ax.rx * (1.0 - self.ap)
        self.v_ax.ry = _ry * self.ap + self.v_ax.ry * (1.0 - self.ap)
        self.v_ax.rz = _rz * self.ap + self.v_ax.rz * (1.0 - self.ap)

    def vr_handler(self):
        while True:
            cont_status = self.vr.get_controller_velocity()
            if cont_status["btn_gripper"]:
                print(cont_status["pose_quat"])

    def run_vr_teleop(self):
        print("Run VR teleoperation mode")
        try:
            # Move to initial joint position
            init_joint = [deg2rad(8.5), deg2rad(-102), deg2rad(-108),
                          deg2rad(-150), deg2rad(-82), deg2rad(0)]
            self.rtde_c.moveJ(init_joint)

            # determine limit positions in each axis
            # [x, y, z, roll, pitch, yaw]
            self.set_velL(v=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print_cnt = 0
            while True:
                start_t = self.rtde_c.initPeriod()
                actual_tcp_p = self.rtde_r.getActualTCPPose()
                _tcp_q = tr.axis_angle_to_quaternion(torch.tensor(actual_tcp_p[3:]))
                tcp_q = torch.cat((_tcp_q[1:], _tcp_q[0].unsqueeze(0)))

                # get velocity command from VR
                cont_status = self.vr.get_controller_status()
                if cont_status["btn_reset_pose"]:
                    self.rtde_c.speedStop()
                    self.rtde_c.moveJ(init_joint)
                    continue

                des_pose = [0.507, -0.064, 0.184, 1.155, 1.267, 1.23]
                dj = self.rtde_c.getInverseKinematics(x=des_pose)
                acc = 1.2
                dt = 1.0 / 500
                # self.rtde_c.speedJ(list(np.array(dj) * 0.001), acc, dt)
                # self.rtde_c.waitPeriod(start_t)

                if print_cnt % 10 == 0:
                    print_cnt = 0
                    print("ik q: ", dj)
                    # print("tcp_q: ", tcp_q)
                    # print("actual_tcp_p: ", actual_tcp_p)
                print_cnt += 1
                continue

                self.set_velL(v=list(cont_status["lin_vel"]) + list(cont_status["ang_vel"]))

                lim_flag = self.limit_check(tcp_p=actual_tcp_p)
                if lim_flag.x_max or lim_flag.x_min: self.v_ax.x *= 0.0
                if lim_flag.y_max or lim_flag.y_min: self.v_ax.y *= 0.0
                if lim_flag.z_max or lim_flag.z_min: self.v_ax.z *= 0.0
                if lim_flag.rx_max or lim_flag.rx_min: self.v_ax.rx *= 0.0
                if lim_flag.ry_max or lim_flag.ry_min: self.v_ax.ry *= 0.0
                if lim_flag.rz_max or lim_flag.rz_min: self.v_ax.rz *= 0.0
                self.rtde_c.speedL(xd=list(self.v_ax.values()), acceleration=1.2)
                self.rtde_c.waitPeriod(start_t)
                print_cnt += 1
        finally:
            print("end of control... ")
            if not hasattr(self, "rtde_c"): return
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()

    def func_test(self):
        init_joint = [deg2rad(8.5), deg2rad(-102), deg2rad(-108),
                      deg2rad(-150), deg2rad(-82), deg2rad(0)]
        self.rtde_c.moveJ(init_joint)

        try:
            cnt = 0
            j_spd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.02]
            for i in range(1000):
                start_t = self.rtde_c.initPeriod()
                self.rtde_c.speedJ(j_spd, 1.2, 1.0 / 500)
                # self.rtde_c.speedL(xd=j_spd, acceleration=1.2)
                if cnt % 100 == 0:
                    print(i, j_spd)
                    print("start_t ", start_t)
                    cnt = 0
                cnt += 1
                self.rtde_c.waitPeriod(start_t)
        finally:
            print("prgram end")
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()
        # sec_joint = [deg2rad(8.5), deg2rad(-102), deg2rad(-108),
        #              deg2rad(-150), deg2rad(-82), deg2rad(90)]
        # self.rtde_c.moveJ(sec_joint)
        # self.rtde_c.moveJ(init_joint)


if __name__ == "__main__":
    u = RealUR3()
    # u.vr_handler()
    # u.workspace_verify()
    u.run_vr_teleop()
    # u.func_test()
