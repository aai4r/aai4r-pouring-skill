import os.path
import sys

from utils.torch_jit_utils import *
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import rtde_control
import rtde_receive
from vr_teleop.gripper.robotiq_gripper_control import RobotiqGripper

from spirl.utils.general_utils import AttrDict
from utils.utils import euler_to_mat3d, orientation_error, rad2deg, deg2rad, CustomTimer, \
    quaternion_real_first, quaternion_real_last

from pytorch3d import transforms as tr
from base import VRWrapper

from rollout_manager import RolloutManager, RobotState


def to_n_pi_pi(rad):     # [0, 2*pi] --> [-pi, pi]
    if rad > 0:
        return rad - (2 * np.pi) if rad >= np.pi else rad
    else:
        return rad + (2 * np.pi) if rad <= np.pi else rad


class BaseRTDE:
    def __init__(self, HOST):
        self.HOST = HOST

        self.rtde_c = rtde_control.RTDEControlInterface(self.HOST)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.HOST)

        # gripper control
        self.gripper = RobotiqGripper(self.rtde_c)
        self.gripper.activate()
        self.gripper.set_force(0)  # range: [0, 100]
        self.gripper.set_speed(10)  # range: [0, 100]
        self.grip_on = False

        self.default_control_params = AttrDict(speed=0.25, acceleration=1.2, blend=0.099, dt=1.0 / 500.0)

    @property
    def dt(self):
        return self.default_control_params.dt

    @property
    def acc(self):
        return self.default_control_params.acceleration


class UR3ControlMode:
    def __init__(self, init_mode="forward"):
        self.cmodes = ["forward", "downward"]
        self.cmodes_d = {st: i for i, st in enumerate(self.cmodes)}
        self.CONTROL_MODE = init_mode
        print("Initial control mode: ", self.CONTROL_MODE)

        self._iposes = AttrDict()
        self._limits = AttrDict()
        self._rpy_base = AttrDict()

        self._limits.forward = AttrDict(x_max=0.53, x_min=0.38,
                                        y_max=0.2, y_min=-0.2,
                                        z_max=0.3, z_min=0.07,
                                        rx_max=deg2rad(135.0), rx_min=deg2rad(-135.0),
                                        ry_max=deg2rad(20.0), ry_min=deg2rad(-5.0),
                                        rz_max=deg2rad(40.0), rz_min=deg2rad(-40.0))

        self._limits.downward = AttrDict(x_max=0.45, x_min=0.18,
                                         y_max=0.2, y_min=-0.2,
                                         z_max=0.15, z_min=0.04,
                                         rx_max=deg2rad(20.0), rx_min=deg2rad(-20.0),
                                         ry_max=deg2rad(5.0), ry_min=deg2rad(-10.0),
                                         rz_max=deg2rad(90.0), rz_min=deg2rad(-90.0))

        self._iposes.forward = [deg2rad(0), deg2rad(-80), deg2rad(-115), deg2rad(-165), deg2rad(-90), deg2rad(0)]
        self._iposes.downward = [deg2rad(4), deg2rad(-80), deg2rad(-115), deg2rad(-74), deg2rad(-270), deg2rad(180)]

        self._rpy_base.forward = [deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)]
        self._rpy_base.downward = [deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)]

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

    def axis_angle_to_euler(self, axis_angle, euler="XYZ"):
        """
        :param axis_angle: 3-dim axis_angle vector
        :param euler: XYZ, ZYX, ZYZ, etc. specified by euler param
        :return: e.g. roll, pitch, yaw by XYZ, yaw, pitch, roll by ZYX, and so on
        """
        mat = tr.axis_angle_to_matrix(axis_angle if torch.is_tensor(axis_angle) else torch.FloatTensor(axis_angle))
        return tr.matrix_to_euler_angles(mat, euler)

    def switching_control_mode(self):
        idx = self.cmodes.index(self.CONTROL_MODE)
        self.CONTROL_MODE = self.cmodes[(idx + 1) % len(self.cmodes)]
        print("CONTROL_MODE: {} --> {}".format(self.cmodes[idx], self.CONTROL_MODE))

    @property
    def iposes(self):
        return self._iposes[self.CONTROL_MODE]

    @property
    def limits(self):
        return self._limits[self.CONTROL_MODE]

    @property
    def rpy_base(self):
        return self._rpy_base[self.CONTROL_MODE]

    @rpy_base.setter
    def rpy_base(self, rpy):
        self._rpy_base[self.CONTROL_MODE] = rpy


class RealUR3(BaseRTDE, UR3ControlMode):
    def __init__(self):
        self.init_vr()
        BaseRTDE.__init__(self, HOST="192.168.0.75")
        UR3ControlMode.__init__(self, init_mode="forward")

        # fwd
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

        self.timer = CustomTimer(duration_sec=1.0)

        self.rollout = RolloutManager(task_name="pouring_skill")
        self.collect_demo = True

    def init_vr(self):
        self.vr = VRWrapper(device="cpu", rot_d=(-89.9, 0.0, 89.9))

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

    def switching_control_mode(self):
        super().switching_control_mode()
        self.rtde_c.speedStop()
        self.rtde_c.moveJ(self.iposes)
        self.set_rpy_base(log=True)

    def goal_pose(self, des_pos, des_rot):
        """
        * Limiting desired position and rotation to working space
        * Input should be numpy array
        :param des_pos:
        :param des_rot:
        :return: compete goal pose
        """
        des_pos[0] = max(self.limits.x_min, min(self.limits.x_max, des_pos[0]))     # x
        des_pos[1] = max(self.limits.y_min, min(self.limits.y_max, des_pos[1]))     # y
        des_pos[2] = max(self.limits.z_min, min(self.limits.z_max, des_pos[2]))     # z

        _q = tr.axis_angle_to_quaternion(torch.FloatTensor(des_rot))  # VR_Q [w, x, y, z]
        q = quaternion_real_last(q=_q)

        qx_axis = tf_vector(q, self.x_axis)  # refer TCP coordinate
        qy_axis = tf_vector(q, self.y_axis)
        qz_axis = tf_vector(q, self.z_axis)

        if self.CONTROL_MODE == "forward":
            # pitch
            z_xz = self.Pxz @ qz_axis   # projection to plane
            dot_z_x = (z_xz @ self.x_axis) / (z_xz.norm() * self.x_axis.norm())
            pitch = torch.acos(dot_z_x)
            pitch *= 1.0 if z_xz[2] < 0 else -1.0
            pitch = max(self.limits.ry_min, min(self.limits.ry_max, pitch))

            # yaw
            z_xy = self.Pxy @ qz_axis
            dot_z_x = (z_xy @ self.x_axis) / (z_xy.norm() * self.x_axis.norm())
            yaw = torch.acos(dot_z_x)
            yaw *= 1.0 if z_xy[1] > 0 else -1.0
            yaw = max(self.limits.rz_min, min(self.limits.rz_max, yaw))

            # TCP: roll --> Fixed: Pitch
            y_yz = self.Pyz @ qy_axis
            dot_y_z = (y_yz @ self.z_axis) / (y_yz.norm() * self.z_axis.norm())
            roll = torch.acos(dot_y_z)
            roll *= 1.0 if y_yz[1] > 0 else -1.0
            roll = max(self.limits.rx_min, min(self.limits.rx_max, roll))

            _roll, _pitch, _yaw = get_euler_xyz(q=q.unsqueeze(0))    # VR axis-angle --> quaternion --> rpy
            _roll, _pitch, _yaw = list(map(to_n_pi_pi, [_roll, _pitch, _yaw]))

            # ZYX order
            _yaw = yaw + self.rpy_base[2]
            _pitch = roll + self.rpy_base[1]
            _roll = pitch + self.rpy_base[0]
        elif self.CONTROL_MODE == "downward":
            # TODO, set limits for downward mode
            # pitch
            y_xz = self.Pxz @ qy_axis  # projection to plane
            dot_y_x = (y_xz @ self.x_axis) / (y_xz.norm() * self.x_axis.norm())
            pitch = torch.acos(dot_y_x)
            pitch *= 1.0 if y_xz[2] < 0 else -1.0
            pitch = max(self.limits.ry_min, min(self.limits.ry_max, pitch))

            # yaw
            y_xy = self.Pxy @ qy_axis
            dot_y_x = (y_xy @ self.x_axis) / (y_xy.norm() * self.x_axis.norm())
            yaw = torch.acos(dot_y_x)
            yaw *= 1.0 if y_xy[1] > 0 else -1.0
            yaw = max(self.limits.rz_min, min(self.limits.rz_max, yaw))

            # TCP: roll --> Fixed: Pitch
            z_yz = self.Pyz @ qz_axis
            dot_z_nz = (z_yz @ -self.z_axis) / (z_yz.norm() * self.z_axis.norm())
            roll = torch.acos(dot_z_nz)
            roll *= 1.0 if z_yz[1] < 0 else -1.0
            roll = max(self.limits.rx_min, min(self.limits.rx_max, roll))

            _roll, _pitch, _yaw = get_euler_xyz(q=q.unsqueeze(0))  # VR axis-angle --> quaternion --> rpy
            _roll, _pitch, _yaw = list(map(to_n_pi_pi, [_roll, _pitch, _yaw]))

            # ZYX order
            _yaw = yaw + self.rpy_base[2]
            _pitch = roll + self.rpy_base[1]
            _roll = pitch + self.rpy_base[0]
        else:
            raise NotImplementedError

        mat = tr.euler_angles_to_matrix(torch.tensor([_yaw, _pitch, _roll]), "ZYX")
        des_aa = tr.matrix_to_axis_angle(mat)
        des_rot = des_aa.tolist()

        return list(des_pos) + list(des_rot)

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

    def move_grip_on_off_toggle(self):
        self.grip_on = not self.grip_on  # toggle
        grip_pos = 0 if self.grip_on else 100
        self.move_grip_to(pos_in_mm=grip_pos)

    def move_grip_on_off(self, grip_action):
        grip_pos = 0 if grip_action else 100
        self.move_grip_to(pos_in_mm=grip_pos)

    def move_grip_to(self, pos_in_mm):
        assert self.gripper
        max_len = 85.0
        val = min(max(int(pos_in_mm), 0), max_len)  # [0, 50] --> [0/50, 50/50]
        val = int((float(val) / max_len) * 50.0 + 0.5)
        self.gripper.move(val)

    def grip_one_hot_state(self):
        return [int(self.grip_on is True), int(self.grip_on is False)]

    def grip_to_bool(self, grip_one_hot):
        assert sum(grip_one_hot) == 1.0
        return True if grip_one_hot[0] == 1.0 else False

    def cont_mode_one_hot_state(self):
        cm_one_hot = [0] * len(self.cmodes)
        cm_one_hot[self.cmodes_d[self.CONTROL_MODE]] = 1
        return cm_one_hot

    def cont_mode_to_str(self, cm_one_hot):
        idx = [i for i, e in enumerate(cm_one_hot) if round(e) != 0]
        return self.cmodes[idx[0]]

    def set_rpy_base(self, log=False):
        actual_tcp_p = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz], axis-angle orientation
        yaw, pitch, roll = self.axis_angle_to_euler(actual_tcp_p[3:], euler="ZYX")
        self.rpy_base = [roll, pitch, yaw]
        if log:
            print("rpy_base ", self.rpy_base)
            print("euler: ", list(map(rad2deg, [roll, pitch, yaw])))

    def vr_handler(self):
        # int_sec = 0.0
        # int_x = 0.0
        while True:
            cont_status = self.vr.get_controller_status()
            if cont_status["btn_trigger"]:
                pq = cont_status["pose_quat"]
                # pq = torch.cat((pq[-1].unsqueeze(0), pq[:3]))   # real first
                pq = quaternion_real_first(q=torch.tensor(pq))
                aa = tr.quaternion_to_axis_angle(pq)

                if cont_status["btn_trackpad"]:
                    print("btn_trackpad")
                if cont_status["btn_control_mode"]:
                    print("btn_control_mode")
                if cont_status["btn_gripper"]:
                    print("btn_gripper")

                # int_x += cont_status["lin_vel"][0] * cont_status["dt"]
                # # integral
                # int_sec += cont_status["dt"]
                # if int_sec >= 1.0:
                #     print("----------------------")
                #     print("(x) m/s: ", int_x)
                #     print("int_sec ", int_sec)
                #     int_sec = 0.0
                #     int_x = 0.0

                # if cont_status["btn_gripper"]:
                #     self.grip_on = not self.grip_on     # toggle
                #     grip_pos = 63 if self.grip_on else 100
                #     self.move_grip_to(pos_in_mm=grip_pos)
                #     print("gripper toggle")

                # if self.timer.elapsed():
                #     print("VR pose: ", pq)
                #     print("Axis-angle: ", aa)
                #     print("lin_vel: ", cont_status["lin_vel"])
                #     print("dt: ", cont_status["dt"])

    def record_frame(self, action, done):
        state = RobotState(joint=self.rtde_r.getActualQ(),
                           gripper=self.grip_one_hot_state(),
                           control_mode=self.cont_mode_one_hot_state())
        info = str({"gripper": self.grip_on, "control_mode": self.CONTROL_MODE})
        self.rollout.append(state=state, action=action, done=done, info=info)

    def play_demo(self):
        # go to initial state in joint space
        init_state, _, _, _ = self.rollout.get(0)
        self.rtde_c.moveJ(init_state.joint)

        # loop for playing demo
        for idx in range(1, self.rollout.len()):
            start_t = self.rtde_c.initPeriod()
            state, action, done, info = self.rollout.get(index=idx)

            if self.cont_mode_to_str(state.control_mode) != self.CONTROL_MODE:
                self.switching_control_mode()
            if self.grip_to_bool(state.gripper) ^ self.grip_on:
                self.move_grip_on_off_toggle()

            goal_j = self.rtde_c.getInverseKinematics(x=action)
            actual_j = self.rtde_r.getActualQ()
            diff_j = (np.array(goal_j) - np.array(actual_j)) * 0.5
            self.rtde_c.speedJ(list(diff_j), self.acc, self.dt)
            self.rtde_c.waitPeriod(start_t)
        self.rtde_c.speedStop()

    def run_vr_teleop(self):
        print("Run VR teleoperation mode")
        try:
            self.rtde_c.moveJ(self.iposes)
            self.set_rpy_base()
            if self.collect_demo: self.record_frame(action=self.rtde_r.getActualTCPPose(), done=0)

            while True:
                start_t = self.rtde_c.initPeriod()

                # get velocity command from VR
                cont_status = self.vr.get_controller_status()
                if cont_status["btn_reset_pose"]:
                    self.record_frame(action=self.rtde_r.getActualTCPPose(), done=1)
                    self.rtde_c.speedStop()
                    self.rtde_c.moveJ(self.iposes)
                    # self.play_demo()
                    self.rollout.show_current_rollout_info()
                    self.rollout.save_to_file()
                    self.rollout.reset()
                    continue

                rot = [0.0, 0.0, 0.0]
                diff_j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                if cont_status["btn_trigger"]:
                    if cont_status["btn_control_mode"]:
                        self.switching_control_mode()

                    if cont_status["btn_gripper"]:
                        self.move_grip_on_off_toggle()

                    vr_q = torch.tensor(cont_status["pose_quat"])
                    vr_q = quaternion_real_first(q=vr_q)
                    vr_a = tr.quaternion_to_axis_angle(vr_q)           # desired aa pose by VR

                    actual_tcp_p = self.rtde_r.getActualTCPPose()
                    actual_j = self.rtde_r.getActualQ()

                    d_pos = np.array(actual_tcp_p[:3]) + cont_status["lin_vel"]
                    d_rot = vr_a.numpy()
                    goal_pose = self.goal_pose(des_pos=d_pos, des_rot=d_rot)    # limit handling

                    if self.collect_demo: self.record_frame(action=goal_pose, done=0)

                    goal_j = self.rtde_c.getInverseKinematics(x=goal_pose)
                    diff_j = (np.array(goal_j) - np.array(actual_j)) * 0.5

                    if self.timer.timeover_active:
                        print("Desired AA: ", vr_q, vr_q.norm())
                        print("Actual AA: ", actual_tcp_p[3:])
                        print("diff: ", rot)
                        print("dt ", cont_status["dt"])
                        # print("TCP: ", actual_tcp_p)
                        # print("des pose: ", des_pose)
                        # print("dj: ", dj)
                        print("----------------------------------")

                self.rtde_c.speedJ(list(diff_j), self.acc, self.dt)
                self.rtde_c.waitPeriod(start_t)
        except ValueError:
            print("Value Error... ")
        finally:
            print("end of control... ")
            if not hasattr(self, "rtde_c"): return
            self.rtde_c.speedStop()
            print("speed stop.. ")
            self.rtde_c.stopScript()
            print("script stop.. ")

    def replay_mode(self):
        self.rtde_c.moveJ(self.iposes)
        self.set_rpy_base()
        while True:
            cont_status = self.vr.get_controller_status()
            if cont_status["btn_reset_pose"]:
                print("reset & replay")
                self.rollout.load_from_file(batch_idx=1, rollout_idx=6)
                self.rollout.show_current_rollout_info()
                self.play_demo()

    def func_test(self):
        self.rollout.load_from_file(batch_idx=1, rollout_idx=6)
        self.rollout.show_current_rollout_info()
        for i in range(self.rollout.len()):
            state, action, done, info = self.rollout.get(index=i)
            print(state, self.grip_to_bool(state.gripper), self.cont_mode_to_str(state.control_mode))
        return
        init_joint = [deg2rad(8.5), deg2rad(-102), deg2rad(-108),
                      deg2rad(-150), deg2rad(-82), deg2rad(0)]
        self.rtde_c.moveJ(init_joint)
        goal_pose = [0.59, 0.065, 0.154, 1.02, 1.353, 1.481]

        try:
            j_spd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.02]
            for i in range(1000):
                start_t = self.rtde_c.initPeriod()
                goal_j = self.rtde_c.getInverseKinematics(x=goal_pose)
                actual_j = self.rtde_r.getActualQ()
                scale = 0.1
                diff_j = np.array(goal_j) - np.array(actual_j)
                self.rtde_c.speedJ(list(diff_j * scale), 1.2, 1.0 / 500)
                if self.timer.elapsed():
                    print(i, j_spd)
                    # print("start_t ", start_t)
                    print("actual_j: ", actual_j)
                    print("goal_j: ", goal_j)
                    print("diff_j: ", diff_j * scale)
                self.rtde_c.waitPeriod(start_t)
        finally:
            print("prgram end")
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()


if __name__ == "__main__":
    u = RealUR3()
    # u.vr_handler()
    # u.workspace_verify()
    # u.run_vr_teleop()
    # u.replay_mode()
    u.func_test()
