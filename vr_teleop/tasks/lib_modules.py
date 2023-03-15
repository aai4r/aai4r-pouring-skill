import copy

from utils.torch_jit_utils import *
from pytorch3d import transforms as tr

from utils.utils import rad2deg, deg2rad, quaternion_real_first, quaternion_real_last, CustomTimer

import rtde_control
import rtde_receive
from vr_teleop.gripper.robotiq_gripper_control import RobotiqGripper, RobotiqGripperExpand

from spirl.utility.general_utils import AttrDict


def to_n_pi_pi(rad):  # [0, 2*pi] --> [-pi, pi]
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
        self.gripper = RobotiqGripperExpand(self.rtde_c, self.HOST)
        self.gripper.activate()
        self.gripper.set_force(0)  # range: [0, 100]
        self.gripper.set_speed(10)  # range: [0, 100]
        self.grip_on = None
        self.move_grip_on_off(grip_action=False)

        self.default_control_params = AttrDict(speed=0.25, acceleration=1.2, blend=0.099, dt=1.0 / 500.0)

    def init_period(self):
        return self.rtde_c.initPeriod()

    def wait_period(self, t):
        self.rtde_c.waitPeriod(t)

    def get_actual_tcp_pose(self):
        return self.rtde_r.getActualTCPPose()

    def get_actual_tcp_pos_ori(self):
        pose = self.rtde_r.getActualTCPPose()
        return pose[:3], pose[3:]

    def get_actual_q(self):
        return self.rtde_r.getActualQ()

    def get_inverse_kinematics(self, tcp_pose):
        try:
            ik_joint = self.rtde_c.getInverseKinematics(x=tcp_pose)
            if len(ik_joint) == 0:
                print("IK exception..! input tcp pose => ", tcp_pose)
                ik_joint = self.get_actual_q()
                # raise ValueError
            return ik_joint
        except ValueError:
            print("IK exception..! input tcp pose => ", tcp_pose)

    def speed_j(self, des_j, acc, dt):
        self.rtde_c.speedJ(des_j, acc, dt)

    def speed_stop(self):
        self.rtde_c.speedStop()

    def move_j(self, joint):
        self.rtde_c.moveJ(joint)

    def stop_script(self):
        self.rtde_c.stopScript()

    def move_grip_on_off_toggle(self):
        self.grip_on = not self.grip_on  # toggle
        grip_pos = 0 if self.grip_on else 100
        self.move_grip_to(pos_in_mm=grip_pos)

    def move_grip_on_off(self, grip_action):
        assert type(grip_action) == bool
        if self.grip_on is not None:
            if grip_action == self.grip_on: return
        grip_pos = 0 if grip_action else 100
        self.move_grip_to(pos_in_mm=grip_pos)
        self.grip_on = grip_action

    def move_grip_to(self, pos_in_mm):
        assert self.gripper
        max_len = 85.0
        val = min(max(int(pos_in_mm), 0), max_len)  # [0, 50] --> [0/50, 50/50]
        val = int((float(val) / max_len) * 50.0 + 0.5)
        self.gripper.move(val)

    def grip_one_hot_state(self):
        return [int(self.grip_on is True), int(self.grip_on is False)]

    def grip_pos(self, normalize=True, list_type=True):
        pos = self.gripper.gripper_to_mm_normalize() if normalize else self.gripper.gripper_to_mm()
        return [pos] if list_type else pos

    @staticmethod
    def grip_onehot_to_bool(grip_one_hot):
        assert sum(grip_one_hot) == 1.0
        return True if grip_one_hot[0] == 1.0 else False

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

        self._rpy_base.forward = [deg2rad(89.9), deg2rad(0.0), deg2rad(89.9)]
        self._rpy_base.downward = [deg2rad(179.9), deg2rad(0.0), deg2rad(89.9)]

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

    def set_rpy_base(self, actual_tcp_pose, log=False):
        # actual_tcp_p = self.get_actual_tcp_pose()  # [x, y, z, rx, ry, rz], axis-angle orientation
        yaw, pitch, roll = self.axis_angle_to_euler(actual_tcp_pose[3:], euler="ZYX")
        self.rpy_base = [roll, pitch, yaw]
        if log:
            print("rpy_base ", self.rpy_base)
            print("euler: ", list(map(rad2deg, [roll, pitch, yaw])))

    def cont_mode_one_hot_state(self):
        cm_one_hot = [0] * len(self.cmodes)
        cm_one_hot[self.cmodes_d[self.CONTROL_MODE]] = 1
        return cm_one_hot

    def cont_mode_to_str(self, cm_one_hot):
        idx = [i for i, e in enumerate(cm_one_hot) if round(e) != 0]
        return self.cmodes[idx[0]]

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

    def goal_pose(self, des_pos, des_rot):
        """
        * Limiting desired position and rotation to working space
        * Input should be numpy array
        :param des_pos:
        :param des_rot:
        :return: compete goal pose
        """
        des_pos[0] = max(self.limits.x_min, min(self.limits.x_max, des_pos[0]))  # x
        des_pos[1] = max(self.limits.y_min, min(self.limits.y_max, des_pos[1]))  # y
        des_pos[2] = max(self.limits.z_min, min(self.limits.z_max, des_pos[2]))  # z

        _q = tr.axis_angle_to_quaternion(torch.FloatTensor(des_rot))  # VR_Q [w, x, y, z]
        q = quaternion_real_last(q=_q)

        qx_axis = tf_vector(q, self.x_axis)  # refer TCP coordinate
        qy_axis = tf_vector(q, self.y_axis)
        qz_axis = tf_vector(q, self.z_axis)

        if self.CONTROL_MODE == "forward":
            # pitch
            z_xz = self.Pxz @ qz_axis  # projection to plane
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

            _roll, _pitch, _yaw = get_euler_xyz(q=q.unsqueeze(0))  # VR axis-angle --> quaternion --> rpy
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
        # print("des_pos: {}, des_rot: {}".format(des_pos, des_rot))

        return list(des_pos) + list(des_rot)

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

    @staticmethod
    def add_noise_angle(inputs, deg=5):
        print("inputs: ", inputs)
        _inputs = np.array(inputs) + deg2rad(deg) * (np.random.rand(len(inputs)) - 0.5) * 2.0
        print("_inputs: ", _inputs)
        return _inputs

    @staticmethod
    def goal_axis_angle_from_act_quat(act_quat, actual_tcp_aa):
        if type(act_quat) == list or type(act_quat) == np.ndarray:
            act_quat = torch.tensor(act_quat).clone()
            act_quat = act_quat / act_quat.norm()  # normalize

        if type(actual_tcp_aa) == list or type(actual_tcp_aa) == np.ndarray:
            actual_tcp_aa = tr.axis_angle_to_quaternion(torch.tensor(actual_tcp_aa).clone())
            actual_tcp_aa = quaternion_real_last(actual_tcp_aa)

        goal_rot_quat = quat_mul(act_quat, actual_tcp_aa)
        goal_rot_quat = quaternion_real_first(q=goal_rot_quat)
        goal_rot_aa = tr.quaternion_to_axis_angle(goal_rot_quat)
        return goal_rot_aa.numpy()

    @staticmethod
    def quat_from_tcp_axis_angle(actual_tcp_aa, tolist=True):
        assert type(actual_tcp_aa) == list or type(actual_tcp_aa) == np.ndarray
        actual_tcp_aa = tr.axis_angle_to_quaternion(torch.tensor(actual_tcp_aa).clone())
        actual_tcp_aa = quaternion_real_last(actual_tcp_aa)
        return actual_tcp_aa.tolist() if tolist else actual_tcp_aa


import pyrealsense2 as rs
import numpy as np
import cv2


def visualize(depth_image, color_image, disp_name=None):
    """
    :param disp_name: name of cv2.namedWindow
    :param depth_image: (h, w, c), uint16
    :param color_image: (h, w, c), uint8
    :return:
    """
    if depth_image is None or color_image is None:
        print("Can't get a frame....")
        return cv2.waitKey(1)
    _depth_image = copy.deepcopy(depth_image)
    _color_image = copy.deepcopy(color_image)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(_depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = _color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(_color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((_color_image, depth_colormap))

    # Show images
    if disp_name:
        cv2.imshow(disp_name, images)
    else:
        wnd_name = 'RealSense D435'
        cv2.namedWindow(wnd_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(wnd_name, images)
    return cv2.waitKey(1)


class RealSense:
    def __init__(self, args=None):
        # Configure depth and color streams
        """
        Currently, each cam is used as following:
            [front] cam is used for recording evaluation process
            [rear] cam is used for observing the environment
        """
        self.cam_id = AttrDict(front='832412070289', rear='832412070267')
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.cam_id.rear)
        self.args = args if args is not None else AttrDict(width=640, height=480, fps=30)

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            raise ConnectionError

        self.config.enable_stream(rs.stream.depth, self.args.width, self.args.height, rs.format.z16, self.args.fps)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, self.args.fps)
        else:
            self.config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)

        # Start streaming
        self.pipeline.start(self.config)

    def display_info(self):
        depth, color = self.get_np_images()
        if depth is None or color is None:
            raise ValueError

        print("====================")
        print("-Vision Information-")
        print("====================")
        print("* Image Stream")
        print("    Width: {}, Height: {}, FPS: {}".format(self.args.width, self.args.height, self.args.fps))
        print("* Actual Data")
        print("    Depth shape: {}, min / max: {} / {}, dtype: {}".format(depth.shape, depth.min(), depth.max(),
                                                                          depth.dtype))
        print("    Color shape: {}, min / max: {} / {}, dtype: {}".format(color.shape, color.min(), color.max(),
                                                                          color.dtype))

    def stop_stream(self):
        self.pipeline.stop()

    def get_np_images(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())  # (h, w, 1)
        color_image = np.asanyarray(color_frame.get_data())  # (h, w, 3)
        return depth_image, color_image


def noisy(image, noise_type='gauss', random_noise=False):
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if random_noise:
        noise_list = ['gauss', 's&p', 'poisson', 'speckle']
        noise_type = noise_list[np.random.randint(0, len(noise_list))]
    if noise_type == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.001
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
    elif noise_type == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.005
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1], coords[2]] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords[0], coords[1], coords[2]] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
    elif noise_type == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * 0.15     # scaled

    if noisy.dtype == np.float32:
        noisy = (np.clip(noisy, 0., 1.) * 255.0).astype(np.uint8)
    return noisy
