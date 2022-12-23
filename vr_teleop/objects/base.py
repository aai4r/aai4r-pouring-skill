import copy
import time

import isaacgym
from utils.utils import *
from isaacgym import gymutil
from isaacgym import gymapi, gymtorch
import torch
import json

from vr_teleop import triad_openvr


class JsonTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonTypeEncoder, self).default(obj)


class IsaacElement:
    def __init__(self, gym, viewer, sim, env, num_envs, device, asset_root):
        """

        :param gym:
        :param viewer:
        :param sim:
        :param env:
        :param device:
        :param asset_root
        """
        self.gym = gym
        self.viewer = viewer
        self.sim = sim
        self.env = env
        self.num_envs = num_envs
        self.device = device
        self.asset_root = asset_root


class ButtonPressedEventHandler:
    def __init__(self):
        self._curr = None
        self._prev = None

    def is_pressed(self, event_stream):
        btn_pressed = False
        self._curr = 0 if event_stream else 1

        if self._prev is not None:
            if (self._curr - self._prev) < 0:
                btn_pressed = True
        self._prev = self.clone(self._curr)
        return btn_pressed

    def clone(self, val):
        return copy.deepcopy(val)


class VRWrapper:
    def __init__(self, device, rot_d=(0.0, 0.0, 0.0)):
        assert len(rot_d) == 3 and isinstance(rot_d, tuple)
        self.device = device
        self.vr = triad_openvr.triad_openvr()
        self.rot = torch.inverse(euler_to_mat3d(deg2rad(rot_d[0]), deg2rad(rot_d[1]), deg2rad(rot_d[2])))
        print("rot: \n", self.rot)

        self.trk_btn = ButtonPressedEventHandler()
        self.menu_btn = ButtonPressedEventHandler()
        self.mode_btn = ButtonPressedEventHandler()
        self.prev_time = time.time()

    def get_controller_status(self):
        d = self.vr.devices["controller_1"].get_controller_inputs()
        lv = np.array([0.0, 0.0, 0.0])    # linear velocity of controller
        av = np.array([0.0, 0.0, 0.0])    # angular velocity of controller
        pq = np.array([0.0, 0.0, 0.0, 1.0])    # pose quaternion

        # button status
        controller_status = {"lin_vel": lv, "ang_vel": av, "pose_quat": pq,
                             "btn_trigger": False, "btn_gripper": False, "btn_reset_pose": False,
                             "trk_x": None, "trk_y": None,
                             "dt": time.time() - self.prev_time}
        self.prev_time = time.time()
        if d['trigger']:
            lv = np.array([v for v in self.vr.devices["controller_1"].get_velocity()]) * 1.0    # m/s
            av = np.array([v for v in self.vr.devices["controller_1"].get_angular_velocity()]) * 1.0  # incremental
            pq = np.array([v for v in self.vr.devices["controller_1"].get_pose_quaternion()])         # absolute

            lv = torch.matmul(self.rot, torch.tensor(lv).unsqueeze(0).T).T.squeeze(0).numpy()
            av = torch.tensor(av).unsqueeze(0)
            _rq = mat_to_quat(self.rot.unsqueeze(0))
            av = quat_apply(_rq, av).squeeze(0).numpy()

            _pq = torch.tensor(pq[3:]).unsqueeze(0)
            _pq[0, 2] *= -1.0   # Left-handed --> Right-handed
            # pq = _pq.squeeze(0).numpy()  # TODO, pure rotation of controller
            _rq_pre = mat_to_quat(euler_to_mat3d(deg2rad(90.0), deg2rad(0.0), deg2rad(0.0)).unsqueeze(0))
            _rq_post = mat_to_quat(euler_to_mat3d(deg2rad(-90.0), deg2rad(0.0), deg2rad(-90.0)).unsqueeze(0))
            pq = quat_mul(_rq_pre, quat_mul(_pq, _rq_post)).squeeze(0).numpy()

            controller_status["lin_vel"] = lv
            controller_status["ang_vel"] = av
            controller_status["pose_quat"] = pq
            controller_status["btn_trigger"] = True
            controller_status["btn_trackpad"] = self.trk_btn.is_pressed(event_stream=d["trackpad_pressed"])
            x, y = d["trackpad_x"], d["trackpad_y"]
            controller_status["trk_x"] = x
            controller_status["trk_y"] = y
            controller_status["btn_gripper"] = controller_status["btn_trackpad"] and (-0.3 < x) and (x < 0.3) and (y < -0.6)
            controller_status["btn_control_mode"] = controller_status["btn_trackpad"] and (-0.3 < x) and (x < 0.3) and (0.6 < y)
            controller_status["btn_reset_pose"] = self.menu_btn.is_pressed(event_stream=d["menu_button"])
        return controller_status


class VRElement:
    def __init__(self, vr, rot):
        """
        :param vr: instance for VR tele-operation
        :param rot: for VR controller calibration
        """
        self.vr = vr
        self.rot = rot

        self.trk_btn_trans = []
        self.trk_btn_toggle = 1


class BaseObject:
    def __init__(self, isaac_elem):
        assert type(isaac_elem) is IsaacElement

        self.gym = isaac_elem.gym
        self.viewer = isaac_elem.viewer
        self.sim = isaac_elem.sim
        self.env = isaac_elem.env
        self.num_envs = isaac_elem.num_envs
        self.device = isaac_elem.device
        self.asset_root = isaac_elem.asset_root
        self._create()

    def _create(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def vr_handler(self):
        raise NotImplementedError

    def draw_coord(self, pos, rot, scale=0.2):     # args type: numpy arrays
        self.gym.clear_lines(self.viewer)
        for p, r in zip(pos, rot):
            pos = torch.tensor(p, device=self.device, dtype=torch.float32)
            rot = torch.tensor(r, device=self.device, dtype=torch.float32)
            px = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device, dtype=torch.float32) * scale)).cpu().numpy()
            py = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device, dtype=torch.float32) * scale)).cpu().numpy()
            pz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device, dtype=torch.float32) * scale)).cpu().numpy()

            p0 = pos.cpu().numpy()
            self.gym.add_lines(self.viewer, self.env, 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
            self.gym.add_lines(self.viewer, self.env, 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
            self.gym.add_lines(self.viewer, self.env, 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])