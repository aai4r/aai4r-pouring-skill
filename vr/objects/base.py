import isaacgym
from utils.utils import *
from isaacgym import gymutil
from isaacgym import gymapi, gymtorch
import torch
import json


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