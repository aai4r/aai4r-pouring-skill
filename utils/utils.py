import numpy as np
import torch
from utils.torch_jit_utils import quat_conjugate, quat_mul
from dataclasses import dataclass

PI = np.pi


def rad2deg(rad):
    return rad * (180.0 / PI)


def deg2rad(deg):
    return deg * (PI / 180.0)


def quat_to_mat(q):
    sqw = q[:, 3] * q[:, 3]
    sqx = q[:, 0] * q[:, 0]
    sqy = q[:, 1] * q[:, 1]
    sqz = q[:, 2] * q[:, 2]

    # invs(inverse square length) is only required if quaternion is not already normalised
    m = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=torch.float)
    invs = 1 / (sqx + sqy + sqz + sqw)
    m[:, 0, 0] = (sqx - sqy - sqz + sqw) * invs    # since sqw + sqx + sqy + sqz = 1 / invs * invs
    m[:, 1, 1] = (-sqx + sqy - sqz + sqw) * invs
    m[:, 2, 2] = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = q[:, 0] * q[:, 1]
    tmp2 = q[:, 2] * q[:, 3]
    m[:, 1, 0] = 2.0 * (tmp1 + tmp2) * invs
    m[:, 0, 1] = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = q[:, 0] * q[:, 2]
    tmp2 = q[:, 1] * q[:, 3]
    m[:, 2, 0] = 2.0 * (tmp1 - tmp2) * invs
    m[:, 0, 2] = 2.0 * (tmp1 + tmp2) * invs
    tmp1 = q[:, 1] * q[:, 2]
    tmp2 = q[:, 0] * q[:, 3]
    m[:, 2, 1] = 2.0 * (tmp1 + tmp2) * invs
    m[:, 1, 2] = 2.0 * (tmp1 - tmp2) * invs
    return m


def mat_to_quat(m):
    w = torch.sqrt(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]) / 2.0
    w4 = (4.0 * w)
    x = (m[:, 2, 1] - m[:, 1, 2]) / w4
    y = (m[:, 0, 2] - m[:, 2, 0]) / w4
    z = (m[:, 1, 0] - m[:, 0, 1]) / w4
    return torch.stack([x, y, z, w], dim=-1).to(m.device)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


from isaacgym.torch_utils import *


@dataclass
class ViaPointProperty:
    pos: float = 5.e-2
    rot: float = 5.e-2
    grip: float = 7.e-3
    wait: float = 0
    tag: str = ""


class TaskPoseList:
    def __init__(self, task_name, num_envs, device):
        self.task_name = task_name
        self.num_envs = num_envs
        self.device = device

        self.pos = []
        self.rot = []
        self.grip = []
        self.err_thres = []

    def gen_pos_variation(self, pivot_pos, pos_var_meter=0.02):
        _pos = to_torch(pivot_pos, device=self.device).repeat((self.num_envs, 1))
        var = (torch.rand_like(_pos) - 0.5) * 2.0
        var_pos = _pos + var * pos_var_meter
        return var_pos

    def gen_rot_variation(self, pivot_quat, rot_var_deg=15):
        _rot = to_torch(pivot_quat, device=self.device).repeat((self.num_envs, 1))
        roll_d = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        pitch_d = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        yaw_d = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg

        q_var = quat_from_euler_xyz(roll=deg2rad(roll_d), pitch=deg2rad(pitch_d), yaw=deg2rad(yaw_d))
        var_rot = quat_mul(_rot, q_var)
        return var_rot

    def gen_grip_variation(self, pivot_grip, grip_var_meter=0.01):
        _grip = to_torch(pivot_grip, device=self.device).repeat((self.num_envs, 1))  # meter
        grip_var = (torch.rand_like(_grip) - 0.5) * 2.0
        var_grip = _grip + grip_var * grip_var_meter
        return var_grip

    def append_pose(self, pos, rot, grip, err=ViaPointProperty()):
        assert len(pos) == len(rot) == len(grip) == self.num_envs
        self.pos.append(pos)
        self.rot.append(rot)
        self.grip.append(grip)
        self.err_thres.append(err)

    def pose_pop(self, index=-1):
        return self.pos.pop(index), self.rot.pop(index), self.grip.pop(index), self.err_thres.pop(index)

    def length(self):
        return len(self.pos)


class TaskPathManager:
    def __init__(self, num_env, num_task_steps, device):
        self.num_env = num_env
        self.num_task_steps = num_task_steps
        self.device = device

        # __step shape: [num_env, 1, dim=([3 , 4 , 1])]
        self._step = torch.zeros(num_env, device=device, dtype=torch.long).unsqueeze(-1).repeat(1, 4).unsqueeze(1)
        self._push_idx = torch.zeros(num_env, device=device, dtype=torch.long)

        """
        (num_env, task_steps, dim)
        """
        self._task_pos = torch.zeros(num_env, num_task_steps, 3, device=device)
        self._task_rot = torch.zeros(num_env, num_task_steps, 4, device=device)
        self._task_grip = torch.zeros(num_env, num_task_steps, 1, device=device)
        self._task_err = torch.zeros(num_env, num_task_steps, 4, device=device)    # (pos_err, rot_err, grip_err, wait)

        # time properties
        self.curr_time = 0
        self.prev_time = 0
        self.elapsed = torch.zeros(num_env, device=device)

        self.init_joint_pos = None

    def set_init_joint_pos(self, joint, noise=0.05):
        assert torch.is_tensor(joint)
        self.init_joint_pos = joint.repeat((self.num_env, 1)) + \
                              (torch.rand(self.num_env, joint.numel(), device=self.device) - 0.5) * noise

    def reset_task(self, env_ids):
        self._task_pos[env_ids] = torch.zeros_like(self._task_pos[env_ids])
        self._task_rot[env_ids] = torch.zeros_like(self._task_rot[env_ids])
        self._task_grip[env_ids] = torch.zeros_like(self._task_grip[env_ids])
        self._task_err[env_ids] = torch.zeros_like(self._task_err[env_ids])
        self._push_idx[env_ids] = torch.zeros_like(self._push_idx[env_ids])
        self._step[env_ids] = torch.zeros_like(self._step[env_ids])

    def push_pose(self, env_ids, pos, rot, grip, err=ViaPointProperty()):
        self._task_pos[env_ids, self._push_idx[env_ids]] = pos[env_ids]
        self._task_rot[env_ids, self._push_idx[env_ids]] = rot[env_ids]
        self._task_grip[env_ids, self._push_idx[env_ids]] = grip[env_ids]

        err_th = torch.tensor([err.pos, err.rot, err.grip, err.wait], device=self.device).repeat(self.num_env, 1)
        self._task_err[env_ids, self._push_idx[env_ids]] = err_th[env_ids]
        self._push_idx[env_ids] += 1

    def update_step_by_checking_arrive(self, ee_pos, ee_rot, ee_grip, dof_pos, sim_time):
        des_pos, des_rot, des_grip, err_th = self.get_desired_pose()
        err_pos = (des_pos - ee_pos).norm(dim=-1)
        err_rot = orientation_error(des_rot, ee_rot).norm(dim=-1)
        err_grip = (des_grip - ee_grip).norm(dim=-1)
        err_joint = (self.init_joint_pos - dof_pos).norm(dim=-1)

        reach_jnt = err_joint < 0.1
        reach_ee = (err_pos < err_th[:, 0]) & (err_rot < err_th[:, 1]) & (err_grip < err_th[:, 2])
        reach = torch.where(self._step[:, 0, 0] == 0, reach_jnt, reach_ee)

        self.curr_time = sim_time
        dt = self.curr_time - self.prev_time
        self.elapsed += torch.where(reach, dt, 0.0)
        self.elapsed = torch.where(self.elapsed > err_th[:, -1], torch.zeros_like(self.elapsed), self.elapsed)
        self.prev_time = self.curr_time

        arrive = torch.where(reach & (self.elapsed <= 0), 1, 0)

        self._step += arrive.unsqueeze(-1).repeat(1, 4).unsqueeze(-2)
        # if arrive.sum() > 0:
        #     print("arrive: ", arrive, self.num_task_steps)
        #     print("step: ", self.__step)
        done_envs = torch.where(self._step[:, 0, 0] >= self.num_task_steps, 1, 0)
        self._step = torch.where(self._step >= self.num_task_steps, torch.zeros_like(self._step), self._step)
        return done_envs

    def get_desired_pose(self):
        pos = torch.gather(self._task_pos, 1, self._step[:, :, :3]).squeeze(-2)
        rot = torch.gather(self._task_rot, 1, self._step[:, :, :]).squeeze(-2)
        grip = torch.gather(self._task_grip, 1, self._step[:, :, :1]).squeeze(-2)
        err_th = torch.gather(self._task_err, 1, self._step[:, :, :4]).squeeze(-2)
        return pos, rot, grip, err_th

    def print_task_status(self):
        print("================== Task Info. ==================")
        print("Num_env: {}, Num_task_steps: {}, Device: {}".format(self.num_env, self.num_task_steps, self.device))
        print("[Shapes] Pos: {}, *** Rot: {}, *** Grip: {}".format(self._task_pos.shape, self._task_rot.shape, self._task_grip.shape))
        print("[Data] Task Pos: \n{}".format(self._task_pos))
        print("[Data] Task Rot: \n{}".format(self._task_rot))
        print("[Data] Task Grip: \n{}".format(self._task_grip))
