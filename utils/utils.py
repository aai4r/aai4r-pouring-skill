import numpy as np
import torch

PI = 3.1415926535


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


class TaskPathManager:
    def __init__(self, num_env, num_task_steps, device):
        self.num_env = num_env
        self.num_task_steps = num_task_steps
        self.device = device

        # __step shape: [num_env, 1, dim=(3 / 4 / 1)]
        step = torch.zeros(num_env, device=device, dtype=torch.long)
        self.__step = step.unsqueeze(-1).repeat(1, 4).unsqueeze(1)
        self.__push_idx = 0

        self.err_thres = {"pos": 1.e-2, "rot": 1.e-2, "grip": 5.e-3}

        self.__task_pos = torch.zeros(num_env, num_task_steps, 3, device=device)
        self.__task_rot = torch.zeros(num_env, num_task_steps, 4, device=device)
        self.__task_grip = torch.zeros(num_env, num_task_steps, 1, device=device)

    def reset_task(self, env_ids):
        self.__push_idx = 0
        self.__step[env_ids] = torch.zeros(len(env_ids), device=self.device)

    def push_task_pose(self, env_ids, pos, rot, grip):
        if self.__push_idx >= self.num_task_steps:
            raise ValueError('Task index is out of bound.. Index {} should be less than {}'
                             .format(self.__push_idx, self.num_task_steps))
        self.__task_pos[env_ids, self.__push_idx] = pos[env_ids]
        self.__task_rot[env_ids, self.__push_idx] = rot[env_ids]
        self.__task_grip[env_ids, self.__push_idx] = grip[env_ids]
        self.__push_idx += 1

    def check_arrive(self, ee_pos, ee_rot, ee_grip):
        des_pos, des_rot, des_grip = self.get_pose()
        err_pos = (des_pos - ee_pos).norm(dim=-1)
        err_rot = (des_rot - ee_rot).norm(dim=-1)
        err_grip = (des_grip - ee_grip).norm(dim=-1)

        arrive = torch.where((err_pos < self.err_thres["pos"]) &
                             (err_rot < self.err_thres["rot"]) &
                             (err_grip < self.err_thres["grip"]), 1, 0)

        self.__step += arrive.unsqueeze(-1).repeat(1, 1, 4)

    def get_pose(self):
        pos = torch.gather(self.__task_pos, 1, self.__step[:, :, :3])
        rot = torch.gather(self.__task_rot, 1, self.__step[:, :, :])
        grip = torch.gather(self.__task_grip, 1, self.__step[:, :, :1])
        return pos, rot, grip

    def print_task_status(self):
        print("================== Task Info. ==================")
        print("Num_env: {}, Num_task_steps: {}, Device: {}".format(self.num_env, self.num_task_steps, self.device))
        print("[Shapes] Pos: {}, *** Rot: {}, *** Grip: {}".format(self.__task_pos.shape, self.__task_rot.shape, self.__task_grip.shape))
        print("[Data] Task Pos: \n{}".format(self.__task_pos))
        print("[Data] Task Rot: \n{}".format(self.__task_rot))
        print("[Data] Task Grip: \n{}".format(self.__task_grip))
