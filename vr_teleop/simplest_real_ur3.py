import socket
import time
import json
import numpy as np

import rtde_control
import rtde_receive
from gripper.robotiq_gripper_control import RobotiqGripper
import isaacgym
from utils.utils import orientation_error


def rad2deg(rad): return rad * (180.0 / np.pi)


def quat_to_real_last(q_real_first):
    return torch.cat((q_real_first[1:], q_real_first[0].unsqueeze(0)))     # [x, y, z, w]


def gripper_test():
    print("Robotiq Gripper Test")

    HOST = "192.168.0.75"  # remote host
    rtde_c = rtde_control.RTDEControlInterface(HOST)
    gripper = RobotiqGripper(rtde_c)

    # Activate the gripper and initialization
    gripper.activate()
    gripper.set_force(0)   # from 0 to 100
    gripper.set_speed(10)  # from 0 to 100

    try:
        max_len = 85.0
        while True:
            val = input('Desired gripper position: ')
            if val == 'q': break
            if not val.isnumeric(): continue
            val = min(max(int(val), 0), max_len)    # [0, 50] --> [0/50, 50/50]
            val = int((float(val) / max_len) * 50.0 + 0.5)
            print("actual value: ", val)
            gripper.move(val)
        # perform gripper actions
        # gripper.close()
        # # gripper.open()
        # gripper.move(20)  # mm
        # time.sleep(1)
    finally:
        rtde_c.servoStop()
        rtde_c.stopScript()
        print("Control Script End")


def real_ur3_rtde_test():
    print("RTDE test")
    HOST = "192.168.0.75"  # remote host

    # Parameters
    velocity = 0.5
    acceleration = 0.5
    dt = 1.0 / 500  # 2ms
    lookahead_time = 0.1
    gain = 300
    joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]

    rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
    rtde_c = rtde_control.RTDEControlInterface(HOST)
    try:
        # Move to initial joint position with a regular moveJ
        rtde_c.moveJ(joint_q)

        # Execute 500Hz control loop for 2 seconds, each cycle is 2ms
        for i in range(1000):
            actual_q = rtde_r.getActualQ()
            actual_tcp_p = rtde_r.getActualTCPPose()
            print("actual_tcp_p: ", actual_tcp_p)
            t_start = rtde_c.initPeriod()
            rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)
            joint_q[0] += 0.001
            joint_q[1] += 0.001
            rtde_c.waitPeriod(t_start)
    finally:
        print("Control Script End")
        rtde_c.servoStop()
        rtde_c.stopScript()


def real_ur3_socket_test():
    print("Real UR3 Test!")

    HOST = "192.168.0.75"   # remote host
    PORT = 30002            # same port number with server

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    cmd = "see_digital_out(2, True)" + "\n"
    # cmd = "get_actual_joint_positions()" + "\n"
    # return: 6-dim vector of radian angles [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
    s.send(cmd.encode('utf-8'))
    time.sleep(0.1)
    data = s.recv(1024)
    s.close()
    print("Raw: ", type(data), data)
    print(data.decode())


from utils.torch_jit_utils import *
import torch
from pytorch3d import transforms as tr


def r2d(rad):
    return rad * (180.0 / np.pi)


def d2r(deg):
    return deg * (np.pi / 180.0)


def rv2rpy(rx, ry, rz):
    theta = np.sqrt(rx * rx + ry * ry + rz * rz)
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta
    cth = np.cos(theta)
    sth = np.sin(theta)
    vth = 1 - np.cos(theta)

    r11 = kx * kx * vth + cth
    r12 = kx * ky * vth - kz * sth
    r13 = kx * kz * vth + ky * sth
    r21 = kx * ky * vth + kz * sth
    r22 = ky * ky * vth + cth
    r23 = ky * kz * vth - kx * sth
    r31 = kx * kz * vth - ky * sth
    r32 = ky * kz * vth + kx * sth
    r33 = kz * kz * vth + cth

    beta = np.arctan2(-r31, np.sqrt(r11 * r11 + r21 * r21))

    if beta > d2r(89.99):
        beta = d2r(89.99)
        alpha = 0
        gamma = np.arctan2(r12, r22)
    elif beta < -d2r(89.99):
        beta = -d2r(89.99)
        alpha = 0
        gamma = -np.arctan2(r12, r22)
    else:
        cb = np.cos(beta)
        alpha = np.arctan2(r21 / cb, r11 / cb)
        gamma = np.arctan2(r32 / cb, r33 / cb)

    return [r2d(gamma), r2d(beta), r2d(alpha)]


def orientation_check():
    # v = torch.tensor([2.0, -1.0])    # a
    # x = torch.tensor([1.0, 0.0])    # b
    # e = v - (torch.dot(v, x) / torch.dot(x, x)) * x
    # print("e: ", e[-1])

    # rpy = torch.tensor([1.15, 1.251, 1.167])
    rpy = torch.tensor([1.824, 0.081, 1.626])

    axis_angle = torch.tensor([0.001, -2.222, 2.222])
    axis_angle = torch.tensor([1.507, 1.534, 0.966])
    # roll, pitch, yaw = rv2rpy(rx=0.041, ry=2.224, rz=2.216)
    # print("rv2rpy: ", roll, pitch, yaw)

    _q = tr.axis_angle_to_quaternion(axis_angle)     # [w, x, y, z]
    q = torch.cat((_q[1:], _q[0].unsqueeze(0)))

    # roll, pitch, yaw = rpy
    # q = quat_from_euler_xyz(roll=roll, pitch=pitch, yaw=yaw)
    x_axis = torch.tensor([1.0, 0.0, 0.0])  # reference axis
    q_axis = tf_vector(q, torch.tensor([0.0, 0.0, 1.0]))
    dot_qx = torch.bmm(q_axis.view(-1, 1, 3), x_axis.view(-1, 3, 1)).squeeze(-1).squeeze(-1)
    dot_xx = torch.bmm(x_axis.view(-1, 1, 3), x_axis.view(-1, 3, 1)).squeeze(-1).squeeze(-1)

    print("q: ", q)
    print("x_axis: ", x_axis, x_axis.norm())
    print("q_axis: ", q_axis, q_axis.norm())
    print(dot_qx, rad2deg(torch.acos(dot_qx)))

    e_x = q_axis - (dot_qx / dot_xx) * x_axis
    e_x = e_x / e_x.norm()
    print("e_x ", e_x, e_x.norm())


def quat_orientation_check():
    ref_axis_angle = torch.tensor([1.201, 1.219, 1.218])  # -90, 90, 180
    target_axis_angle = torch.tensor([2.226, 0.016, 2.227])  # -90, 90, -90

    # ref_axis_angle = torch.tensor([0.0, 0.0, 0.0])  # 0, 0, 0
    # target_axis_angle = tr.quaternion_to_axis_angle(torch.tensor([0.5, 0.5, -0.5, 0.5]))  # 0, 0, 90


    rq = quat_to_real_last(tr.axis_angle_to_quaternion(ref_axis_angle))
    tq = quat_to_real_last(tr.axis_angle_to_quaternion(target_axis_angle))
    eq = orientation_error(desired=tq.unsqueeze(0), current=rq.unsqueeze(0)).squeeze(0)
    print("rq [x, y, z, w]: ", rq)
    print("tq [x, y, z, w]: ", tq)
    print("eq [x, y, z]: ", eq)


if __name__ == "__main__":
    quat_orientation_check()
    # real_ur3_socket_test()
    # real_ur3_rtde_test()
    # gripper_test()
    # orientation_check()

