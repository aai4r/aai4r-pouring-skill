import socket
import time
import json
import numpy as np

import rtde_control
import rtde_receive
from gripper.robotiq_gripper_control import RobotiqGripper


def rad2deg(rad): return rad * (180.0 / np.pi)


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


if __name__ == "__main__":
    # real_ur3_socket_test()
    # real_ur3_rtde_test()
    gripper_test()
