import rtde_control
from .robotiq_preamble import ROBOTIQ_PREAMBLE
import time
import socket


class RobotiqGripper(object):
    """
    RobotiqGripper is a class for controlling a robotiq gripper using the
    ur_rtde robot interface.

    Attributes:
        rtde_c (rtde_control.RTDEControlInterface): The interface to use for the communication
    """

    def __init__(self, rtde_c):
        """
        The constructor for RobotiqGripper class.

        Parameters:
           rtde_c (rtde_control.RTDEControlInterface): The interface to use for the communication
        """
        self.rtde_c = rtde_c

    def call(self, script_name, script_function):
        return self.rtde_c.sendCustomScriptFunction(
            "ROBOTIQ_" + script_name,
            ROBOTIQ_PREAMBLE + script_function
        )

    def activate(self):
        """
        Activates the gripper. Currently the activation will take 5 seconds.

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        ret = self.call("ACTIVATE", "rq_activate()")
        time.sleep(5)  # HACK
        return ret

    def set_speed(self, speed):
        """
        Set the speed of the gripper.

        Parameters:
            speed (int): speed as a percentage [0-100]

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        return self.call("SET_SPEED", "rq_set_speed_norm(" + str(speed) + ")")

    def set_force(self, force):
        """
        Set the force of the gripper.

        Parameters:
            force (int): force as a percentage [0-100]

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        return self.call("SET_FORCE", "rq_set_force_norm(" + str(force) + ")")

    def move(self, pos_in_mm):
        """
        Move the gripper to a specified position in (mm).

        Parameters:
            pos_in_mm (int): position in millimeters.

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        return self.call("MOVE", "rq_move_and_wait_mm(" + str(pos_in_mm) + ")")

    def open(self):
        """
        Open the gripper.

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        return self.call("OPEN", "rq_open_and_wait()")

    def close(self):
        """
        Close the gripper.

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        return self.call("CLOSE", "rq_close_and_wait()")


class RobotiqGripperExpand(RobotiqGripper):
    def __init__(self, rtde_c, HOST):
        super().__init__(rtde_c)
        self.HOST = HOST
        self.PORT = 63352    # fixed port for robotiq gripper

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.HOST, self.PORT))

        # gripper params
        self.set_var_list = ['ACT', 'GTO', 'ATR', 'ARD', 'FOR', 'SPE', 'POS', 'LBP', 'LRD', 'LBL', 'LGN', 'MSC', 'MOD']
        self.get_var_list = ['FLT', 'OBJ', 'STA', 'PRE', 'GTO']
        self.closed_mm = 0.
        self.open_mm = 85.
        self.closed_norm = 100.
        self.open_norm = 0.

        self.actual_min = 3
        self.actual_max = 227

        if not self.rq_is_gripper_activated():
            self.activate()

        self.target_grip_mm = self.gripper_to_mm_normalize()

    def rq_set_var(self, var_name, value):
        assert var_name in self.set_var_list
        var_name_str = "SET " + var_name + " " + str(value) + "\n"
        ack = False
        while not ack:
            ack = self.socket_send_recv(var_name_str, 2 ** 2)
        return ack

    def rq_get_var(self, var_name, n_bytes=2 ** 2):
        assert var_name in self.get_var_list
        var_name_str = "GET " + var_name + "\n"
        return self.socket_send_recv(var_name_str)

    def socket_send(self, text):
        assert self.sock is not None
        self.sock.sendall(bytes(text, 'utf-8'))

    def socket_recv(self, n_bytes=2 ** 10):
        return self.sock.recv(n_bytes)

    def socket_send_recv(self, text, n_bytes=2 ** 10):
        self.socket_send(text)
        return self.socket_recv(n_bytes=n_bytes)

    def rq_is_gripper_activated(self):
        var_name_str = "GET STA"    # status: 0=reset, 1=activating, 3=active
        gSTA = self.socket_send_recv(text=var_name_str)
        return True if self.is_STA_gripper_activated(gSTA) else False

    # def rq_activate(self):
    #     if not self.rq_is_gripper_activated():
    #         self.rq_reset()
    #     self.rq_set_var("ACT", 1)

    def rq_reset(self):
        self.rq_set_var("ACT", 0)
        self.rq_set_var("ATR", 0)

    def get_gripper_action(self, normalize=True):
        self.target_grip_mm = max(min(self.target_grip_mm, self.open_mm), self.closed_mm)
        return self.target_grip_mm / self.open_mm if normalize else self.target_grip_mm

    @staticmethod
    def is_STA_gripper_activated(list_of_bytes):
        if len(list_of_bytes) != 1:   # list length is not 1
            return False
        if list_of_bytes[0] == 51:  # byte is '3'?
            return True
        return False

    def grasping_by_hold(self, step=-5.0):
        value_mm = self.gripper_to_mm()
        value_mm += step
        self.target_grip_mm = value_mm
        return self.rq_move_mm(value_mm)

    def gripper_to_mm(self):
        var_name_str = "GET POS\n"
        data = self.socket_send_recv(var_name_str)

        gripper_value = int(data.decode('utf-8').split(' ')[-1])  # [0, 255]
        value_norm = ((gripper_value - self.actual_min) / (self.actual_max - self.actual_min)) * 100  # [0, 100]

        slope = (self.closed_mm - self.open_mm) / (self.closed_norm - self.open_norm)
        value_mm = slope * (value_norm - self.closed_norm) + self.closed_mm

        if value_mm > self.open_mm:
            value_mm_limited = self.open_mm
        elif value_mm < self.closed_mm:
            value_mm_limited = self.closed_mm
        else:
            value_mm_limited = value_mm
        return value_mm_limited

    def gripper_to_mm_normalize(self):
        value_mm = self.gripper_to_mm()
        return value_mm / self.open_mm

    def mm_to_gripper(self, value_mm):
        slope = (self.closed_norm - self.open_norm) / (self.closed_mm - self.open_mm)
        value_norm = (value_mm - self.closed_mm) * slope + self.closed_norm
        value_gripper = value_norm * self.actual_max / 100.

        if value_gripper > self.actual_max:
            value_gripper_limited = self.actual_max
        elif value_gripper < self.actual_min:
            value_gripper_limited = self.actual_min
        else:
            value_gripper_limited = round(value_gripper)
        return value_gripper_limited

    def rq_move_mm(self, pos_mm):
        pos_gripper = self.mm_to_gripper(pos_mm)
        var_name_str = "SET POS " + str(pos_gripper) + "\n"
        ack = self.socket_send_recv(var_name_str, n_bytes=2 ** 3)
        return ack

    def rq_move_mm_norm(self, pos_mm_norm):
        return self.rq_move_mm(pos_mm=pos_mm_norm * self.open_mm)
