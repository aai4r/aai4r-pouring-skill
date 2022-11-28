import os
import sys
import shutil
import time

import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator

import json
import socket
from threading import Event, Thread


form_class = uic.loadUiType("robot_gui.ui")[0]


class RobotControlGUI(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("Robot Control GUI")

        ipRange = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])"  # Part of the regular expression
        ipRegex = QRegExp("^" + ipRange + "\\." + ipRange + "\\." + ipRange + "\\." + ipRange + "$")
        ipValidator = QRegExpValidator(ipRegex, self)
        self.lineEdit_target_ip.setValidator(ipValidator)

        self.btn_connect.clicked.connect(self.socket_connect)
        self.btn_disconnect.clicked.connect(self.socket_disconnect)
        self.threads = []
        self.events = []

    def socket_disconnect(self):
        if not hasattr(self, "client_socket"):
            return
        try:
            self.client_socket.shutdown(socket.SHUT_RDWR)
            self.client_socket.close()
        except:
            pass

    def socket_connect(self):
        HOST = self.lineEdit_target_ip.text()   # default: 127.0.0.1
        PORT = self.spinBox_port.value()     # default: 9999

        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((HOST, PORT))
            self.textEdit_log.append("Connected !")
        except ConnectionAbortedError as e:
            self.textEdit_log.append(str(e) + " Check the ip address or port number")
        except ConnectionRefusedError as e:
            self.textEdit_log.append(str(e) + " Check the server is opened")
            return False

        event = Event()
        t = Thread(target=self._threaded, args=(event, ))
        self.events.append(event)
        self.threads.append(t)
        t.start()
        return True

    def _threaded(self, event):
        while True:
            if event.is_set():
                print("Kill Thread...")
                self.client_socket.close()
                break

            try:
                message = "CONTROL SIGNAL"
                self.client_socket.send(message.encode())

                json_data = self.client_socket.recv(1024)
                self.parse_to_gui(json_data=json_data.decode())
            except:
                pass

    def parse_to_gui(self, json_data):
        parsed = json.loads(json_data)

        # joint
        show_deg = self.checkBox_show_deg.checkState()
        r2d = 180.0 / np.pi if show_deg else 1
        joints = list(map(lambda x: str(round(x * r2d, 2)), parsed.get("joint").values()))
        self.lineEdit_j1_disp.setText(joints[0])
        self.lineEdit_j2_disp.setText(joints[1])
        self.lineEdit_j3_disp.setText(joints[2])
        self.lineEdit_j4_disp.setText(joints[3])
        self.lineEdit_j5_disp.setText(joints[4])
        self.lineEdit_j6_disp.setText(joints[5])

        # gripper
        gripper = list(map(lambda x: str(round(x, 2)), parsed.get("gripper").values()))
        self.lineEdit_grip_disp.setText(gripper[0])

        # pose
        # : grip_pose, ee_pose, etc.
        mm = 1000.0
        grip_pos = list(map(lambda x: str(round(x * mm, 2)), parsed.get("grip_pos").values()))
        grip_rot = list(map(lambda x: str(round(x, 2)), parsed.get("grip_rot").values()))
        self.lineEdit_x_disp.setText(grip_pos[0])
        self.lineEdit_y_disp.setText(grip_pos[1])
        self.lineEdit_z_disp.setText(grip_pos[2])
        self.lineEdit_rx_disp.setText(grip_rot[0])
        self.lineEdit_ry_disp.setText(grip_rot[1])
        self.lineEdit_rz_disp.setText(grip_rot[2])

    def kill_all_threads(self):
        [e.set() for e in self.events]

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.kill_all_threads()


if __name__ == "__main__":
    app = QApplication(sys.argv)    # QApplication : for program execution
    mgr = RobotControlGUI()     # instance
    mgr.show()                      # display the program instance
    app.exec_()                     # make the program enter into an event loop
