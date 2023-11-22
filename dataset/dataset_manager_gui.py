import os
import sys
import shutil
import time

import h5py
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore
from PIL import Image, ImageQt

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = parent_dir[:parent_dir.find(parent_dir.split('/')[-1])-1]
sys.path.append(parent_dir)

from spirl.utility.general_utils import AttrDict

#UI파일 연결 단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("dataset_manager_gui.ui")[0]
R, G, B = 0, 1, 2


class SkillDatasetManager(QMainWindow, form_class):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.path = None

        self.setWindowTitle("Skill Rollout Dataset Manager")

        # UI interface connection
        self.btn_function.clicked.connect(self.btn_function_clicked)
        self.btn_open_folder.clicked.connect(self.set_root_path)

        # tree view initialize
        self.fs_model = QFileSystemModel()
        self.treeview_files.setModel(self.fs_model)
        selmodel = self.treeview_files.selectionModel()
        selmodel.selectionChanged.connect(self.handle_selection_changed)

        # self.treeview_files.clicked.connect(self.on_treeview_clicked)
        self.treeview_files.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeview_files.customContextMenuRequested.connect(self.on_treeview_custom_context_menu)

        # auto play check box
        self.cb_autoplay.stateChanged.connect(self.auto_play_checkbox_state_changed)

        # timer setup for auto play
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.func_timeout)

        # spin box for auto play time tick
        self.sb_autoplay_tick.setRange(1, 30)

        # horizontal slider
        self.hslider_img_step.valueChanged.connect(self.progress_hslider_value_changed)
        self.hslider_img_step.setRange(0, 0)

        # dataset
        self.data = AttrDict()

        self.initialize_default_status(task="pouring_water_img")

    def initialize_default_status(self, task):
        # default mainframe size
        self.resize(1300, 1000)

        # default root path
        folder_name = QtCore.QDir.currentPath()
        folder_name = os.path.dirname(folder_name)
        folder_name = os.path.join(folder_name, "data", task)
        self.lineEdit_path.setText(folder_name)
        self.path = folder_name
        self.update_treeview()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Delete:
            index = self.treeview_files.currentIndex()
            index_item = self.fs_model.index(index.row(), 0, index.parent())
            file_name = self.fs_model.fileName(index_item)
            file_path = self.fs_model.filePath(index_item)
            if file_name:
                self.delete_file(file_name=file_name, file_path=file_path)

        elif e.key() == QtCore.Qt.Key_Escape:
            print("ESC")

    def delete_file(self, file_name, file_path, confirm_dialog=True):
        reply = QMessageBox.Yes
        if confirm_dialog:
            reply = QMessageBox.question(self, "Message", "Are you sure to delete the {}?".format(file_name),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)
            start, sep = file_path.find("batch"), file_path.rfind('/')
            folder_name = file_path[start:sep]
            self.textEdit_log.append("{}, {} is removed..".format(folder_name, file_name))

    def on_treeview_custom_context_menu(self, position):
        index = self.treeview_files.currentIndex()
        index_item = self.fs_model.index(index.row(), 0, index.parent())
        file_name = self.fs_model.fileName(index_item)
        file_path = self.fs_model.filePath(index_item)

        menu = QMenu()
        delete_action = menu.addAction("Delete")
        action = menu.exec_(self.treeview_files.mapToGlobal(position))
        if action == delete_action:
            self.delete_file(file_name=file_name, file_path=file_path)

    def on_treeview_clicked(self, index):
        index_item = self.fs_model.index(index.row(), 0, index.parent())
        file_name = self.fs_model.fileName(index_item)
        file_path = self.fs_model.filePath(index_item)

        if file_name.endswith('.h5'):   # do some actions only on .h5 files
            print("file_name: ", file_name)
            print("file_path: ", file_path)

    def auto_play_checkbox_state_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.timer.start()
            self.textEdit_log.append("Auto play is activated")
        else:
            self.timer.stop()
            self.textEdit_log.append("Auto play is deactivated")

    def func_timeout(self):
        if hasattr(self.data, "images"):
            _tick = self.sb_autoplay_tick.value()
            self.data.step = min(self.data.max_step, self.data.step + _tick)
            self.refresh()

    def progress_hslider_value_changed(self, value):
        self.data.step = value
        self.refresh()

    def refresh(self):
        self.hslider_img_step.setValue(self.data.step)
        self.update_image()
        self.update_pad_mask()
        self.step_progress_label_update()

    def step_progress_label_update(self):
        self.lb_step_prog.setText(" {} / {} ".format(self.data.step, self.data.max_step))

    def update_pad_mask(self):
        if hasattr(self.data, "pad_mask"):
            _size = self.lb_pad_mask.size()
            _pad_img = np.zeros((2, len(self.data.pad_mask), 3), dtype=np.uint8)
            _pad_img[0, :, G] = self.data.pad_mask * 255
            _pad_img[0, :, R] = (1 - self.data.pad_mask) * 255
            _pad_img[1, self.data.step, R:G+1] = 255

            pad_img = Image.fromarray(_pad_img, mode="RGB")
            qt_pad_img = ImageQt.ImageQt(pad_img)
            pixmap = QtGui.QPixmap.fromImage(qt_pad_img)
            pixmap = pixmap.scaled(_size.width(), _size.height())
            self.lb_pad_mask.setPixmap(pixmap)
            self.lb_pad_mask.show()
        else:
            self.textEdit_log.append("Does NOT have pad_mask")

    def update_image(self):
        if hasattr(self.data, "images"):
            img = Image.fromarray(self.data.images[self.data.step], mode='RGB')
            b, g, r = img.split()
            img = Image.merge("RGB", (r, g, b))
            qt_img = ImageQt.ImageQt(img)
            _size = self.lb_img.size()
            _val = min(_size.width(), _size.height())
            pixmap = QtGui.QPixmap.fromImage(qt_img)
            pixmap = pixmap.scaled(_val, _val, QtCore.Qt.KeepAspectRatio)
            self.lb_img.setPixmap(pixmap)
            self.lb_img.show()
        else:
            self.textEdit_log.append("Does NOT have images")

    def handle_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if indexes:
            index = indexes[0]
            index_item = self.fs_model.index(index.row(), 0, index.parent())
            file_name = self.fs_model.fileName(index_item)
            file_path = self.fs_model.filePath(index_item)

            if file_name.endswith('.h5'):
                start, sep = file_path.find("batch"), file_path.rfind('/')
                folder_name = file_path[start:sep]
                self.lineEdit_batch_name.setText(folder_name)
                self.lineEdit_rollout_name.setText(file_name)

                with h5py.File(file_path, 'r') as f:
                    self.textEdit_data_info.clear()
                    self.data = AttrDict()
                    key = 'traj{}'.format(0)
                    for name in f[key].keys():
                        if name in ['actions', 'states', 'rewards', 'terminals', 'pad_mask', 'extra']:
                            self.data[name] = f[key + '/' + name][()].astype(np.float32)
                        elif name in ['images']:
                            self.data[name] = f[key + '/' + name][()].astype(np.uint8)
                        else:
                            self.data[name] = f[key + '/' + name][()]
                            self.textEdit_data_info.append(info)
                            continue
                        info = "{}: \n    shape: {}, \n    type: {}, \n    min / max: {:.2f} / {:.2f}".\
                            format(name, self.data[name].shape, self.data[name].dtype,
                                   self.data[name].min(), self.data[name].max())
                        self.textEdit_data_info.append(info)

                    # step value
                    self.data.step = 0
                    self.data.max_step = len(self.data.pad_mask) - 1

                    self.update_image()
                    self.update_pad_mask()

                    # slider setting
                    self.hslider_img_step.setValue(self.data.step)
                    self.hslider_img_step.setRange(0, self.data.max_step)

                    self.step_progress_label_update()

    def update_treeview(self):
        if self.path:
            self.fs_model.setRootPath(self.path)
            index_root = self.fs_model.index(self.path)
            self.treeview_files.setRootIndex(index_root)

    def set_root_path(self):
        folder_name = QFileDialog.getExistingDirectory(self)
        self.lineEdit_path.setText(folder_name)
        self.path = folder_name
        self.update_treeview()

    def dataset_statistics(self):
        n_rollouts, n_frames = 0, 0
        for _bat in os.listdir(self.path):  # rollout file level
            _path = os.path.join(self.path, _bat)
            file_names = [fn for fn in os.listdir(_path) if any(fn.endswith(ext) for ext in ['h5'])]
            n_rollouts += len(file_names)
            for _file in file_names:        # frame level
                with h5py.File(os.path.join(_path, _file), 'r') as f:
                    key = 'traj{}'.format(0)
                    for name in f[key].keys():
                        if name in ['actions', 'states', 'rewards', 'terminals', 'pad_mask']:
                            n_frames += len(f[key + '/' + name][()])
                            break
        self.textEdit_log.append("Total # of rollouts: {:,}".format(n_rollouts))
        self.textEdit_log.append("Total # of frames: {:,}".format(n_frames))

    def btn_function_clicked(self):
        print("function button clicked...")
        print(self.frameSize())
        self.dataset_statistics()


if __name__ == "__main__":
    app = QApplication(sys.argv)    # QApplication : for program execution
    mgr = SkillDatasetManager()     # instance
    mgr.show()                      # display the program instance
    app.exec_()                     # make the program enter into an event loop