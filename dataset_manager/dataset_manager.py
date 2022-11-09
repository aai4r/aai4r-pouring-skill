import os
import sys
import shutil
import h5py
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore
from PIL import Image, ImageQt

from spirl.utils.general_utils import AttrDict

#UI파일 연결 단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("dataset_manager.ui")[0]
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

        # horizontal slider
        self.hslider_img_step.valueChanged.connect(self.value_changed)
        self.hslider_img_step.setRange(0, 0)

        # dataset
        self.data = AttrDict()

        self.initialize_default_status(task="pouring_water_img")

    def initialize_default_status(self, task):
        # default mainframe size
        self.resize(1300, 970)

        # default root path
        folder_name = QtCore.QDir.currentPath()
        folder_name = os.path.dirname(folder_name)
        folder_name = os.path.join(folder_name, "data", task)
        self.lineEdit_path.setText(folder_name)
        self.path = folder_name
        self.update_treeview()

    def on_treeview_custom_context_menu(self, position):
        index = self.treeview_files.currentIndex()
        index_item = self.fs_model.index(index.row(), 0, index.parent())
        file_name = self.fs_model.fileName(index_item)
        file_path = self.fs_model.filePath(index_item)

        menu = QMenu()
        delete_action = menu.addAction("Delete")
        action = menu.exec_(self.treeview_files.mapToGlobal(position))
        if action == delete_action:
            reply = QMessageBox.question(self, "Message", "Are you sure to delete the {}?".format(file_name),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)

    def on_treeview_clicked(self, index):
        index_item = self.fs_model.index(index.row(), 0, index.parent())
        file_name = self.fs_model.fileName(index_item)
        file_path = self.fs_model.filePath(index_item)

        if file_name.endswith('.h5'):   # do some actions only on .h5 files
            print("file_name: ", file_name)
            print("file_path: ", file_path)

    def value_changed(self, value):
        self.data.step = value
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
            self.textEdit_log.setText("Does NOT have pad_mask")

    def update_image(self):
        if hasattr(self.data, "images"):
            img = Image.fromarray(self.data.images[self.data.step], mode='RGB')
            qt_img = ImageQt.ImageQt(img)
            _size = self.lb_img.size()
            _val = min(_size.width(), _size.height())
            pixmap = QtGui.QPixmap.fromImage(qt_img)
            pixmap = pixmap.scaled(_val, _val, QtCore.Qt.KeepAspectRatio)
            self.lb_img.setPixmap(pixmap)
            self.lb_img.show()
        else:
            self.textEdit_log.setText("Does NOT have images")

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
                        if name in ['actions', 'states', 'rewards', 'terminals', 'pad_mask']:
                            self.data[name] = f[key + '/' + name][()].astype(np.float32)
                        elif name in ['images']:
                            self.data[name] = f[key + '/' + name][()].astype(np.uint8)
                        self.textEdit_data_info.append("{}: shape: {}".format(name, self.data[name].shape))

                    # step value
                    self.data.step = 0
                    self.data.max_step = len(self.data.pad_mask) - 1

                    self.update_image()
                    self.update_pad_mask()

                    # slider setting
                    self.hslider_img_step.setRange(0, self.data.max_step)
                    self.hslider_img_step.setValue(self.data.step)

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

    def btn_function_clicked(self):
        print("function button clicked...")
        print(self.frameSize())


if __name__ == "__main__":
    app = QApplication(sys.argv)    # QApplication : for program execution
    mgr = SkillDatasetManager()     # instance
    mgr.show()                      # display the program instance
    app.exec_()                     # make the program enter into an event loop
