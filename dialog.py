import os
import subprocess
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog

import dialog_ui


class Dialog(QDialog, dialog_ui.Ui_Dialog):

    def __init__(self):
        super(Dialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(600, 400)

        self.result = 0
        self.btn_check.clicked.connect(self.click_check)
        self.btn_cancel.clicked.connect(self.click_cancel)
        # 設定視窗為無標題框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.show_msg('訓練結束', '是否繼續使用應用程式訓練')

    def show_msg(self, title, msg):
        self.lb_show_title.setText(title)
        self.lb_show_msg.setText(msg)

    def click_check(self):
        self.result = 1
        self.close()
        python_executable = sys.executable
        subprocess.Popen(["python", os.path.abspath("main.py")], executable=python_executable).wait()

    def click_cancel(self):
        self.result = 0
        self.close()

    # 不可使用 esc 關閉視窗
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = Dialog()
    form.show()
    sys.exit(app.exec_())
