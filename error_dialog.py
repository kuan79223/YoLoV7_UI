import sys

from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QDialog
import error_dialog_ui


class ErrorDialog(QDialog, error_dialog_ui.Ui_Dialog):

    def __init__(self):
        super(ErrorDialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(600, 200)
        self.btn_check.clicked.connect(self.check_close)

    def show_error(self, msg):
        self.lb_show_error.setText(msg)

    def check_close(self):
        self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = ErrorDialog()
    form.show()
    sys.exit(app.exec_())
