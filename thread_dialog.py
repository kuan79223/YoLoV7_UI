from PyQt5.QtCore import QThread, pyqtSignal
from error_dialog import ErrorDialog


class ThreadDialog(QThread):
    SIGNAL_SHOW_DIALOG = pyqtSignal()
    SIGNAL_CLOSE_DIALOG = pyqtSignal()

    def __init__(self, message):
        super(ThreadDialog, self).__init__()
        self.msg = message
        self.error_dialog = ErrorDialog()
        self.SIGNAL_SHOW_DIALOG.connect(self.show_dialog)
        self.SIGNAL_CLOSE_DIALOG.connect(self.close_dialog)

    def run(self):
        print('open dialog')
        self.SIGNAL_SHOW_DIALOG.emit()

    def show_dialog(self):
        self.error_dialog.show_error(self.msg)
        self.error_dialog.show()

    def close_dialog(self):
        self.SIGNAL_CLOSE_DIALOG.emit()
        print('close dialog')
        self.error_dialog.close()
