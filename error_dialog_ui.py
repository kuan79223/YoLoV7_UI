# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'error_dialog_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("dialog_tool")
        Dialog.resize(422, 210)
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(12)
        Dialog.setFont(font)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lb_show_error = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        self.lb_show_error.setFont(font)
        self.lb_show_error.setText("")
        self.lb_show_error.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_show_error.setObjectName("lb_show_error")
        self.verticalLayout.addWidget(self.lb_show_error)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.btn_check = QtWidgets.QPushButton(Dialog)
        self.btn_check.setObjectName("btn_check")
        self.horizontalLayout.addWidget(self.btn_check)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("dialog_tool", "dialog_tool"))
        self.btn_check.setText(_translate("dialog_tool", "確定"))
