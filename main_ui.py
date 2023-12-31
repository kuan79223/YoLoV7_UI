# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(989, 731)
        MainWindow.setStyleSheet("background:black")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lb_project_path = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lb_project_path.setFont(font)
        self.lb_project_path.setStyleSheet("color:white")
        self.lb_project_path.setText("")
        self.lb_project_path.setObjectName("lb_project_path")
        self.verticalLayout_2.addWidget(self.lb_project_path)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setStyleSheet("background:white")
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_project = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btn_project.setFont(font)
        self.btn_project.setStyleSheet("height:50px;\n"
"background:white;\n"
"color:black")
        self.btn_project.setObjectName("btn_project")
        self.horizontalLayout.addWidget(self.btn_project)
        self.lb_train_param = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lb_train_param.setFont(font)
        self.lb_train_param.setStyleSheet("color:white")
        self.lb_train_param.setText("")
        self.lb_train_param.setObjectName("lb_train_param")
        self.horizontalLayout.addWidget(self.lb_train_param)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 8)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.stack_layout = QtWidgets.QVBoxLayout()
        self.stack_layout.setObjectName("stack_layout")
        self.tab_widget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        self.tab_widget.setFont(font)
        self.tab_widget.setObjectName("tab_widget")
        self.Tab_Training = QtWidgets.QWidget()
        self.Tab_Training.setObjectName("Tab_Training")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.Tab_Training)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.cbb_imgSize_select = QtWidgets.QComboBox(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.cbb_imgSize_select.setFont(font)
        self.cbb_imgSize_select.setStyleSheet("background:white;\n"
"min-height:50px")
        self.cbb_imgSize_select.setObjectName("cbb_imgSize_select")
        self.gridLayout_2.addWidget(self.cbb_imgSize_select, 3, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("height:50px;\n"
"color:white;\n"
"border: 2px solid white")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 1, 3, 1, 1)
        self.cbb_epoch_select = QtWidgets.QComboBox(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.cbb_epoch_select.setFont(font)
        self.cbb_epoch_select.setStyleSheet("background:white;\n"
"min-height:50px")
        self.cbb_epoch_select.setObjectName("cbb_epoch_select")
        self.gridLayout_2.addWidget(self.cbb_epoch_select, 3, 5, 1, 1)
        self.cbb_device_select = QtWidgets.QComboBox(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.cbb_device_select.setFont(font)
        self.cbb_device_select.setStyleSheet("background:white;\n"
"min-height:50px")
        self.cbb_device_select.setObjectName("cbb_device_select")
        self.gridLayout_2.addWidget(self.cbb_device_select, 3, 7, 1, 1)
        self.cbb_batchSize_select = QtWidgets.QComboBox(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.cbb_batchSize_select.setFont(font)
        self.cbb_batchSize_select.setStyleSheet("background:white;\n"
"min-height:50px")
        self.cbb_batchSize_select.setObjectName("cbb_batchSize_select")
        self.gridLayout_2.addWidget(self.cbb_batchSize_select, 3, 3, 1, 1)
        self.cbb_classScale_select = QtWidgets.QComboBox(self.Tab_Training)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbb_classScale_select.sizePolicy().hasHeightForWidth())
        self.cbb_classScale_select.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.cbb_classScale_select.setFont(font)
        self.cbb_classScale_select.setStyleSheet("background:white;\n"
"height:50px")
        self.cbb_classScale_select.setObjectName("cbb_classScale_select")
        self.gridLayout_2.addWidget(self.cbb_classScale_select, 3, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("height:50px;\n"
"color:white;\n"
"border: 2px solid white")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("height:50px;\n"
"color:white;\n"
"border: 2px solid white")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 5, 1, 1)
        self.btn_model = QtWidgets.QPushButton(self.Tab_Training)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.btn_model.setFont(font)
        self.btn_model.setStyleSheet("height:50px;\n"
"background:#66ccff;\n"
"color:white;\n"
"border: 2px solid white")
        self.btn_model.setObjectName("btn_model")
        self.gridLayout_2.addWidget(self.btn_model, 1, 8, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("height:50px;\n"
"color:white;\n"
"border: 2px solid white")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 1, 7, 1, 1)
        self.btn_train_run = QtWidgets.QPushButton(self.Tab_Training)
        font = QtGui.QFont()
        font.setFamily("新細明體")
        font.setPointSize(20)
        self.btn_train_run.setFont(font)
        self.btn_train_run.setStyleSheet("height:50px;\n"
"background:green;\n"
"color:white;\n"
"")
        self.btn_train_run.setObjectName("btn_train_run")
        self.gridLayout_2.addWidget(self.btn_train_run, 3, 8, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.Tab_Training)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color:white;\n"
"min-height:50px;\n"
"max-height:50px;\n"
"border: 2px solid white")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.chartLayout = QtWidgets.QVBoxLayout()
        self.chartLayout.setObjectName("chartLayout")
        self.verticalLayout.addLayout(self.chartLayout)
        self.verticalLayout.setStretch(1, 5)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.verticalLayout_7.addLayout(self.horizontalLayout_4)
        self.tab_widget.addTab(self.Tab_Training, "")
        self.Tab_Predic = QtWidgets.QWidget()
        self.Tab_Predic.setObjectName("Tab_Predic")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.Tab_Predic)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.predictLayout = QtWidgets.QVBoxLayout()
        self.predictLayout.setObjectName("predictLayout")
        self.verticalLayout_6.addLayout(self.predictLayout)
        self.tab_widget.addTab(self.Tab_Predic, "")
        self.stack_layout.addWidget(self.tab_widget)
        self.verticalLayout_2.addLayout(self.stack_layout)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 8)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_project.setText(_translate("MainWindow", "Project"))
        self.label_5.setText(_translate("MainWindow", "批次數量"))
        self.label_4.setText(_translate("MainWindow", "影像解析度:"))
        self.label_6.setText(_translate("MainWindow", "訓練次數"))
        self.btn_model.setText(_translate("MainWindow", "Model"))
        self.label_7.setText(_translate("MainWindow", "選擇硬體設備"))
        self.btn_train_run.setText(_translate("MainWindow", "Train"))
        self.label_2.setText(_translate("MainWindow", "訓練類別比例:"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.Tab_Training), _translate("MainWindow", "Training"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.Tab_Predic), _translate("MainWindow", "Predict"))
