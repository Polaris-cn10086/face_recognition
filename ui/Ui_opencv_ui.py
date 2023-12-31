# Form implementation generated from reading ui file 'd:\Pyhton\Project\test\face_recognition\ui\opencv_ui.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        mainwindow.setObjectName("mainwindow")
        mainwindow.resize(1186, 650)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(mainwindow)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_school_image = QtWidgets.QLabel(parent=mainwindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_school_image.sizePolicy().hasHeightForWidth())
        self.label_school_image.setSizePolicy(sizePolicy)
        self.label_school_image.setMaximumSize(QtCore.QSize(144, 144))
        self.label_school_image.setText("")
        self.label_school_image.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.label_school_image.setScaledContents(True)
        self.label_school_image.setObjectName("label_school_image")
        self.verticalLayout.addWidget(self.label_school_image)
        self.label_name = QtWidgets.QLabel(parent=mainwindow)
        font = QtGui.QFont()
        font.setFamily("仿宋")
        font.setPointSize(12)
        self.label_name.setFont(font)
        self.label_name.setObjectName("label_name")
        self.verticalLayout.addWidget(self.label_name)
        self.label_student_id = QtWidgets.QLabel(parent=mainwindow)
        font = QtGui.QFont()
        font.setFamily("仿宋")
        font.setPointSize(12)
        self.label_student_id.setFont(font)
        self.label_student_id.setObjectName("label_student_id")
        self.verticalLayout.addWidget(self.label_student_id)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.label_head_portrait = QtWidgets.QLabel(parent=mainwindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_head_portrait.sizePolicy().hasHeightForWidth())
        self.label_head_portrait.setSizePolicy(sizePolicy)
        self.label_head_portrait.setMaximumSize(QtCore.QSize(144, 144))
        self.label_head_portrait.setText("")
        self.label_head_portrait.setScaledContents(True)
        self.label_head_portrait.setObjectName("label_head_portrait")
        self.verticalLayout.addWidget(self.label_head_portrait)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_selected_title = QtWidgets.QLabel(parent=mainwindow)
        self.label_selected_title.setStyleSheet("font: 16pt \"楷体\";")
        self.label_selected_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_selected_title.setObjectName("label_selected_title")
        self.verticalLayout_3.addWidget(self.label_selected_title)
        self.selected_label = QtWidgets.QLabel(parent=mainwindow)
        self.selected_label.setText("")
        self.selected_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.selected_label.setObjectName("selected_label")
        self.verticalLayout_3.addWidget(self.selected_label)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 10)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_result_title = QtWidgets.QLabel(parent=mainwindow)
        self.label_result_title.setStyleSheet("font: 16pt \"楷体\";")
        self.label_result_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_result_title.setObjectName("label_result_title")
        self.verticalLayout_2.addWidget(self.label_result_title)
        self.result_label = QtWidgets.QLabel(parent=mainwindow)
        self.result_label.setText("")
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_label.setObjectName("result_label")
        self.verticalLayout_2.addWidget(self.result_label)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 10)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.load_image_btn = QtWidgets.QPushButton(parent=mainwindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.load_image_btn.sizePolicy().hasHeightForWidth())
        self.load_image_btn.setSizePolicy(sizePolicy)
        self.load_image_btn.setStyleSheet("font: 16pt \"楷体\";")
        self.load_image_btn.setObjectName("load_image_btn")
        self.horizontalLayout_2.addWidget(self.load_image_btn)
        self.load_video_btn = QtWidgets.QPushButton(parent=mainwindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.load_video_btn.sizePolicy().hasHeightForWidth())
        self.load_video_btn.setSizePolicy(sizePolicy)
        self.load_video_btn.setStyleSheet("font: 16pt \"楷体\";")
        self.load_video_btn.setObjectName("load_video_btn")
        self.horizontalLayout_2.addWidget(self.load_video_btn)
        self.load_camera_btn = QtWidgets.QPushButton(parent=mainwindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.load_camera_btn.sizePolicy().hasHeightForWidth())
        self.load_camera_btn.setSizePolicy(sizePolicy)
        self.load_camera_btn.setStyleSheet("font: 16pt \"楷体\";")
        self.load_camera_btn.setObjectName("load_camera_btn")
        self.horizontalLayout_2.addWidget(self.load_camera_btn)
        self.save_person_btn = QtWidgets.QPushButton(parent=mainwindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_person_btn.sizePolicy().hasHeightForWidth())
        self.save_person_btn.setSizePolicy(sizePolicy)
        self.save_person_btn.setStyleSheet("font: 16pt \"楷体\";")
        self.save_person_btn.setObjectName("save_person_btn")
        self.horizontalLayout_2.addWidget(self.save_person_btn)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.verticalLayout_4.setStretch(0, 10)
        self.verticalLayout_4.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 10)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.retranslateUi(mainwindow)
        QtCore.QMetaObject.connectSlotsByName(mainwindow)

    def retranslateUi(self, mainwindow):
        _translate = QtCore.QCoreApplication.translate
        mainwindow.setWindowTitle(_translate("mainwindow", "人脸识别"))
        self.label_name.setText(_translate("mainwindow", "姓名："))
        self.label_student_id.setText(_translate("mainwindow", "学号："))
        self.label_selected_title.setText(_translate("mainwindow", "源"))
        self.label_result_title.setText(_translate("mainwindow", "识别后"))
        self.load_image_btn.setText(_translate("mainwindow", "识别图片"))
        self.load_video_btn.setText(_translate("mainwindow", "识别视频"))
        self.load_camera_btn.setText(_translate("mainwindow", "识别摄像"))
        self.save_person_btn.setText(_translate("mainwindow", "存储识别对象"))
