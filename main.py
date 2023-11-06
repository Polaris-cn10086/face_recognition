import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
import os
import sys

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ui.Ui_opencv_ui import Ui_mainwindow
from ui.Ui_save_person import Ui_savewindow
# from qt_material import apply_stylesheet


def train():
    path = "./data/jm"
    faces = []
    ids = []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    face_detector = cv2.CascadeClassifier(
        "./train/haarcascade_frontalface_alt2.xml")

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        face = face_detector.detectMultiScale(img_numpy)
        id = int(os.path.split(imagePath)[1].split('.')[0].split("-")[0])

    # print(face)

    for x, y, w, h in face:
        ids.append(id)
        faces.append(img_numpy[y:y+h, x:x+w])
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write('./train/train.yml')


# 将Opencv中的mat转换为QT中的image
def CV2QImage(image):

    height, width, depth = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = QImage(image.data, width, height, width * depth, QImage.Format.Format_RGB888)
    return image

# 适配中文
def cv_image_add_text(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "C:\WINDOWS\FONTS\MSYHL.TTC", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 人脸检测，返回处理后的识别结果
def face_recognition(image):

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("./train/train.yml")
    # 准备识别图片
    # image = cv2.imread(file_path)
    gary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier(
        './train/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gary_image)
    for x, y, width, height in face:
        cv2.rectangle(image, (x, y), (x + width, y + height),
                      color=(0, 0, 255), thickness=2)

        # 人脸识别
        id, confidence = recognizer.predict(gary_image[y:y+height, x:x+width])

        # id 筛选
        name = name_dict.get(str(id), "未知")

        # 置信度筛选
        if confidence < 80.0:

            # 添加标签
            image = cv_image_add_text(image, "{} {:.3f}".format(
                name, confidence), x-5, y-50, (0, 0, 255), 40)
        else:
            image = cv_image_add_text(
                image, "未知", x-5, y-50, (0, 0, 255), 40)

        # print(id, "\t", confidence)
    return image


class SavePersonWindow(QWidget, Ui_savewindow):

    mySignal = pyqtSignal(str)

    def __init__(self):
        super(SavePersonWindow, self).__init__()
        self.setupUi(self)
        self.save_name
        self.bind_slots()

    def save_name(self):
        save_name = self.name_lineedit.text()
        print(save_name)

        self.mySignal.emit(save_name)

        self.close()

    def bind_slots(self):
        self.no_btn.clicked.connect(self.close)
        self.yes_btn.clicked.connect(self.save_name)


class MainWindow(QWidget, Ui_mainwindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.info()
        self.timer = QTimer()
        self.timer.setInterval(21)
        self.video = None
        self.save_name = None
        self.save_person_image = None
        self.bind_slots()

    # 个人信息处理
    def info(self):
        self.label_head_portrait.setPixmap(QPixmap("./ui/toux.jpg"))
        self.label_school_image.setPixmap(QPixmap("./ui/yjtp.png"))
        
    # 图片处理
    def image_pred(self, file_path):
        image = cv2.imread(file_path)
        return CV2QImage(face_recognition(image))

    # 视频处理
    def video_pred(self):
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
        else:
            image = frame.copy()
            image = face_recognition(image)
            self.selected_label.setPixmap(
                QPixmap.fromImage(CV2QImage(frame)).scaledToWidth(500))
            self.result_label.setPixmap(
                QPixmap.fromImage(CV2QImage(image)).scaledToWidth(500))

    # 打开图片
    def open_image(self):

        self.timer.stop()
        filepath = QFileDialog.getOpenFileName(
            self, "选择图片", "./images", "*.jpg;*.png")
        if filepath[0]:
            filepath = filepath[0]
            qimage = self.image_pred(filepath)
            self.selected_label.setPixmap(
                QPixmap(filepath).scaledToWidth(500))
            self.result_label.setPixmap(
                QPixmap.fromImage(qimage).scaledToWidth(500))

    # 打开相机
    def open_camera(self):
        self.video = cv2.VideoCapture(0)
        self.timer.start()

    # 打开视频
    def open_video(self):
        filepath = QFileDialog.getOpenFileName(
            self, "选择视频", "./video", "*.mp4")
        self.video = cv2.VideoCapture(filepath[0])

        self.timer.start()

    # 打开存储窗口
    def open_save_window(self):

        self.SavePerson = SavePersonWindow()
        self.SavePerson.mySignal.connect(self.get_save_name)
        if self.video is None:
            QMessageBox.warning(self, "错误", "未打开摄像头")
            return

        ret, frame = self.video.read()
        if not ret:
            return
        self.save_person_image = frame.copy()
        self.SavePerson.label_image.setPixmap(
            QPixmap.fromImage(CV2QImage(cv2.flip(frame, 1)).scaledToWidth(300)))

        # 保存文件
        # cv2.imwrite("./data/jm/aaa.jpg", frame)

        # cv2.imshow("save", frame)

        self.SavePerson.show()

    # 传参函数
    def get_save_name(self, connect):

        self.save_name = connect
        print(self.save_name, "已获取")

        # 保存图像
        # is_writed = cv2.imwrite("./data/jm/" + str(int(list(name_dict.keys())[-1]) + 1) + "-" + self.save_name +
        #             ".jpg", self.save_person_image)
        

        # 解决imwrite未报错但不写入图片的问题
        cv2.imencode(".jpg",self.save_person_image)[1].tofile("./data/jm/" + str(int(list(name_dict.keys())[-1]) + 1) + "-" + self.save_name +".jpg")

        # if is_writed:
        # 训练数据集
        train()
        # 更新字典
        name_dict[str(int(list(name_dict.keys())[-1]) + 1)] = self.save_name
        #else:
        #print("未能存入图片")

    def bind_slots(self):
        self.load_image_btn.clicked.connect(self.open_image)
        self.load_camera_btn.clicked.connect(self.open_camera)
        self.timer.timeout.connect(self.video_pred)
        self.load_video_btn.clicked.connect(self.open_video)
        self.save_person_btn.clicked.connect(self.open_save_window)


if __name__ == "__main__":
    # 导入配置文件
    name_dict = {}
    with open("./data/setting.csv", mode="r",encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            name_dict[row[0]] = row[1]
    app = QApplication(sys.argv)
    window = MainWindow()

    # apply_stylesheet(app, theme='light_blue.xml')
    window.show()

    app.exec()

    # 更新配置文件
    with open("./data/setting.csv", mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list(name_dict.items()))
    cv2.destroyAllWindows()
