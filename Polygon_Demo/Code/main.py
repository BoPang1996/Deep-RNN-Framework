# -*- coding: UTF-8 -*-

import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication)
import math
from run import deep_rnn_annotate

# coordinates of the selected object
record = [0, 0, 0, 0]
# selected mode
select = ''
# instance number
cur_ins_id = 0


# screen shot
class WScreenShot(QWidget):
    win = ''
    # 自定义图片保存路径
    save_path = "../images/save.jpg"

    @classmethod
    def run(cls, x, y):  # screenshot
        cls.win = cls(x, y)
        cls.win.show()

    def __init__(self, x, y, parent=None):
        super(WScreenShot, self).__init__(parent)
        self.bias = QPoint(x, y)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet('''background-color:black; ''')
        self.setWindowOpacity(0.6)
        desktopRect = QDesktopWidget().screenGeometry()
        self.setGeometry(x, y, 1024, 512)
        self.setCursor(Qt.CrossCursor)
        self.blackMask = QBitmap(desktopRect.size())
        self.blackMask.fill(Qt.black)
        self.mask = self.blackMask.copy()
        self.isDrawing = False
        self.startPoint = QPoint()
        self.endPoint = QPoint()

    # 自定义绘画事件
    def paintEvent(self, event):
        if self.isDrawing:
            self.mask = self.blackMask.copy()
            pp = QPainter(self.mask)
            pen = QPen()
            pen.setStyle(Qt.NoPen)
            pp.setPen(pen)
            brush = QBrush(Qt.white)
            pp.setBrush(brush)
            pp.drawRect(QRect(self.startPoint, self.endPoint))
            self.setMask(QBitmap(self.mask))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.startPoint = event.pos()
            self.endPoint = self.startPoint
            self.isDrawing = True

    def mouseMoveEvent(self, event):
        if self.isDrawing:
            self.endPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # record the coordinates of the selected object
            record[0] = (self.startPoint.x())
            record[1] = (self.startPoint.y())
            record[2] = (self.endPoint.x())
            record[3] = (self.endPoint.y())
            screenshot = QApplication.primaryScreen().grabWindow(QApplication.desktop().winId())
            rect = QRect(self.startPoint + self.bias, self.endPoint + self.bias)
            outputRegion = screenshot.copy(rect)
            # save the selected object
            outputRegion.save(self.save_path, format='JPG', quality=100)
            self.close()


class MyLabel(QLabel):
    def __init__(self, parent=None):
        QLabel.__init__(self, parent)
        self.points_list = []

        self.threshold = 5
        self.flag = False
        self.choose_index = -1
        self.choose_polygon = -1

    def addPolygon(self, points):
        self.points_list.append(points)

    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()

        min_distance = 100000000
        min_poly = -1
        min_index = -1
        for i in range(len(self.points_list)):
            for j in range(len(self.points_list[i])):
                distance = math.sqrt((x - self.points_list[i][j].x()) ** 2 + (y - self.points_list[i][j].y()) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    min_poly = i
                    min_index = j

        if event.buttons() == Qt.LeftButton:
            if min_poly != -1 and min_index != -1 and min_distance < self.threshold:
                self.flag = True
                self.choose_polygon = min_poly
                self.choose_index = min_index

        elif event.buttons() == Qt.RightButton:
            print("....")
            if min_index != -1 and min_distance < self.threshold:
                del self.points_list[min_poly][min_index]
                self.update()

    def mouseReleaseEvent(self, event):
        self.flag = False
        self.choose_polygon = -1
        self.choose_index = -1

    def mouseMoveEvent(self, event):
        if self.flag:
            x = event.x()
            y = event.y()

            self.points_list[self.choose_polygon][self.choose_index].setX(x)
            self.points_list[self.choose_polygon][self.choose_index].setY(y)

            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        for i in range(len(self.points_list)):

            painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
            painter.drawPolygon(QPolygon(self.points_list[i]))
            painter.setPen(QPen(Qt.black, 8, Qt.SolidLine))
            painter.drawPoints(QPolygon(self.points_list[i]))


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.resize(1250, 600)
        menubar = self.menuBar()
        menubar.addMenu('&Menu')
        menubar.addMenu('&Mode')
        menubar.addMenu('&Instruction')
        self.setWindowTitle("Deep RNN Annotator")
        self.setWindowIcon(QIcon('icon.png'))

        self.setFixedSize(self.width(), self.height())
        self.label = MyLabel(self)
        self.label.setText("             Waiting to load image...")
        self.label.setFixedSize(1024, 512)
        self.label.move(10, 50)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(0,0,0);font-size:40px;font-weight:bold;font-family:宋体;}"
                                 )

        font = QtGui.QFont()
        font.setFamily('微软雅黑')
        font.setBold(True)
        font.setPointSize(12)
        font.setWeight(60)

        self.combo = QComboBox(self)
        self.combo.addItem('Local Mode')
        self.combo.addItem('Server Mode')
        self.combo.setFont(font)
        self.combo.setGeometry(QtCore.QRect(1060, 60, 150, 40))

        btn = QPushButton(self)
        btn.setText("Select Image")
        # btn.move(1060, 140)
        btn.clicked.connect(self.openimage)
        btn.setFont(font)
        btn.setGeometry(QtCore.QRect(1060, 140, 150, 40))

        btn2 = QPushButton(self)
        btn2.setText("Choose Object")
        # btn2.move(1060, 380)
        btn2.clicked.connect(self.screenshot)
        btn2.setFont(font)
        btn2.setGeometry(QtCore.QRect(1060, 220, 150, 40))

        btn3 = QPushButton(self)
        btn3.setText("Annotate")
        # btn3.move(1060, 460)
        btn3.clicked.connect(self.labelling)
        btn3.setFont(font)
        btn3.setGeometry(QtCore.QRect(1060, 300, 150, 40))

        self.pic_x = 0
        self.pic_y = 0

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "Open Image", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.points_list = []
        self.label.setPixmap(jpg)

    def screenshot(self):
        self.x = self.label.x() + self.geometry().x()
        self.y = self.label.y() + self.geometry().y()
        a = WScreenShot(self.x, self.y)
        a.run(self.x, self.y)

    def getPoints(self, lt):
        points = []
        temp = 0
        print(len(lt))
        for i in range(len(lt)):
            if i % 2:
                points.append(QPoint(temp, lt[i]))
            else:
                temp = lt[i]
        return points

    def labelling(self):
        global cur_ins_id
        global record
        global select
        cur_ins_id += 1

        select = self.combo.currentText()
        ret = deep_rnn_annotate()
        for i in range(0, len(ret), 2):
            ret[i] = ret[i] + record[0]
            ret[i+1] = ret[i+1] + record[1]
        points = self.getPoints(ret)
        self.label.addPolygon(points)
        self.repaint()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = MyWindow()
    my.show()
    sys.exit(app.exec_())
