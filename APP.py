import sys
import os
from NoiseRestoration.filters import noise_gaussian, median_filter
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QApplication, QLabel, QRadioButton, QListWidget, QListWidgetItem, QStatusBar, QVBoxLayout
from PyQt5 import QtGui, uic
from PyQt5.QtGui import QImage, QPixmap
import cv2 as cv

def cv2qim(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    q_image = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    return q_image

class NoiseRestorationAPP(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('APP.ui', self)

        # main variables
        self.img_width = int(app.primaryScreen().size().width()/2 - 200)
        self.img_height = int(app.primaryScreen().size().height()/2 - 20)

        # defining widgets
        self.images_list = self.findChild(QListWidget, 'images_list')
        self.original_label = self.findChild(QLabel, 'original_label')
        self.noisy_label = self.findChild(QLabel, 'noisy_label')
        self.restored_label = self.findChild(QLabel, 'restored_label')
        self.images_layout = self.findChild(QVBoxLayout, 'images_layout')

        # connecting signals
        self.images_list.itemClicked.connect(self.update_original)

        # initial functions
        self.populate_list("images\\")

    def update_original(self):
        item = self.images_list.currentItem()

        q_image = item.data(5)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.original_label.setPixmap(QPixmap(q_image))
        self.update_noisy()

    def update_noisy(self):
        original = self.images_list.currentItem().data(6)
        noisy = noise_gaussian(original, 5, 50)
        self.images_list.currentItem().setData(7, noisy)

        q_image = cv2qim(noisy)
        if q_image.width() > self.img_width:
            q_image = q_image.scaledToWidth(self.img_width)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.noisy_label.setPixmap(QPixmap(q_image))
        self.update_restored()

    def update_restored(self):
        noisy = self.images_list.currentItem().data(7)
        restored = median_filter(noisy, 5)
        self.images_list.currentItem().setData(8, restored)

        q_image = cv2qim(restored)
        if q_image.width() > self.img_width:
            q_image = q_image.scaledToWidth(self.img_width)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.restored_label.setPixmap(QPixmap(q_image))


    def populate_list(self, folder_path):
        self.images_list.clear()
        images = os.listdir(folder_path)

        for image in images:
            path = os.path.join(folder_path, image)

            icon = QtGui.QIcon()
            icon.addPixmap(QPixmap(path), QtGui.QIcon.Normal, QtGui.QIcon.Off)

            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(0, QPixmap(path))
            item.setData(5, QImage(path))
            # Original, Noisy, Restored (6, 7, 8)
            item.setData(6, cv.imread(path))
            self.images_list.addItem(item)

        self.images_list.setCurrentItem(self.images_list.item(0))
        self.update_original()
        self.showMaximized()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = NoiseRestorationAPP()
    demo.show()
    sys.exit(app.exec_())
