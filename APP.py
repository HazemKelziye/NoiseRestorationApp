import sys
import os
from NoiseRestoration.filters import noise_gaussian, noise_gamma, noise_rayleigh, noise_uniform, noise_exponential, noise_salt_pepper, median_filter
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QButtonGroup, QSlider
from PyQt5 import QtGui, uic
from PyQt5.QtGui import QImage, QPixmap
import cv2 as cv
from PyQt5.QtCore import pyqtSignal, Qt

def cv2qim(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    q_image = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    return q_image

class NoiseRestorationAPP(QMainWindow):
    noise_signal = pyqtSignal(int, int, int)

    def __init__(self):
        super().__init__()
        uic.loadUi('APP.ui', self)

        # main variables
        self.img_width = int(app.primaryScreen().size().width()/2 - 200)
        self.img_height = int(app.primaryScreen().size().height()/2 - 20)
        self.original = None
        self.noisy = None

        # defining widgets
        self.images_list = self.findChild(QListWidget, 'images_list')
        self.original_label = self.findChild(QLabel, 'original_label')
        self.noisy_label = self.findChild(QLabel, 'noisy_label')
        self.restored_label = self.findChild(QLabel, 'restored_label')
        self.images_layout = self.findChild(QVBoxLayout, 'images_layout')
        self.noise_radio_group = self.findChild(QButtonGroup, 'noise_radio_group')
        self.noiseProp1_label = self.findChild(QLabel, 'noiseProp1_label')
        self.noiseProp2_label = self.findChild(QLabel, 'noiseProp2_label')
        self.noiseProp1_slider = self.findChild(QSlider, 'noiseProp1_slider')
        self.noiseProp2_slider = self.findChild(QSlider, 'noiseProp2_slider')
        self.prop1Value_label = self.findChild(QLabel, 'prop1Value_label')
        self.prop2Value_label = self.findChild(QLabel, 'prop2Value_label')

        # connecting signals
        self.images_list.itemClicked.connect(self.update_original)
        self.noise_radio_group.idClicked.connect(self.update_noise_gui)
        self.noise_signal.connect(self.update_noisy)
        self.noiseProp1_slider.valueChanged.connect(self.noise_slider_moved)
        self.noiseProp2_slider.valueChanged.connect(self.noise_slider_moved)

        # initial functions
        self.populate_list("images\\")

    def update_noise_gui(self, button):

        # gaussian
        if button == -2:
            self.noiseProp2_label.show()
            self.prop2Value_label.show()
            self.noiseProp2_slider.show()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 10)
            self.noiseProp1_slider.setTickInterval(1)

            self.noiseProp2_label.setText("Variance =")
            self.noiseProp2_slider.setRange(0, 100)
            self.noiseProp2_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-2, prop1, prop2)

        # gamma
        elif button == -3:
            self.noiseProp2_label.show()
            self.prop2Value_label.show()
            self.noiseProp2_slider.show()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 100)
            self.noiseProp1_slider.setTickInterval(1)

            self.noiseProp2_label.setText("Variance =")
            self.noiseProp2_slider.setRange(0, 100)
            self.noiseProp2_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-3, prop1, prop2)

        # Salt and pepper
        elif button == -4:
            self.noiseProp2_label.hide()
            self.noiseProp2_slider.hide()
            self.prop2Value_label.hide()

            self.noiseProp1_label.setText("Probability =")
            self.noiseProp1_slider.setRange(0, 10)
            self.noiseProp1_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-4, prop1, prop2)

        # Exponential
        elif button == -5:
            self.noiseProp2_label.hide()
            self.prop2Value_label.hide()
            self.noiseProp2_slider.hide()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 100)
            self.noiseProp1_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-5, prop1, prop2)

        # Rayleigh
        elif button == -6:
            self.noiseProp2_label.hide()
            self.noiseProp2_slider.hide()
            self.prop2Value_label.hide()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 100)
            self.noiseProp1_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-6, prop1, prop2)

        # uniform
        elif button == -7:
            self.noiseProp2_label.show()
            self.noiseProp2_slider.show()
            self.prop1Value_label.show()
            self.prop2Value_label.show()

            self.noiseProp1_label.setText("a =")
            self.noiseProp1_slider.setRange(0, 255)
            self.noiseProp1_slider.setTickInterval(1)

            self.noiseProp2_label.setText("b =")
            self.noiseProp2_slider.setRange(0, 255)
            self.noiseProp2_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-2, prop1, prop2)

    def noise_slider_moved(self):
        filter = self.noise_radio_group.checkedId()
        prop1 = self.noiseProp1_slider.sliderPosition()
        prop2 = self.noiseProp2_slider.sliderPosition()
        self.noise_signal.emit(filter, prop1, prop2)

    def update_original(self):
        item = self.images_list.currentItem()

        image = item.data(Qt.UserRole)
        q_image = cv2qim(image)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.original_label.setPixmap(QPixmap(q_image))
        self.noise_slider_moved()

    def update_noisy(self, noise_filter, prop1, prop2):
        original = self.images_list.currentItem().data(Qt.UserRole)

        if noise_filter == -2:
            noisy = noise_gaussian(original, prop1, prop2)

        elif noise_filter == -3:
            prop1 /= 10
            noisy = noise_gamma(original, prop1, prop2)

        elif noise_filter == -4:
            prop1 /= 100
            noisy = noise_salt_pepper(original, prop1)

        elif noise_filter == -5:
            noisy = noise_exponential(original, prop1)

        elif noise_filter == -6:
            noisy = noise_rayleigh(original, prop1)

        elif noise_filter == -7:
            if prop2 < prop2:
                prop2 = prop1
            noisy = noise_uniform(original, prop1, prop2)

        else:
            noisy = self.original
            print("couldn't specify filter type")

        self.prop1Value_label.setText(str(prop1))
        self.prop2Value_label.setText(str(prop2))

        q_image = cv2qim(noisy)
        if q_image.width() > self.img_width:
            q_image = q_image.scaledToWidth(self.img_width)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.noisy_label.setPixmap(QPixmap(q_image))
        #self.update_restored()

    def update_restored(self, prop1, prop2, prop3):
        self.noisy = self.images_list.currentItem().data(7)

        restored = median_filter(self.noisy, 5)
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
            item.setData(Qt.UserRole, cv.imread(path))
            self.images_list.addItem(item)

        self.images_list.setCurrentItem(self.images_list.item(0))
        self.update_original()
        self.noise_signal.emit(-2, 0, 0)
        self.showMaximized()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = NoiseRestorationAPP()
    demo.show()
    sys.exit(app.exec_())
