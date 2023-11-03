import sys
from os import listdir
from os.path import isfile, join, splitext
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
        QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QFileDialog)
from PyQt5 import QtGui, QtCore

class Config:
    
    files = []
    fileMapping = {}
    extrinsicAttentionImageIndex = 0
    WordInput = "CAMERA"
    imageR = None
    imageL = None
    
    def __init__(self):
        self.files = []
        self.fileMapping = {}
        self.extrinsicAttentionImageIndex = 0
        self.WordInput = "CAMERA"
        self.imageR = None
        self.imageL = None
        self.cali = ()

        self.image1 = None
        self.image2 = None

        # img for Q5
        self.augImgs = []
        self.infImg = None
        self.qt_img_lable = None
        self.img_pred_label = None
    
    def load_files(self):
        folderPath = QFileDialog.getExistingDirectory()
        files = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        files.sort(key=lambda x: int(splitext(x)[0]))
        files = [join(folderPath, f) for f in files]

        # reload files and clibration result
        self.files = files
        self.cali = ()
        # print(str(self.files))

        return files
    
    def load_imageL(self):
        filename, filetype = QFileDialog.getOpenFileName()
        if filename:
            self.imageL = filename
    
    def load_imageR(self):
        filename, filetype = QFileDialog.getOpenFileName()
        if filename:
            self.imageR = filename
    
    def load_image1(self):
        filename, filetype = QFileDialog.getOpenFileName()
        if filename:
            self.image1 = filename

    def load_image2(self):
        filename, filetype = QFileDialog.getOpenFileName()
        if filename:
            self.image2 = filename

    def load_augImgs(self):
        folderPath = QFileDialog.getExistingDirectory()
        files = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        # files.sort(key=lambda x: int(splitext(x)[0]))
        files = [join(folderPath, f) for f in files]

        # reload files
        self.augImgs = files
        print(str(self.augImgs))

        return files

    # def load_augImg(self):
    #     filename, filetype = QFileDialog.getOpenFileName()
    #     if filename:
    #         self.augImg = filename
    #     pixmap = QtGui.QPixmap(self.augImg)
    #     pixmap = pixmap.scaled(200, 200)
    #     self.qt_img_lable.setPixmap(pixmap)

    def check_files(self):
        if self.files == None or len(self.files) == 0:
            print("No files selected")
            return False
        return True

# Init global config
global_config = Config()
