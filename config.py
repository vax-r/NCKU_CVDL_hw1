import sys
from os import listdir
from os.path import isfile, join, splitext
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
        QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QFileDialog)

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
    
    def check_files(self):
        if self.files == None or len(self.files) == 0:
            print("No files selected")
            return False
        return True

# Init global config
global_config = Config()
