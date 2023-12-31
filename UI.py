import sys
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
        QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QFileDialog, QComboBox, QLabel, QLineEdit)

from config import global_config
from calibration import Calibration
from augmentor import Augmentor
from stereomap import StereoMap
from sift import SIFT
from vgg19 import VGG19

local_cali = Calibration()
local_aug = Augmentor()
local_stereo = StereoMap()
local_sift = SIFT()
local_vgg = VGG19()

class UI(QWidget):
    
    def __init__(self, parent=None):
        super(UI, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.createSetting(), 0, 0)
        grid.addWidget(self.createQ1(), 0, 1)
        grid.addWidget(self.createQ2(), 0, 2)
        grid.addWidget(self.createQ3(), 0, 3)
        grid.addWidget(self.createQ4(), 1, 0)
        grid.addWidget(self.createQ5(), 1, 1)
        grid.addWidget(self.createInf(), 1, 2)

        self.setLayout(grid)

        self.setWindowTitle("CVDL HW1")


    def createSetting(self):
        groupBox = QGroupBox("Load Image")
        load_folder_button = QPushButton("Load Folder")
        load_rimage_button = QPushButton("Load Image R")
        load_limage_button = QPushButton("Load Image L")

        load_folder_button.clicked.connect(global_config.load_files)
        load_rimage_button.clicked.connect(global_config.load_imageR)
        load_limage_button.clicked.connect(global_config.load_imageL)

        vbox = QVBoxLayout()
        vbox.addWidget(load_folder_button)
        vbox.addWidget(load_rimage_button)
        vbox.addWidget(load_limage_button)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def createQ1(self):
        groupBox = QGroupBox("1. Camara Calibration")
        
        corners_button = QPushButton("1.1 Find corners")
        intrinsic_button = QPushButton("1.2 FInd intrinsic")

        extrinsic_combo = QComboBox()
        extrinsic_combo.addItems(['%d' % (i+1,) for i in range(0, 15)])
        extrinsic_box = QGroupBox("1.3 Find extrinsic")
        extrinsic_button = QPushButton("1.3 Find extrinsic")
        subBox = QVBoxLayout()
        subBox.addWidget(extrinsic_combo)
        subBox.addWidget(extrinsic_button)
        subBox.addStretch(1)
        extrinsic_box.setLayout(subBox)

        distortion_button = QPushButton("1.4 Find distortion")
        result_button = QPushButton("1.5 Show result")

        corners_button.clicked.connect(local_cali.corner_detect)
        intrinsic_button.clicked.connect(local_cali.IntrinMtr)
        extrinsic_button.clicked.connect(lambda: local_cali.ExtrinMtr(extrinsic_combo.currentText()))
        distortion_button.clicked.connect(local_cali.Distortion)
        result_button.clicked.connect(local_cali.Undistort)

        vbox = QVBoxLayout()
        vbox.addWidget(corners_button)
        vbox.addWidget(intrinsic_button)
        vbox.addWidget(extrinsic_box)
        vbox.addWidget(distortion_button)
        vbox.addWidget(result_button)

        groupBox.setLayout(vbox)

        return groupBox
    
    def createQ2(self):
        groupBox = QGroupBox("2. Augmented Reality")

        text_input = QLineEdit()
        text_input.setGeometry(0, 0, 100, 30)
        board_button = QPushButton("2.1 Show Words on Board")
        vertical_button = QPushButton("2.2 Show Words Vertically")

        text_input.textChanged.connect(local_aug.GetTextInput)
        board_button.clicked.connect(local_aug.WordProject2D)
        vertical_button.clicked.connect(local_aug.WordProject3D)

        vbox = QVBoxLayout()
        vbox.addWidget(text_input)
        vbox.addWidget(board_button)
        vbox.addWidget(vertical_button)

        groupBox.setLayout(vbox)

        return groupBox

    def createQ3(self):
        groupBox = QGroupBox("3. Stereo Disparity Map")

        map_button = QPushButton("3.1 Stereo Disparity Map")
        map_button.clicked.connect(local_stereo.stereoDisparityMap)

        vbox = QVBoxLayout()
        vbox.addWidget(map_button)

        groupBox.setLayout(vbox)

        return groupBox
    
    def createQ4(self):
        groupBox = QGroupBox("4. SIFT")

        image1_button = QPushButton("Load Image 1")
        image2_button = QPushButton("Load Image 2")
        keypoints_button = QPushButton("4.1 Keypoints")
        matched_button = QPushButton("4.2 Matched Keypoints")

        image1_button.clicked.connect(global_config.load_image1)
        image2_button.clicked.connect(global_config.load_image2)
        keypoints_button.clicked.connect(local_sift.keypoints)
        matched_button.clicked.connect(local_sift.matchedKeypoints)

        vbox = QVBoxLayout()
        vbox.addWidget(image1_button)
        vbox.addWidget(image2_button)
        vbox.addWidget(keypoints_button)
        vbox.addWidget(matched_button)

        groupBox.setLayout(vbox)
        return groupBox

    def createQ5(self):
        groupBox = QGroupBox("5. VGG19")

        image_button = QPushButton("Load Image")
        augmented_button = QPushButton("5.1 Show Augmented Images")
        model_button = QPushButton("5.2 Show Model Structure")
        accloss_button = QPushButton("5.3 Show acc and loss")
        interference_button = QPushButton("5.4 Interference")

        image_button.clicked.connect(local_vgg.load_infImage)
        augmented_button.clicked.connect(local_vgg.showAugmentationImg)
        model_button.clicked.connect(local_vgg.Print_model_struct)
        accloss_button.clicked.connect(local_vgg.show_acc_loss)
        interference_button.clicked.connect(local_vgg.inference)

        vbox = QVBoxLayout()
        vbox.addWidget(image_button)
        vbox.addWidget(augmented_button)
        vbox.addWidget(model_button)
        vbox.addWidget(accloss_button)
        vbox.addWidget(interference_button)

        groupBox.setLayout(vbox)
        return groupBox
    
    def createInf(self):
        groupBox = QGroupBox("Inference output")

        self.imgLabel = QLabel("", self)
        global_config.qt_img_lable = self.imgLabel
        self.predLabel = QLabel("", self)
        global_config.img_pred_label = self.predLabel

        vbox = QVBoxLayout()
        vbox.addWidget(self.imgLabel)
        vbox.addWidget(self.predLabel)

        groupBox.setLayout(vbox)
        return groupBox

# Draw Basic UI layouts
def CreateUI():
    app = QApplication(sys.argv)
    ui_Instance = UI()
    ui_Instance.show()
    sys.exit(app.exec_())