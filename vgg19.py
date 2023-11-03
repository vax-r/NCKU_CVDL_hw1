import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision.transforms import v2
import torchvision.models as models
import torchvision
import torchsummary
import PIL
from PyQt5.QtWidgets import (QFileDialog)
from PyQt5 import QtGui, QtCore

from tqdm import tqdm

from config import global_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

batch_size = 64

testset = torchvision.datasets.CIFAR10(root='./VGG19/data', train=False,
                                    download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

class VGG19():
    def __init__(self):
        self.infImage = None
        self.Model_PATH = "./VGG19/cifar_net.pth"
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def showAugmentationImg(self):
        fnames = ["automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30),
        ])

        to_tensor = v2.Compose([
            v2.PILToTensor()
        ])

        to_img = v2.ToPILImage()

        imgs = []
        for fname in fnames:
            fpath = "./Dataset_CvDl_Hw1/Dataset_CvDl_Hw1/Q5_image/Q5_1/" + fname + ".png"
            im = PIL.Image.open(fpath)
            im = to_tensor(im)
            imgs.append(transforms(im))

        fig = plt.figure()
        for ind, img in enumerate(imgs):
            plt.subplot(3, 3, ind+1)
            plt.imshow(to_img(img))
        plt.show()
        
    # Print model structure
    def Print_model_struct(self):

        model = models.vgg19_bn(num_classes=10)
        model = model.to(device)

        print(torchsummary.summary(model, input_size=(3 ,32, 32)))

    # show model training acc and loss
    def show_acc_loss(self):
        acc_fig = mpimg.imread("./VGG19/Accuracy.png")
        loss_fig = mpimg.imread("./VGG19/Loss.png")
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(2,1,1)
        plt.imshow(acc_fig)
        fig.add_subplot(2,1,2)
        plt.imshow(loss_fig)
        plt.show()

    def load_infImage(self):
        filename, filetype = QFileDialog.getOpenFileName()
        if filename:
            global_config.infImg = filename
        
        pixmap = QtGui.QPixmap(global_config.infImg)
        scaled_pixmap = pixmap.scaled(128, 128, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        global_config.qt_img_lable.setPixmap(scaled_pixmap)

        # print(filename)

    def inference(self):
        if global_config.infImg is None:
            print("Please load the image to be infereced first")
            return
        
        model = models.vgg19_bn(num_classes=10)
        model.load_state_dict(torch.load(self.Model_PATH, map_location=torch.device(device)))

        img = cv2.imread(global_config.infImg)
        img = cv2.imdecode(np.fromfile(global_config.infImg, dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = v2.Compose(
            [
                v2.Resize((32, 32)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        with torch.no_grad():
            model.eval()

            img = transform(img)
            img=img.unsqueeze(0).to(device)

            outputs = model(img)

            _, pred = torch.max(outputs, 1)

            out_str = "predicted = " + str(self.classes[pred[0]])
            global_config.img_pred_label.setText(out_str)

            prob = [0 if i < 0 else i for i in outputs[0].tolist()]
            plt.bar([*range(10)], prob, tick_label=self.classes, width=0.5)
            plt.show()


    def loadImage(self):
        return
