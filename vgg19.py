import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

from config import global_config
import PIL

class VGG19():
    def __init__(self):
        pass

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
        


    def loadImage(self):
        return
