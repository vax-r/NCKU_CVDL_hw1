import sys
import cv2
import numpy as np

from config import global_config

CV2PreviewSize = 950

def resize(img):
    height, width = img.shape[0], img.shape[1]
    height = int(height * CV2PreviewSize / width)
    width = CV2PreviewSize
    return cv2.resize(img, (width, height))


class Calibration():
    W = 11
    H = 8
    result = ()

    def __init__(self):
        self.W = 11
        self.H = 8
        self.result = ()

    def corner_detect(self):
        if global_config.files:
            print("Detecting corners...")
        else:
            print("No files found")
            return

        for file in global_config.files:
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
            width, height = img.shape[0], img.shape[1]
            retval, corners = cv2.findChessboardCorners(
                img, (self.W, self.H), None)

            if retval:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(
                    img, (self.W, self.H), corners, retval)
            cv2.imshow('Corner Detection', resize(img))
            cv2.waitKey(500)