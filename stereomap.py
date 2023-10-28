import cv2
import numpy as np

from config import global_config
from calibration import resize, CV2PreviewSize


class StereoMap():
    def __init__(self):
        self.imgL = None
        self.imgR = None
        self.disparity = 0
        self.scaleFactor = 0

    def stereoDisparityMap(self):
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        if global_config.imageL == None or global_config.imageR == None:
            print("Please load both imageL and imageR")
            return
        
        self.imgL = cv2.imdecode(np.fromfile(global_config.imageL, dtype=np.uint8), 1)
        self.imgR = cv2.imdecode(np.fromfile(global_config.imageR, dtype=np.uint8), 1)

        self.disparity = stereo.compute(
            cv2.cvtColor(self.imgL, cv2.COLOR_BGR2GRAY), 
            cv2.cvtColor(self.imgR, cv2.COLOR_BGR2GRAY))

        self.scaleFactor = self.imgL.shape[1] / CV2PreviewSize

        self.imgL = resize(self.imgL)
        self.imgR = resize(self.imgR)
        self.disparity = resize(self.disparity)

        # normalize
        disparityGray = cv2.normalize(
            self.disparity, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        cv2.imshow("imgL", self.imgL)
        cv2.namedWindow("imgL")
        cv2.setMouseCallback('imgL', self.clickOnCVCanvas)
        cv2.imshow("imgR", self.imgR)
        cv2.imshow("disparity", disparityGray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def clickOnCVCanvas(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x2 = int(x - self.disparity[y][x] / 16 / self.scaleFactor)
            imgTmp = np.copy(self.imgR)
            cv2.circle(imgTmp, (x2, y), 3, (0, 255, 0), -1)
            cv2.imshow("imgR", imgTmp)