import sys
import cv2
import numpy as np

from config import global_config
from calibration import Calibration, resize

class Augmentor():
    def __init__(self):
        self.cali = Calibration()
    
    def WordProjection(self, lib):
        if global_config.check_files() == False:
            return

        retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.cali.cali_camera()

        # load open cv alphabet lib
        fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

        # generate objectPoints
        objectPoints = []
        shiftX = 7
        shiftY = 5
        for (j, token) in enumerate(global_config.WordInput):
            k = 0
            data = fs.getNode(token).mat().reshape(-1)
            while k < len(data):
                objectPoints.append([
                    data[k+1] + shiftY,
                    data[k] + shiftX,
                    -data[k+2]
                ])
                k += 3
            shiftX -= 3
            if j == 2:
                shiftX = 7
                shiftY -= 3

        # for every image
        for (j, file) in enumerate(global_config.files):
            # get project points
            imagePoints, jacobian = cv2.projectPoints(np.array(
                objectPoints, dtype='float'), rvecs[j], tvecs[j], cameraMatrix, distCoeffs)

            # load original image
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)

            # draw lines
            i = 0
            while i < len(imagePoints):
                cv2.line(img,
                         (int(imagePoints[i][0][0]),
                          int(imagePoints[i][0][1])),
                         (int(imagePoints[i+1][0][0]),
                          int(imagePoints[i+1][0][1])),
                         (0, 0, 255), 10)
                i += 2

            cv2.imshow("AR", resize(img))
            cv2.waitKey(1000)
    
    def WordProject2D(self):
        self.WordProjection("./Dataset_CvDl_Hw1/Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_onboard.txt")
    
    def WordProject3D(self):
        self.WordProjection("./Dataset_CvDl_Hw1/Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_vertical.txt")