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
        if global_config.check_files() == False:
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
        
        return "OK"

    def cali_camera(self):
        imgPoints = []
        objPoints = [] 
        if global_config.check_files() == False:
            return
                
        for file in global_config.files:
            imgP = []
            objP = []
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
                for (j, corner) in enumerate(corners):
                    imgP.append([corner[0][0], corner[0][1]])
                    objP.append(
                        [int(j/self.W), j % self.W, 0])
            imgPoints.append(imgP)
            objPoints.append(objP)
        
        global_config.cali = cv2.calibrateCamera(
            np.array(objPoints, dtype='float32'),
            np.array(imgPoints, dtype='float32'),
            (width, height), None, None)
        
        return global_config.cali

    def IntrinMtr(self):
        if global_config.check_files() == False:
            return
        if not global_config.cali:
            self.cali_camera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = global_config.cali
        print("Intrinsic:")
        print(cameraMatrix)
        return cameraMatrix

    def ExtrinMtr(self, i):
        if global_config.check_files() == False:
            return
        if not global_config.cali:
            self.cali_camera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = global_config.cali

        i = int(i)
        R = np.zeros((3, 3), dtype='float')
        cv2.Rodrigues(rvecs[i], R)
        t = np.array(tvecs[i])

        print("Extrinsic:")
        print(np.concatenate((R, t), axis=1))
    
    def Distortion(self):
        if global_config.check_files() == False:
            return
        if not global_config.cali:
            self.cali_camera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = global_config.cali
        print("Distortion:")
        print(distCoeffs)
        return distCoeffs

    def Undistort(self):
        if global_config.check_files() == False:
            return
        if not global_config.cali:
            self.cali_camera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = global_config.cali
        for file in global_config.files:
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
            dest = cv2.undistort(img, cameraMatrix, distCoeffs)
            compare = np.concatenate((img, dest), axis=1)
            cv2.imshow("Results", resize(compare))
            cv2.waitKey(500)