import sys
import cv2
import numpy as np

from config import global_config

class SIFT():
    def __init__(self):
        self.algorithm = cv2.SIFT_create()
    
    def keypoints(self):
        if global_config.image1 is None:
            print("Image 1 not loaded")
            return
        
        img = cv2.imdecode(np.fromfile(global_config.image1, dtype=np.uint8), 1)
        self.algorithm = cv2.SIFT_create()

        keypoints, _ = self.algorithm.detectAndCompute(img, None)

        res = cv2.drawKeypoints(img, keypoints, None, (0, 255, 0))
        cv2.namedWindow("4-1 keypoints", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("4-1 keypoints", res)
    
    def matchedKeypoints(self):
        if global_config.image1 is None or global_config.image2 is None:
            print("Both image 1 and image 2 need to be loaded first")
            return
        img1 = cv2.imdecode(np.fromfile(global_config.image1, dtype=np.uint8), 1)
        img2 = cv2.imdecode(np.fromfile(global_config.image2, dtype=np.uint8), 1)
        self.algorithm = cv2.SIFT_create()
        key1, des1 = self.algorithm.detectAndCompute(img1, None)
        key2, des2 = self.algorithm.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        matcher = cv2.FlannBasedMatcher(index_params,search_params)
        rowMatched = matcher.knnMatch(des1, des2, k = 2)
        matchedOneToTwo = []

        for (m, n) in rowMatched:
            if m.distance < n.distance * 0.7:
                matchedOneToTwo.append((m, n))
        
        img3 = cv2.drawMatchesKnn(
            img1, key1, img2, key2, matchedOneToTwo, None, (0, 255, 255), (0, 255, 0), None) 
        cv2.namedWindow("4-2 matched keypoints", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("4-2 matched keypoints", img3)