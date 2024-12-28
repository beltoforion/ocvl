import sys
import numpy as np
import cv2

from matcher.match_exception import *
from matcher.matcher_base import MatcherBase

class IcpMatcher(MatcherBase):
    def __init__(self, max_iterations = 100, median_threshold = 10):
        super(IcpMatcher, self).__init__("IcpMatcher")      
        self.__max_iterations = max_iterations
        self.__median_threshold = median_threshold

    def __del_miss(self, dist, max_dist, th_rate = 0.8):
        th_dist = max_dist * th_rate
        return np.where(dist.T[0] < th_dist)

    def __is_converge(self, Tr, scale):
        delta_angle = 0.0001
        delta_scale = scale * 0.0001
        
        min_cos = 1 - delta_angle
        max_cos = 1 + delta_angle
        min_sin = -delta_angle
        max_sin = delta_angle
        min_move = -delta_scale
        max_move = delta_scale
        
        return min_cos < Tr[0, 0] and Tr[0, 0] < max_cos and \
            min_cos < Tr[1, 1] and Tr[1, 1] < max_cos and \
            min_sin < -Tr[1, 0] and -Tr[1, 0] < max_sin and \
            min_sin < Tr[0, 1] and Tr[0, 1] < max_sin and \
            min_move < Tr[0, 2] and Tr[0, 2] < max_move and \
            min_move < Tr[1, 2] and Tr[1, 2] < max_move


    @property
    def max_iterations(self):
        return self.__max_iterations


    @max_iterations.setter
    def max_iterations(self, value):
        self.__max_iterations = value


    @property
    def median_threshold(self):
        return self.__median_threshold


    @median_threshold.setter
    def median_threshold(self, value):
        self.__median_threshold = value


    def match(self, d1, d2):
        src = np.array([d1.T], copy=True).astype(np.float32)
        dst = np.array([d2.T], copy=True).astype(np.float32)
        
        knn = cv2.ml.KNearest_create()
        responses = np.array(range(len(d1[0]))).astype(np.float32)
        knn.train(src[0], cv2.ml.ROW_SAMPLE, responses)
            
        trans = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

        max_dist = sys.maxsize
        scale_x = np.max(d1[0]) - np.min(d1[0])
        scale_y = np.max(d1[1]) - np.min(d1[1])
        scale = max(scale_x, scale_y)

        for i in range(self.__max_iterations):
            ret, results, neighbours, dist = knn.findNearest(dst[0], 1)

            median_dist = np.median(dist)

            indeces = results.astype(np.int32).T     
            keep_idx = self.__del_miss(dist, max_dist)  
            indeces = np.array([indeces[0][keep_idx]])       
            dst = dst[0, keep_idx]

            T, T2 = cv2.estimateAffinePartial2D(dst[0], src[0, indeces],  True, method=cv2.RANSAC, confidence=0.995, maxIters=2000)
            if T is None:
                raise MatchException(MatchError.AFFINE_MATCH_FAILED)

            max_dist = np.max(dist)
            dst = cv2.transform(dst, T)
            trans = np.dot(np.vstack((T,[0,0,1])), trans)

            if self.__is_converge(T, scale):
                if median_dist > self.__median_threshold:
                    # The point cloud may have been matched but the median distance is too big
                    raise MatchException(MatchError.MEDIAN_DISTANCE_EXCEEDED)
                else:
                    return trans

        return None