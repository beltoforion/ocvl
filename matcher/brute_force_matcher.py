import sys
import numpy as np
import cv2

import matplotlib.pyplot as plt

from matcher.match_exception import *
from matcher.matcher_base import MatcherBase


class BruteForceMatcher(MatcherBase):
    def __init__(self):
        super(BruteForceMatcher, self).__init__("BruteForceMatcher") 

    def match(self, ref_data, data):
        src = np.array(ref_data.T, copy=True).astype(np.float32)
        dst = np.array(data.T, copy=True).astype(np.float32)
        
        knn = cv2.ml.KNearest_create()
        responses = np.array(range(src.shape[0])).astype(np.float32)
        knn.train(src, cv2.ml.ROW_SAMPLE, responses)

        # approximate the point cloud bounding box size. We will use
        # this value for estimating reasonable limits to estimate the
        # match quality
        scale_x = np.max(src[:,0]) - np.min(src[:,0])
        scale_y = np.max(src[:,1]) - np.min(src[:,1])
        scale = max(scale_x, scale_y)

        discard_threshold = (scale / 50)**2

        # 
        pt_ref = src[0]

        best_idx = -1
        best_median = sys.maxsize
        best_shift = None

        plt.ion()

        for i in range(dst.shape[0]):
            points = dst.copy()

            # subtract distance of point i of the dst set to point 0 of 
            # the reference set. This moves point i of the second set on top of 
            # point 0 of the source set.
            shift = points[i] - pt_ref
            points -= shift

            # find nearest
            ret, results, neighbours, dist = knn.findNearest(points, 1)
            median_dist = np.median(dist)

#            print(f'{i}: best={best_idx}; len={len(dist)}; median={median_dist};')

#            plt.clf()
#            plt.plot(src[:,0], src[:,1], marker='o',linestyle='None')
#            plt.plot(points[:,0], points[:,1], marker='x',linestyle='None')
#            plt.show() 

            if median_dist < best_median:
                best_idx = i
                best_median = median_dist

                # Figure out how many points are close enough to the corresponding 
                # points of the reference set.
                good_idx, _ = np.where(dist < discard_threshold)

                # compute the average distance vector of the good points to their reference 
                # points
                offset = np.array([0.0,0.0])
                for idx in good_idx:
                    ref_idx = int(results[idx,0])
                    offset += src[ref_idx] - points[idx]

                # average remaining offset
                offset /= len(good_idx)
                best_shift = offset - shift




        trans = np.array([[1, 0, best_shift[0]],
                          [0, 1, best_shift[1]],
                          [0, 0, 1]])

        plt.clf()
#        plt.plot(dst[:,0], dst[:,1], marker='o',linestyle='None')
        dst += best_shift
        plt.plot(src[:,0], src[:,1], marker='o',linestyle='None')
        plt.plot(dst[:,0], dst[:,1], marker='+',linestyle='None')
        plt.show()                           

        return trans


