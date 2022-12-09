import numpy as np
import cv2

from detectors.detector_base import DetectorBase


class BlobDetector(DetectorBase):
    def __init__(self, params = None):
        super(BlobDetector, self).__init__("BlobDetector")       

        if params == None:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByColor = False
            params.blobColor = 255
            params.minThreshold = 100
            params.maxThreshold = 255
            params.filterByArea = True
            params.minArea = 2
            params.maxArea = 10000
            params.filterByCircularity = True
            params.minCircularity = 0.8
            params.filterByConvexity = False
            params.filterByInertia = False

        self.detector = cv2.SimpleBlobDetector_create(params)

    def after_load(self, file : str):
        print(f'{self.name}.after_load()')

    def search(self, image : np.array, threshold : float = None):
        if image is None:
            raise Exception('Image is null!')

        print(f'{self.name}.search()')
        blobs = self.detector.detect(image, None)

        # return format is array of 
        #   x, y, width, height, score, classid
        # per detected blob
        points = np.array([(point.pt[0], point.pt[1], point.size, point.size, point.response, 0) for point in blobs], dtype="float")

        return points