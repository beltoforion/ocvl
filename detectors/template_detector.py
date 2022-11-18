import numpy as np
import cv2

from detectors.detector_base import DetectorBase
from imutils.object_detection import non_max_suppression


class TemplateDetector(DetectorBase):
    def __init__(self, method):
        super(TemplateDetector, self).__init__("TemplateDetector")       
        
        if method in [cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF]:
            raise Exception("serch requires a normalized algorithm!")

        self._method = method
        self._threshold = 0.95

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def after_load(self, file : str):
        print(f'{self.name}.after_load()')

    def search(self, image : np.array, threshold : float = None):
        if image is None:
            raise Exception('Image is null!')

        if self._method in [cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF]:
            raise Exception("serch requires a normalized algorithm!")

        print(f'{self.name}.search()')
        res = cv2.matchTemplate(image, self._pattern, self._method)
        if self._method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            res = 1 - res

        if threshold is None:
            threshold = self._threshold

        img_height, img_width = image.shape[:2]

        max_val = 1
        rects = []

        while max_val > threshold:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            x, y = max_loc
            if max_val > threshold:
                h1 = np.clip(max_loc[1] - self._height//2, 0, img_height)
                h2 = np.clip(max_loc[1] + self._height//2 + 1, 0, img_height)

                w1 = np.clip(max_loc[0] - self._width//2, 0, img_width)
                w2 = np.clip(max_loc[0] + self._width//2 + 1, 0, img_width)
                res[h1:h2, w1:w2] = 0   
                rects.append((x, y, x + self._width, y + self._height, max_val))

        return np.array(rects)


    def search2(self, image : np.array, threshold : float = None):
        if image is None:
            raise Exception('Image is null!')

        print(f'{self.name}.search()')
        res = cv2.matchTemplate(image, self._pattern, self._method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if threshold is None:
            threshold = self._threshold

        (yCoords, xCoords) = np.where(res >= threshold)

        # loop over our starting (x, y)-coordinates
        rects = []    
        for (x, y) in zip(xCoords, yCoords):
            rects.append((x, y, x + self._width, y + self._height))

#        indexes = cv2.dnn.NMSBoxes(np.array(rects), score, threshold, self._nms) 

        # apply non-maxima suppression to the rectangles
        #pick = non_max_suppression_fast(np.array(rects), self._nms)
        pick = non_max_suppression(np.array(rects))
        #pick = self.NMS(rects)

        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        h, w = res.shape[:2]
        scale = 700/w
        buf = cv2.resize(image, (int(scale * w), int(scale * h)))
        cv2.imshow('found', buf)
        cv2.waitKey(0)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if self._method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + self._width, top_left[1] + self._height)
        cv2.rectangle(image, top_left, bottom_right, 255, 2)

        x : int = 0
        y : int = 0
        s : float = 0

        return (x, y, s)        