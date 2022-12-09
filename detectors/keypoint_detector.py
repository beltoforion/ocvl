import numpy as np
import cv2

from detectors.detector_base import DetectorBase


class KeypointDetector(DetectorBase):
    def __init__(self):
        super(KeypointDetector, self).__init__("KeypointDetector")       

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

#        if threshold is None:
#            threshold = self._threshold

        print(f'{self.name}.search()')
        orb = cv2.ORB_create()
        keypoints, _ = orb.detectAndCompute(image, None)

        result = [(kp.pt[0], kp.pt[1], kp.size/2, kp.size/2, kp.response, 0) for kp in keypoints]

        # The result still contains keypoint clusters centered at the same position
        # we now run nonmax suppression to clean this up. I dont want any overlap.
        return np.array(result)