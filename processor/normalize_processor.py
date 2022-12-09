from processor.processor_base import ProcessorBase
import numpy as np
import cv2


class NormalizeProcessor(ProcessorBase):
    def __init__(self, scale = 0.5):
        super(NormalizeProcessor, self).__init__("NormalizeProcessor")      
        self._scale = scale


    def process(self, image : np.array) -> np.array:
        h, w = image.shape[:2]
        #result = cv2.equalizeHist(image)

#        clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(20,20))
#        result = clahe.apply(image)
        mean_r, _, _, _ = cv2.mean(image)
        result = cv2.convertScaleAbs(image, alpha=1.7, beta=-1.3*mean_r)
#        result = cv2.normalize(image,  None, 0, 255, cv2.NORM_MINMAX)
        return result