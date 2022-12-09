from processor.processor_base import ProcessorBase
import numpy as np
import cv2


class AdaptiveGuaussianThresholdProcessor(ProcessorBase):
    def __init__(self):
        super(AdaptiveGuaussianThresholdProcessor, self).__init__("AdaptiveGuaussianThresholdProcessor")      

    def process(self, image : np.array) -> np.array:
        th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -32) 
        return th