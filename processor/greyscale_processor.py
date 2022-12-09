from processor.processor_base import ProcessorBase
import numpy as np
import cv2


class GreyscaleProcessor(ProcessorBase):
    def __init__(self):
        super(GreyscaleProcessor, self).__init__("GreyscaleProcessor")      

    def process(self, image : np.array) -> np.array:
        if len(image.shape)==2:
            return image
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) 
        return gray