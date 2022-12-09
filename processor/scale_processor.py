from processor.processor_base import ProcessorBase
import numpy as np
import cv2


class ScaleProcessor(ProcessorBase):
    def __init__(self, scale = 0.5):
        super(ScaleProcessor, self).__init__("ScaleProcessor")      
        self._scale = scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def process(self, image : np.array) -> np.array:
        h, w = image.shape[:2]
        image_scaled = cv2.resize(image, (int(w*self._scale), int(h*self._scale) ), interpolation=cv2.INTER_CUBIC)
        return image_scaled