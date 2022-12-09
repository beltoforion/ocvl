from processor.processor_base import ProcessorBase
import numpy as np
import cv2


class MedianProcessor(ProcessorBase):
    def __init__(self, size):
        super(MedianProcessor, self).__init__("MedianProcessor")      
        self._size = size

    def process(self, image : np.array) -> np.array:
        image = cv2.medianBlur(image, self._size)
        return image