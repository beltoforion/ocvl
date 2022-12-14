from processor.processor_base import ProcessorBase
import numpy as np
import cv2


class ShiftProcessor(ProcessorBase):
    def __init__(self, offset = np.array([0.0, 0.0])):
        super(ShiftProcessor, self).__init__("ShiftProcessor")      
        self.__offset = offset

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, value):
        self.__offset = value.copy()

    def process(self, image : np.array) -> np.array:
        M = np.float32([[1, 0, self.__offset[0]],
	                    [0, 1, self.__offset[1]]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))  
        return shifted