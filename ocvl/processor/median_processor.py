import cv2

from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.input_output import *


class MedianProcessor(ProcessorBase):
    def __init__(self, size = 3):
        super(MedianProcessor, self).__init__("MedianProcessor")      
        self.__size = size
        
        self._add_input(Input(self))
        self._add_output(Output())


    @property
    def size(self):
        return self.__size


    @size.setter
    def size(self, value):
        self.__size = value


    def process(self):
        image = self._inputs[0].data.image
        image = cv2.medianBlur(image, self.__size)

        self._outputs[0].set(IoData(image))
