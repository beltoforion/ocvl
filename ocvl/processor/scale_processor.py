import cv2

from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.input_output import *


class ScaleProcessor(ProcessorBase):
    def __init__(self):
        super(ScaleProcessor, self).__init__("ScaleProcessor")      
        self.__interpolation = cv2.INTER_LINEAR
        self.__scale = -1
        self.__target_size = (1920, 1080)

        self._add_input(Input(self))
        self._add_output(Output())


    @property 
    def target_size(self):
        """ Get target size of the scale transformation.

            You cannot specify a scale and a target size at the same time.

          :param value: tuple of (width, height)
        """
        return self.__target_size


    @target_size.setter
    def target_size(self, value):
        """ Set target size of the scale transformation.

            You cannot specify a scale and a target size at the same time.

          :param value: tuple of (width, height)
        """
        self.__scale = -1
        self.__target_size = value


    @property
    def scale(self):
        return self.__scale


    @scale.setter
    def scale(self, value):
        self.__scale = value
        self.__target_size = None


    @property
    def interpolation(self):
        return self.__interpolation


    @interpolation.setter
    def interpolation(self, value):
        self.__interpolation = value


    def process(self):
        image = self._inputs[0].data.image
        h, w = image.shape[:2]
        
        if self.__scale == -1:
            w, h = self.__target_size
        else:
            w = int(w*self.__scale)
            h = int(h*self.__scale)

        image_scaled = cv2.resize(image, (int(w), int(h)), interpolation = self.interpolation)
        
        self._outputs[0].set(IoData(image_scaled))