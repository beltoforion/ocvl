import cv2
import numpy as np
from enum import Enum
from numba import jit

from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.input_output import Input, Output
from ocvl.processor.io_data import IoData


class RgbJoinMethod(Enum):
    COLOR = 1,
    THREE_COLOR = 2


class RbgJoinProcessor(ProcessorBase):
    def __init__(self):
        super(RbgJoinProcessor, self).__init__("RbgJoinProcessor")      

        self.__method = RgbJoinMethod.COLOR

        # one input
        self._add_input(Input(self))
        self._add_input(Input(self))
        self._add_input(Input(self))

        # three outputs one for each channel
        self._add_output(Output())

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, value):
        self.__method = value

    def __rgbify(self, image):
        h, w, d = image.shape
        nh = int(h * 2)
        nw = int(w * 1.5)
        
        rgb = np.zeros((nh, nw, 3), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                # Read rgb values of two neighboring pixels from the input image
                b, g, r = image[y, x, :]

                if (x%2 == 0):
                    ox = (int)(x*1.5)
                    rgb[2*y + 0, ox,     :] = (0, g, 0)
                    rgb[2*y + 0, ox + 1, :] = (b, 0, 0)
                    rgb[2*y + 1, ox,     :] = (0, 0, r)
                else:
                    ox = 1+(int)(x*1.5 - 0.5)
                    rgb[2*y + 0, ox,     :] = (0,  0,  r)
                    rgb[2*y + 1, ox,     :] = (b,  0,  0)
                    rgb[2*y + 1, ox - 1, :] = (0,  g,  0)

        return rgb


    def process(self):
        ch_blue  = self._inputs[0].data.image
        ch_green = self._inputs[1].data.image
        ch_red   = self._inputs[2].data.image
        
        image_color = cv2.merge((ch_blue, ch_green, ch_red))

        if self.__method==RgbJoinMethod.COLOR:
            # nothing to do here
            pass
        elif self.__method == RgbJoinMethod.THREE_COLOR:
            image_color = self.__rgbify(image_color)
        else:
            raise NotImplementedError(f"RgbJoinMethod {self.__method} is not implemented!")            
        
        self._outputs[0].set(IoData(image_color))

