from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.input_output import *

import cv2


class GreyscaleProcessor(ProcessorBase):
    def __init__(self):
        super(GreyscaleProcessor, self).__init__("GreyscaleProcessor")      
        self._add_input(Input(self))
        self._add_output(Output())

    def process(self):
        image = self._inputs[0].data.image

        if len(image.shape)==2:
            self._outputs[0].set(IoData(image))
        else:            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) 
            self._outputs[0].set(IoData(gray))
