from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.input_output import *

import cv2


class RgbSplitProcessor(ProcessorBase):
    def __init__(self):
        super(RgbSplitProcessor, self).__init__("RgbSplitProcessor")      
        # one input
        self._add_input(Input(self))

        # three outputs one for each channel
        self._add_output(Output())
        self._add_output(Output())
        self._add_output(Output())                


    def process(self):
        image = self._inputs[0].data.image

        ch_blue, ch_green, ch_red = cv2.split(image)

        self._outputs[0].set(IoData(ch_blue))
        self._outputs[1].set(IoData(ch_green))
        self._outputs[2].set(IoData(ch_red))                