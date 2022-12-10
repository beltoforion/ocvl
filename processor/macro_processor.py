from processor.processor_base import ProcessorBase
import numpy as np
import cv2

class MacroProcessor(ProcessorBase):
    def __init__(self):
        super(MacroProcessor, self).__init__("MacroProcessor")      
        self._processors = []

    def add(self, processor : ProcessorBase):
        if processor is None:
            return

        self._processors.append(processor)

    def process(self, image : np.array) -> np.array:
        ct = 0
        for p in self._processors:
            image = p.process(image)
            ct += 1
#            cv2.imwrite(f'process_{ct}_{p.name}.jpg', image)

        return image            
