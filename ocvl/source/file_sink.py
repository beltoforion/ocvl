import cv2
import os

from ocvl.processor.input_output import *
from ocvl.source.source_sink import *


class FileSink(Sink):
    def __init__(self):
        super(FileSink, self).__init__()
        
        self.__output_path = "out.png"
        self.__output_format = OutputFormat.SAME_AS_INPUT

        self._add_input(Input(self))

    @property
    def output_format(self):
        return self.__output_format
    
    @output_format.setter
    def output_format(self, output_format):
        self.__output_format = output_format

    @property
    def output_path(self):
        return self.__output_path
    
    @output_path.setter
    def output_path(self, output_path):
        self.__output_path = output_path

    def process(self):
        file_name, file_ext = os.path.splitext(self.output_path)
        
        if self.__output_format==OutputFormat.SAME_AS_INPUT:
            output = file_name + file_ext
        elif self.__output_format==OutputFormat.PNG:
            output = file_name + ".png"
        else:
            raise(Exception("Invalid output format"))
        
        cv2.imwrite(output, self.input[0].data.image)
        cv2.imshow(output, self.input[0].data.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def end_of_series(self):
        pass

