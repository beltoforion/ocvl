import pathlib
import rawpy
import numpy as np
import cv2

from ocvl.processor.input_output import *
from ocvl.source.source_sink import *
      

class FileSource(Source):
    def __init__(self, file_path):
        super(FileSource, self).__init__("FileSource")
        
        self.__file_path = file_path
        self.__max_num_frames = -1
        self._add_output(Output())

    @property
    def file_path(self):
        return self.__file_path
    
    @file_path.setter
    def file_path(self, path):
        self.__file_path = path

    @property
    def max_num_frames(self):
        return self.__max_num_frames
    
    @max_num_frames.setter
    def max_num_frames(self, value):
        self.__max_num_frames = value

    def start(self):
        if not pathlib.Path(self.__file_path).exists():
            raise FileNotFoundError(f"Input file not found: {self.__file_path}")
        
        ext = pathlib.Path(self.__file_path).suffix
        if ext.lower()=='.mp4':
            cap = cv2.VideoCapture(self.__file_path)
            more_frames = True

            ct = 0
            while more_frames:
                more_frames, frame = cap.read()
                if not more_frames:
                    pass

                self.output[0].set(IoData(frame) if more_frames else IoData(None))    
                if not more_frames:                
                    cap.release()
                    break
                
                if ct>=self.__max_num_frames and self.__max_num_frames>=0:
                    cap.release()
                    break
                
                ct += 1

        elif ext.lower()=='.cr2':
            image : np.ndarray = rawpy.imread(self.__file_path).postprocess() 
        #		image : np.array = rawpy.imread(file).postprocess(output_bps=16) 
        #		image = np.float32(image) # image.astype(np.float32)
        #		image = image / 65535.0
            self.output[0].set(IoData(image))
        elif ext.lower()=='.jpg' or ext.lower()=='.png' or ext.lower()=='.jpeg':
            image : np.ndarray = cv2.imread(self.__file_path)
            self.output[0].set(IoData(image))
        else:
            raise ValueError("Unsupported file type: " + ext)
