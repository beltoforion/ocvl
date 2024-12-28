from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.io_data import IoData
from ocvl.source.source_sink import *


class Output:
    def __init__(self):
        self.__inputs = []

    def set(self, data : IoData):
        if not isinstance(data, IoData):
           raise TypeError("Input data must be of type IoData")
        
        for i in self.__inputs:
            i.set(data)

    def connect(self, input):
        if input not in self.__inputs:
            self.__inputs.append(input)

            
class Input:
    def __init__(self, owner : Sink | None = None):
        self.__data = None
        self.__owner = owner


    @property
    def data(self):
        return self.__data   


    def set(self, data : IoData):
        if not isinstance(data, IoData):
           raise TypeError("Input data must be of type IoData")

        self.__data = data

        if self.__owner is not None:
            self.__owner.signal_input_ready(self.__data)


    def is_ready(self):
        return self.__data is not None        


    def clear(self):
        self.__data = None