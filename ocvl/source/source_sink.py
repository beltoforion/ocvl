from abc import ABC, abstractmethod

from ocvl.processor.io_data import *


class OutputFormat(Enum):
    SAME_AS_INPUT = 0
    GIF = 1
    PNG = 2


class Sink(ABC):
    def __init__(self):
        self._inputs = []

    def _add_input(self, input):
        self._inputs.append(input)

    @property
    def input(self):
        return self._inputs
    
    def signal_input_ready(self, data : IoData):
        if not isinstance(data, IoData):
           raise TypeError("Input data must be of type IoData")

        for i in self._inputs:
            if not i.is_ready():
                return 

        if data.content == IoDataType.END_OF_STREAM:
            self.end_of_series()
        else:
            self.process()

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def end_of_series(self):
        pass


class Source(ABC):
    def __init__(self, name):
        self._outputs = []        
        self.__name = name

    @property
    def name(self):
        return self.__name      
    
    def _add_output(self, output):
        self._outputs.append(output)

    @property
    def output(self):
        return self._outputs
    
    @abstractmethod
    def start(self):    
        pass
