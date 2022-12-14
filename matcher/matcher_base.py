from abc import ABC, abstractmethod
import numpy as np


class MatcherBase(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name
        
    @abstractmethod
    def match(self, d1, d2):        
        pass