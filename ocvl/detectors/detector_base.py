import cv2

from abc import ABC, abstractmethod


class DetectorBase(ABC):
    def __init__(self, name):
        self._name = name
        self._pattern = None

    def load(self, file):
        self._pattern = cv2.imread(file)
        self._width, self._height, _ = self._pattern.shape
        self.after_load(file)

    @property
    def name(self):
        return self._name

    @property
    def image(self):
        return self._pattern

    @abstractmethod
    def after_load(self, file):
        pass

    @abstractmethod
    def search(self, file):
        pass