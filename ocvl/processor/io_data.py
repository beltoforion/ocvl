from enum import Enum


class IoDataType(Enum):
    IMAGE = 0,
    END_OF_STREAM = 1


class IoData:
    def __init__(self, image = None):
        self.__image = image
        if image is None:
            self.__type = IoDataType.END_OF_STREAM
        else:
            self.__type = IoDataType.IMAGE

    @property
    def content(self):
        return self.__type

    @property
    def image(self):
        return self.__image
    