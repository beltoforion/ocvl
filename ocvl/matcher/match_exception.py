
from enum import Enum

class MatchError(Enum):
    AFFINE_MATCH_FAILED = 1
    MEDIAN_DISTANCE_EXCEEDED = 2
    UNKNOWN = 3


class MatchException(Exception):
    def __init__(self, err : MatchError):
        self.__errc = err

    @property
    def error(self):
        return self.__errc

    @error.setter
    def error(self, value):
        self.__errc = value


