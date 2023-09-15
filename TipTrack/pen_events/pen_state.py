# Enum to define State of a Point
from enum import Enum


class PenState(Enum):
    NEW = -1  # A new event where it is not yet sure what state it will have
    CLICK = 0
    DRAG = 1
    DOUBLE_CLICK = 2
    HOVER = 3
