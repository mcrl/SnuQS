from enum import Enum

class AcceleratorType(Enum):
    CPU = 0
    CUDA = 1
    HYBRID = 2

class OffloadType(Enum):
    NONE = 0
    CPU = 1
    STORAGE = 2
