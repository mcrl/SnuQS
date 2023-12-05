from .buffer import Buffer
from typing import Union
import snuqs._C


class MemoryBuffer(Buffer):
    def __init__(self, size: int):
        self.impl = snuqs._C.MemoryBuffer(size)

    def __getitem__(self, key: int):
        if isinstance(key, slice):
            raise NotImplementedError("slice is not supported")
        else:
            return self.impl.__getitem__(key)

    def __setitem__(self, key: Union[int, slice], val: float):
        if isinstance(key, slice):
            raise NotImplementedError("slice is not supported")
        else:
            return self.impl.__setitem__(key, val)
