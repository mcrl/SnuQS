from .buffer import Buffer
from typing import Union
import snuqs._C


class StorageBuffer(Buffer):
    def __init__(self, size: int):
        self.impl = snuqs._C.StorageBuffer(size)

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, slice):
            raise NotImplementedError("slice is not supported")
        else:
            return self.impl.__getitem__(key)

    def __setitem__(self, key: Union[int, slice], val: float):
        if isinstance(key, slice):
            raise NotImplementedError("slice is not supported")
        else:
            return self.impl.__setitem__(key, val)
