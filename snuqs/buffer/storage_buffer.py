from .buffer import Buffer
from typing import Union
import snuqs._C
import yaml


class StorageBuffer(Buffer):
    def __init__(self, size: int, file_name):
        with open(file_name) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.impl = snuqs._C.StorageBuffer(size, config['devices'])

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, slice):
            raise NotImplementedError("slice is not supported")
        else:
            return self.impl.__getitem__(key)

    def __setitem__(self, key: Union[int, slice], val: complex):
        if isinstance(key, slice):
            raise NotImplementedError("slice is not supported")
        else:
            return self.impl.__setitem__(key, val)
