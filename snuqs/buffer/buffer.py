from abc import ABCMeta, abstractmethod
from typing import Union
import snuqs._C


class Buffer(metaclass=ABCMeta):
    @ abstractmethod
    def __getitem__(self, key: Union[int, slice]):
        pass

    @ abstractmethod
    def __setitem__(self, key: Union[int, slice], val: complex):
        pass

    @ abstractmethod
    def __len__(self):
        pass
