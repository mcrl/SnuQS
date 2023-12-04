from typing import List
from .qreg import Qreg
from .creg import Creg
from .qop import Qop


class Circuit:
    def __init__(self, name: str):
        self.name = name
        self.ops = []

    def append(self, op):
        self.ops.append(op)

    def __repr__(self):
        s = ""
        for op in self.ops:
            s += op.__repr__() + "\n"
        return s
