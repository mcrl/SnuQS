from dataclasses import dataclass
from .qreg import Qreg
from .creg import Creg
from .qop import Qop

@dataclass
class QuantumCircuit:
    name: str
    ops: list[Qop]
    qreg: Qreg
    creg: Creg


    def __repr__(self):
        s = ""
        for op in self.ops:
            s += op.__repr__() + "\n"
        return s


    def prepend_op(self, op):
        self.ops = [op] + self.ops

    
    def push_op(self, op):
        self.ops.append(op)
