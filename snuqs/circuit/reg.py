from snuqs._C import Reg, Qreg, Creg
#from abc import ABCMeta
#
#
#class Reg(metaclass=ABCMeta):
#    def __init__(self, name: str, dim: int):
#        if dim == 0:
#            raise ValueError("Qreg of dim 0 is not allowed")
#
#        self.name = name
#        self.dim = dim
#
#    def __repr__(self):
#        return f"{self.name}"
#
#
#class Qreg(Reg):
#    pass
#
#
#class Creg(Reg):
#    pass
