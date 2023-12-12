from snuqs._C import Qarg, Carg
#from .reg import Reg
#from abc import ABCMeta
#
#
#class Arg(metaclass=ABCMeta):
#    def __init__(self, reg: Reg, index: int = -1):
#        self.reg = reg
#        self.index = index
#        self.dim = 1 if index != -1 else reg.dim
#
#    def __repr__(self):
#        return f"{self.reg.name}[{self.index}]"
#
#
#class Qarg(Arg):
#    pass
#
#
#class Carg(Arg):
#    pass
