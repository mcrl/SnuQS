from snuqs._C import Circuit
#from .reg import Qreg, Creg
#from .qop import Qop
#
#
#class Circuit:
#    def __init__(self, name: str):
#        self.name = name
#        self.qops = []
#        self.qregs = []
#        self.cregs = []
#
#    def append_qreg(self, qreg: Qreg):
#        self.qregs.append(qreg)
#
#    def append_creg(self, creg: Creg):
#        self.cregs.append(creg)
#
#    def append(self, op: Qop):
#        self.qops.append(op)
#
#    def __repr__(self):
#        s = f"Circuit<'{self.name}'>\n"
#
#        s += "qregs:\n"
#        for qreg in self.qregs:
#            s += f"    {qreg}\n"
#
#        s += "cregs:\n"
#        for creg in self.cregs:
#            s += f"    {creg}\n"
#
#        s += "qops:\n"
#        for op in self.qops:
#            s += f"    {op.__repr__()}\n"
#        return s
