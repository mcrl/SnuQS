from .qasm_utils import *
from enum import Enum

class SymbolType(Enum):
    QREG = 1
    CREG = 2
    GATE = 3

class ParamType(Enum):
    QUBITPARAM = 1
    HYPERPARAM = 2

class QASMContext:
    def __init__(self):
        self.symbols = {}
        self.params = []

    def insert_symbol(self, sym, typ, ctx):
        self.symbols[sym] = (typ, ctx)

    def contains_symbol(self, sym):
        return sym in self.symbols

    def contains_qreg(self, sym):
        return sym in self.symbols and self.symbols[sym][0] == SymbolType.QREG

    def contains_creg(self, sym):
        return sym in self.symbols and self.symbols[sym][0] == SymbolType.CREG

    def contains_gate(self, sym):
        return sym in self.symbols and self.symbols[sym][0] == SymbolType.GATE

    def get_qreg(self, sym):
        return self.symbols[sym][1]

    def get_creg(self, sym):
        return self.symbols[sym][1]

    def get_gate(self, sym):
        return self.symbols[sym][1]

    def get_reg_dim(self, sym):
        return int(self.symbols[sym][1].NNINTEGER().getText())

    def get_num_args(self, sym):
        idlist =  self.symbols[sym][1].gatedecl().idlist()
        return len(idlist_to_listid(idlist))

    def create_scope(self):
        self.params.append({})

    def destroy_scope(self):
        self.params.pop()

    def insert_param(self, sym, typ, ctx):
        self.params[-1][sym] = (typ, ctx)

    def contains_param(self, sym):
        return sym in self.params[-1]

    def contains_qubitparam(self, sym):
        return sym in self.params[-1] and self.params[-1][sym][0] == ParamType.QUBITPARAM

    def contains_hyperparam(self, sym):
        return sym in self.params[-1] and self.params[-1][sym][0] == ParamType.HYPERPARAM

