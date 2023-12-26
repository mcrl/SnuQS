from enum import Enum
from typing import Optional
from .qasm2_scope import Qasm2Scope


class Qasm2SymbolTable:
    class Type(Enum):
        QREG = 1
        CREG = 2
        OPAQUE = 3
        GATE = 4

    def __init__(self):
        self.sym_dict = {}
        self.scope_dict = {}

    def find(self, sym: str):
        if sym in self.sym_dict:
            return self.sym_dict[sym]
        return None

    def contains(self, sym: str, typ: Optional[Type] = None):
        if sym in self.sym_dict:
            if typ is None:
                return True
            else:
                return self.find(sym)[0] == typ
        else:
            return False

    def insert(self, sym: str, typ: Type, ctx: any):
        self.sym_dict[sym] = (typ, ctx)

    def items(self):
        return self.sym_dict.items()

    def attach_scope(self, sym: str, scope: Qasm2Scope, ctx: any):
        self.scope_dict[sym] = (scope, ctx)
