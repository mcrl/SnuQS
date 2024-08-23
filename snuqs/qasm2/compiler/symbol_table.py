from enum import Enum


class SymbolTable:
    class Type(Enum):
        QREG = 1
        CREG = 2
        OPAQUE = 3
        GATE = 4

    def __init__(self):
        self.table = {}

    def find(self, name):
        if name in self.table:
            return self.table[name]
        else:
            return None

    def insert(self, name, symbol_type: Type, ctx):
        self.table[name] = Symbol(name, symbol_type, ctx)


class Symbol:
    def __init__(self, name, typ: SymbolTable.Type, ctx):
        self.name = name
        self.type = typ
        self.ctx = ctx
