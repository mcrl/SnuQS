from enum import Enum
from typing import Optional


class Qasm2Scope:
    class Type(Enum):
        TARGET = 5
        PARAM = 6

    def __init__(self):
        self.scope = {}

    def contains(self, sym: str, typ: Optional[Type] = None):
        if sym in self.scope:
            if typ is None:
                return True
            else:
                return self.scope[sym][0] == typ
        else:
            return False

    def insert(self, sym: str, typ: Type, ctx: any):
        self.scope[sym] = (typ, ctx)
