from .qasm_utils import idlist_to_listid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional

class ParamType(Enum):
    QUBITPARAM = 1
    HYPERPARAM = 2


@dataclass
class QasmScope:
    params: Dict[str, Tuple[ParamType, Any]] = field(
        default_factory=lambda: {})


class SymbolType(Enum):
    QREG = 1
    CREG = 2
    QOP = 3


@dataclass
class QasmSymbolTable3:
    symbols: Dict[str, Tuple[SymbolType, Any]] = field(
        default_factory=lambda: {})


class QasmContext:
    def __init__(self):
        self.scopes = []
        pass

    def enter_scope(self):
        self.scopes.append(QasmScope())

    def exit_scope(self):
        self.scopes.pop()
