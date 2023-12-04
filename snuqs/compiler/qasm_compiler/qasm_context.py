from .qasm_utils import idlist_to_listid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional

QGATE_DICTIONARY = {
    # (name, num_qubits, num_params)
    # 1-qubit gates
    'ID':      (1, 0),
    'X':       (1, 0),
    'Y':       (1, 0),
    'Z':       (1, 0),
    'H':       (1, 0),
    'S':       (1, 0),
    'SDG':     (1, 0),
    'T':       (1, 0),
    'TDG':     (1, 0),
    'SX':      (1, 0),
    'SXDG':    (1, 0),
    'P':       (1, 1),
    'RX':      (1, 1),
    'RY':      (1, 1),
    'RZ':      (1, 1),
    'U0':      (1, 1),
    'U1':      (1, 1),
    'U2':      (1, 2),
    'U3':      (1, 3),
    'U':       (1, 3),
    # 2-qubit gates
    'CX':      (2, 0),
    'CZ':      (2, 0),
    'CY':      (2, 0),
    'SWAP':    (2, 0),
    'CH':      (2, 0),
    'CSX':     (2, 0),
    'CRX':     (2, 1),
    'CRY':     (2, 1),
    'CRZ':     (2, 1),
    'CU1':     (2, 1),
    'CP':      (2, 1),
    'RXX':     (2, 1),
    'RZZ':     (2, 1),
    'CU3':     (2, 3),
    'CU':      (2, 4),
    # 3-qubit gates
    'CCX':     (3, 0),
    'CSWAP':   (3, 0),
    'RCCX':    (3, 0),
    'RC3X':    (3, 0),
    # 4-qubit gates
    'C3X':     (4, 0),
    'C3SQRTX': (4, 0),
    # 5-qubit gates
    'C4X':     (5, 0),
}


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
