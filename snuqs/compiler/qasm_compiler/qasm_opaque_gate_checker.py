from .generated.QASMParser import QASMParser
from .qasm_stage import QasmStage
from .qasm_scope import QasmScope
from .qasm_symbol_table import QasmSymbolTable
from .qasm_utils import explist_to_listexp, idlist_to_listid, arglist_to_listarg

OPAQUE_GATES = [
    'id',
    'x',
    'y',
    'z',
    'h',
    's',
    'sdg',
    't',
    'tdg',
    'sx',
    'sxdg',
    'p',
    'rx',
    'ry',
    'rz',
    'u0',
    'u1',
    'u2',
    'u3',
    'u',
    'cx',
    'cy',
    'cz',
    'swap',
    'ch',
    'csx',
    'crx',
    'cry',
    'crz',
    'cp',
    'cu1',
    'rxx',
    'rzz',
    'cu3',
    'cu',
    'ccx',
    'cswap',
    'rccx',
    'rc3x',
    'c3x',
    'c3sqrtx',
    'c4x',
]


class QasmOpaqueGateChecker(QasmStage):
    def __init__(self, symtab: QasmSymbolTable):
        self.symtab = symtab

    # Enter a parse tree produced by QASMParser#opaqueStatement.
    def enterOpaqueStatement(self, ctx: QASMParser.OpaqueStatementContext):
        symbol = ctx.ID().getText()
        if symbol not in OPAQUE_GATES:
            raise LookupError(f"illegal opaque gate {symbol}")
