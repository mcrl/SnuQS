from .generated.Qasm2Parser import Qasm2Parser
from .qasm2_stage import Qasm2Stage
from .qasm2_symbol_table import Qasm2SymbolTable
from .qasm2_walker import Qasm2Walker

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

    'initialize',
]


class Qasm2OpaqueGateChecker(Qasm2Stage):
    def __init__(self, symtab: Qasm2SymbolTable):
        super().__init__()
        self.symtab = symtab

    # Enter a parse tree produced by Qasm2Parser#opaqueStatement.
    def enterOpaqueStatement(self, ctx: Qasm2Parser.OpaqueStatementContext):
        symbol = ctx.ID().getText()
        if symbol not in OPAQUE_GATES:
            raise LookupError(f"illegal opaque gate {symbol}")
