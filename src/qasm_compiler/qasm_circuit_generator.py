from .generated.QASMParser import QASMParser
from .qasm_stage import QasmStage
from .qasm_symbol_table import QasmSymbolTable
from typing import Dict

from snuqs.circuit import Circuit, Qgate
from snuqs.circuit import Qreg, Creg, Qarg, Carg
from snuqs.circuit import U, CX, Measure, Reset, Barrier, Cond, Custom
from snuqs.circuit import Parameter, Identifier, BinOp, BinOpType, NegOp
from snuqs.circuit import UnaryOp, UnaryOpType, Parenthesis, Constant, Pi


class QasmCircuitGenerator(QasmStage):
    def __init__(self, circuit: Circuit, symtab: QasmSymbolTable):
        self.symtab = symtab
        self.circuit = circuit

    def binOp(self, op: str,
              expr0: QASMParser.ExpContext,
              expr1: QASMParser.ExpContext,
              param_map: Dict[str, Parameter]
              ):
        if op == '+':
            return BinOp(BinOpType.ADD,
                         self.expAsParameter(expr0, param_map),
                         self.expAsParameter(expr1, param_map))
        elif op == '-':
            return BinOp(BinOpType.SUB,
                         self.expAsParameter(expr0, param_map),
                         self.expAsParameter(expr1, param_map))
        elif op == '*':
            return BinOp(BinOpType.MUL,
                         self.expAsParameter(expr0, param_map),
                         self.expAsParameter(expr1, param_map))
        elif op == '/':
            return BinOp(BinOpType.DIV,
                         self.expAsParameter(expr0, param_map),
                         self.expAsParameter(expr1, param_map))

    def unaryOp(self,
                op: str,
                expr: QASMParser.ExpContext,
                param_map: Dict[str, Parameter]):
        if op == 'sin':
            return UnaryOp(UnaryOpType.SIN,
                           self.expAsParameter(expr, param_map))
        elif op == 'cos':
            return UnaryOp(UnaryOpType.COS,
                           self.expAsParameter(expr, param_map))
        elif op == 'tan':
            return UnaryOp(UnaryOpType.TAN,
                           self.expAsParameter(expr, param_map))
        elif op == 'exp':
            return UnaryOp(UnaryOpType.EXP,
                           self.expAsParameter(expr, param_map))
        elif op == 'ln':
            return UnaryOp(UnaryOpType.LN,
                           self.expAsParameter(expr, param_map))
        elif op == 'sqrt':
            return UnaryOp(UnaryOpType.SQRT,
                           self.expAsParameter(expr, param_map))

    def expAsParameter(self, exp: QASMParser.ExpContext,
                       param_map: Dict[str, Parameter]):
        if exp.binop():
            return self.binOp(exp.binop().getText(),
                              exp.exp()[0],
                              exp.exp()[1],
                              param_map)
        elif exp.negop():
            return NegOp(self.expAsParameter(exp.exp()[0], param_map))
        elif exp.unaryop():
            return self.unaryOp(exp.unaryop().getText(), exp.exp()[0], param_map)
        elif exp.exp():
            return Parenthesis(self.expAsParameter(exp.exp()[0], param_map))
        elif exp.ID():
            symbol = exp.ID().getText()
            if symbol in param_map:
                return param_map[symbol]
            else:
                return Identifier(self.creg_map[symbol])
        elif exp.complex_():
            real = 0
            imag = 0
            numbers = [float(v.getText()) for v in exp.complex_().REAL()]
            real = numbers[0]
            if len(numbers) == 2:
                imag = numbers[1]
            op = exp.complex_().addsub().getText()
            if op == '+':
                val = real + imag * 1j
            elif op == '-':
                val = real - imag * 1j
            return Constant(val)
        elif exp.REAL():
            return Constant(float(exp.REAL().getText()))
        elif exp.NNINTEGER():
            return Constant(float(exp.NNINTEGER().getText()))
        else:
            return Pi()

    def createQarg(self, ctx: QASMParser.QargContext):
        qreg = self.qreg_map[ctx.ID().getText()]
        if ctx.NNINTEGER():
            index = int(ctx.NNINTEGER().getText())
            qarg = Qarg(qreg, index)
        else:
            qarg = Qarg(qreg)
        return qarg

    def createCarg(self, ctx: QASMParser.CargContext):
        creg = self.creg_map[ctx.ID().getText()]
        if ctx.NNINTEGER():
            index = int(ctx.NNINTEGER().getText())
            carg = Carg(creg, index)
        else:
            carg = Carg(creg)
        return carg

    def createGop(self,
                  ctx: QASMParser.GopContext,
                  qubit_map: Dict[str, Qreg],
                  param_map: Dict[str, Parameter]
                  ):
        if ctx.gopUGate():
            qreg = qubit_map[ctx.gopUGate().ID().getText()]
            params = [self.expAsParameter(exp, param_map)
                      for exp in ctx.gopUGate().explist().exp()]
            return U([qreg], params)
        elif ctx.gopCXGate():
            qregs = [qubit_map[_id.getText()] for _id in ctx.gopCXGate().ID()]
            return CX(qregs)
        elif ctx.gopBarrier():
            qregs = [qubit_map[_id.getText()]
                     for _id in ctx.gopBarrier().idlist().ID()]
            return Barrier(qregs)
        elif ctx.gopCustomGate():
            qregs = [qubit_map[_id.getText()]
                     for _id in ctx.gopCustomGate().idlist().ID()]
            params = []
            if ctx.gopCustomGate().explist():
                params = [self.expAsParameter(exp, param_map)
                          for exp in ctx.gopCustomGate().explist().exp()]

            symbol = ctx.gopCustomGate().ID().getText()
            if symbol in self.opaque_map:
                for subcls in Qgate.__subclasses__():
                    if subcls.__name__ == symbol.upper():
                        return subcls(qregs, params)
            else:
                symbol = ctx.gopCustomGate().ID().getText()
                decl = self.gate_map[symbol]
                qregs = [
                    qubit_map[_id.getText()] for _id in ctx.gopCustomGate().idlist().ID()
                ]
                params = []
                if ctx.gopCustomGate().explist():
                    params = [self.expAsParameter(exp, param_map)
                              for exp in ctx.gopCustomGate().explist().exp()]
                _qubit_map = {
                    _id.getText(): qreg for _id, qreg in zip(decl.idlist().ID(), qregs)
                }
                _param_map = {}
                if decl.paramlist():
                    _param_map = {
                        _id.getText(): param for _id, param in zip(decl.paramlist().ID(), params)
                    }

                gops = []
                for gop in decl.goplist().gop():
                    gops.append(self.createGop(gop, _qubit_map, _param_map))

                return Custom(symbol, gops, qregs, params)
        elif ctx.gopReset():
            qreg = qubit_map[ctx.gopReset().ID().getText()]
            return Reset([qreg])

    def createCustomGate(self, ctx: QASMParser.QopCustomGateContext):
        symbol = ctx.ID().getText()
        qargs = [self.createQarg(qarg) for qarg in ctx.arglist().qarg()]
        params = []
        if ctx.explist():
            params = [self.expAsParameter(exp, {})
                      for exp in ctx.explist().exp()]

        if symbol in self.opaque_map:
            decl = self.opaque_map[symbol]
            for subcls in Qgate.__subclasses__():
                if subcls.__name__ == symbol.upper():
                    return subcls(qargs, params)
        else:
            decl = self.gate_map[symbol]
            qubit_map = {
                _id.getText(): qarg for _id, qarg in zip(decl.idlist().ID(), qargs)
            }
            param_map = {}
            if decl.paramlist():
                param_map = {
                    _id.getText(): param for _id, param in zip(decl.paramlist().ID(), params)
                }

            gops = []
            if decl.goplist():
                for gop in decl.goplist().gop():
                    gops.append(self.createGop(gop, qubit_map, param_map))

            return Custom(symbol, gops, qargs, params)

    def createQop(self, ctx: QASMParser.QopStatementContext):
        if ctx.qopUGate():
            qarg = self.createQarg(ctx.qopUGate().qarg())
            params = [self.expAsParameter(exp, {})
                      for exp in ctx.qopUGate().explist().exp()]
            return U([qarg], params)
        elif ctx.qopCXGate():
            qarg0 = self.createQarg(ctx.qopCXGate().qarg()[0])
            qarg1 = self.createQarg(ctx.qopCXGate().qarg()[1])
            return CX([qarg0, qarg1])
        elif ctx.qopMeasure():
            qarg = self.createQarg(ctx.qopMeasure().qarg())
            carg = self.createCarg(ctx.qopMeasure().carg())
            return Measure([qarg], [carg])
        elif ctx.qopReset():
            qarg = self.createQarg(ctx.qopReset().qarg())
            return Reset([qarg])
        elif ctx.qopCustomGate():
            return self.createCustomGate(ctx.qopCustomGate())

    # Enter a parse tree produced by QASMParser#mainprogram.
    def enterMainprogram(self, ctx: QASMParser.MainprogramContext):
        self.qreg_map = {}
        self.creg_map = {}
        self.gate_map = {}
        self.opaque_map = {}
        for symbol, (typ, ctx) in self.symtab.items():
            if typ == QasmSymbolTable.Type.QREG:
                dim = int(ctx.NNINTEGER().getText())
                qreg = Qreg(symbol, dim)
                self.qreg_map[symbol] = qreg
                self.circuit.append_qreg(qreg)
            elif typ == QasmSymbolTable.Type.CREG:
                dim = int(ctx.NNINTEGER().getText())
                creg = Creg(symbol, dim)
                self.creg_map[symbol] = creg
                self.circuit.append_creg(creg)
            elif typ == QasmSymbolTable.Type.GATE:
                self.gate_map[symbol] = ctx
            elif typ == QasmSymbolTable.Type.OPAQUE:
                self.opaque_map[symbol] = ctx
            else:
                raise ValueError(
                    f"unknown symbol {symbol} in the symbol table.")

    # Exit a parse tree produced by QASMParser#mainprogram.
    def exitMainprogram(self, ctx: QASMParser.MainprogramContext):
        pass

    # Enter a parse tree produced by QASMParser#qopStatement.
    def enterQopStatement(self, ctx: QASMParser.QopStatementContext):
        qop = self.createQop(ctx)
        self.circuit.append(qop)

    # Enter a parse tree produced by QASMParser#ifStatement.
    def enterIfStatement(self, ctx: QASMParser.IfStatementContext):
        creg = self.creg_map[ctx.ID().getText()]
        value = int(ctx.NNINTEGER().getText())

        qop = self.createQop(ctx.qopStatement())
        self.circuit.append(Cond(qop, creg, value))

        # Disable visiting children
        self.getChildren = ctx.getChildren
        ctx.getChildren = lambda: []
        return None

    # Exit a parse tree produced by QASMParser#ifStatement.
    def exitIfStatement(self, ctx: QASMParser.IfStatementContext):
        ctx.getChildren = self.getChildren

    # Enter a parse tree produced by QASMParser#barrierStatement.
    def enterBarrierStatement(self, ctx: QASMParser.BarrierStatementContext):
        qargs = [self.createQarg(qarg) for qarg in ctx.arglist().qarg()]
        self.circuit.append(Barrier(qargs))
