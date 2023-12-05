from .generated.QASMParser import QASMParser
from .qasm_stage import QasmStage
from .qasm_symbol_table import QasmSymbolTable
from typing import Dict

from snuqs.circuit import Circuit, Qgate
from snuqs.circuit import Qreg, Creg
from snuqs.circuit import U, CX, Measure, Reset, Barrier, Cond, Custom
from snuqs.circuit import Parameter, BinOp, NegOp, UnaryOp, Parenthesis, Identifier, Constant, Pi


class QasmCircuitGenerator(QasmStage):
    def __init__(self, circuit: Circuit, symtab: QasmSymbolTable):
        self.symtab = symtab
        self.circuit = circuit

    def binOp(self, op: str,
              expr0: QASMParser.ExpContext,
              expr1: QASMParser.ExpContext):
        if op == '+':
            return BinOp(BinOp.Type.ADD,
                         self.expAsParameter(expr0),
                         self.expAsParameter(expr1))
        elif op == '-':
            return BinOp(BinOp.Type.SUB,
                         self.expAsParameter(expr0),
                         self.expAsParameter(expr1))
        elif op == '*':
            return BinOp(BinOp.Type.MUL,
                         self.expAsParameter(expr0),
                         self.expAsParameter(expr1))
        elif op == '/':
            return BinOp(BinOp.Type.DIV,
                         self.expAsParameter(expr0),
                         self.expAsParameter(expr1))

    def unaryOp(self, op: str, expr: QASMParser.ExpContext):
        if op == 'sin':
            return UnaryOp(UnaryOp.Type.SIN, self.expAsParameter(expr))
        elif op == 'cos':
            return UnaryOp(UnaryOp.Type.COS, self.expAsParameter(expr))
        elif op == 'tan':
            return UnaryOp(UnaryOp.Type.TAN, self.expAsParameter(expr))
        elif op == 'exp':
            return UnaryOp(UnaryOp.Type.EXP, self.expAsParameter(expr))
        elif op == 'ln':
            return UnaryOp(UnaryOp.Type.LN, self.expAsParameter(expr))
        elif op == 'sqrt':
            return UnaryOp(UnaryOp.Type.SQRT, self.expAsParameter(expr))

    def expAsParameter(self, exp: QASMParser.ExpContext):
        if exp.binop():
            return self.binOp(exp.binop().getText(),
                              exp.exp()[0],
                              exp.exp()[1])
        elif exp.negop():
            return NegOp(self.expAsParameter(exp.exp()[0]))
        elif exp.unaryop():
            return self.unaryOp(exp.unaryop().getText(), exp.exp()[0])
        elif exp.exp():
            return Parenthesis(self.expAsParameter(exp.exp()[0]))
        elif exp.ID():
            return Identifier(self.creg_map[exp.ID().getText()])
        elif exp.REAL():
            return Constant(float(exp.REAL().getText()))
        elif exp.NNINTEGER():
            return Constant(float(exp.NNINTEGER().getText()))
        else:
            return Pi()

    def qargAsQreg(self, ctx: QASMParser.QargContext):
        qreg = self.qreg_map[ctx.ID().getText()]
        if ctx.NNINTEGER():
            dim = int(ctx.NNINTEGER().getText())
            qreg = qreg[dim]
        return qreg

    def cargAsCreg(self, ctx: QASMParser.CargContext):
        creg = self.creg_map[ctx.ID().getText()]
        if ctx.NNINTEGER():
            dim = int(ctx.NNINTEGER().getText())
            creg = creg[dim]
        return creg

    def createGop(self,
                  ctx: QASMParser.GopContext,
                  qubit_map: Dict[str, Qreg],
                  param_map: Dict[str, Parameter]):
        if ctx.gopUGate():
            qreg = qubit_map[ctx.gopUGate().ID().getText()]
            self.creg_map, tmp = param_map, self.creg_map
            params = [self.expAsParameter(exp)
                      for exp in ctx.gopUGate().explist().exp()]
            self.creg_map = tmp
            return U([qreg], params=params)
        elif ctx.gopCXGate():
            qregs = [qubit_map[_id.getText()] for _id in ctx.gopCXGate().ID()]
            return CX(qregs)
        elif ctx.gopBarrier():
            qregs = [qubit_map[_id.getText()]
                     for _id in ctx.gopBarrier().idlist().ID()]
            return CX(qregs)
        elif ctx.gopCustomGate():
            qregs = [qubit_map[_id.getText()]
                     for _id in ctx.gopCustomGate().idlist().ID()]
            self.creg_map, tmp = param_map, self.creg_map
            params = []
            if ctx.gopCustomGate().explist():
                params = [self.expAsParameter(exp)
                          for exp in ctx.gopCustomGate().explist().exp()]
            self.creg_map = tmp

            symbol = ctx.gopCustomGate().ID().getText()
            if symbol in self.opaque_map:
                for subcls in Qgate.__subclasses__():
                    if subcls.__name__ == symbol.capitalize():
                        return subcls(qregs, params=params)
            else:
                symbol = ctx.gopCustomGate().ID().getText()
                decl = self.gate_map[symbol]
                qregs = [
                    qubit_map[_id.getText()] for _id in ctx.gopCustomGate().idlist().ID()
                ]
                self.creg_map, tmp = param_map, self.creg_map
                params = []
                if ctx.gopCustomGate().explist():
                    params = [self.expAsParameter(exp)
                              for exp in ctx.gopCustomGate().explist().exp()]
                self.creg_map = tmp
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

                return Custom(symbol, gops, qregs, params=params)

    def createCustomGate(self, ctx: QASMParser.QopCustomGateContext):
        symbol = ctx.ID().getText()
        qreg = [self.qargAsQreg(qarg) for qarg in ctx.arglist().qarg()]
        params = []
        if ctx.explist():
            params = [self.expAsParameter(exp) for exp in ctx.explist().exp()]

        if symbol in self.opaque_map:
            decl = self.opaque_map[symbol]
            for subcls in Qgate.__subclasses__():
                if subcls.__name__ == symbol.upper():
                    return subcls(qreg, params=params)
        else:
            decl = self.gate_map[symbol]
            qubit_map = {
                _id.getText(): qreg for _id, qreg in zip(decl.idlist().ID(), qreg)
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

            return Custom(symbol, gops, qreg, params=params)

    def createQop(self, ctx: QASMParser.QopStatementContext):
        if ctx.qopUGate():
            qreg = self.qargAsQreg(ctx.qopUGate().qarg())
            params = [self.expAsParameter(exp)
                      for exp in ctx.qopUGate().explist().exp()]
            return U([qreg], params=params)
        elif ctx.qopCXGate():
            qreg0 = self.qargAsQreg(ctx.qopCXGate().qarg()[0])
            qreg1 = self.qargAsQreg(ctx.qopCXGate().qarg()[1])
            return CX([qreg0, qreg1])
        elif ctx.qopMeasure():
            qreg = self.qargAsQreg(ctx.qopMeasure().qarg())
            creg = self.cargAsCreg(ctx.qopMeasure().carg())
            return Measure([qreg], [creg])
        elif ctx.qopReset():
            qreg = self.qargAsQreg(ctx.qopReset().qarg())
            return Reset([qreg])
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
                self.qreg_map[symbol] = Qreg(symbol, dim)
            elif typ == QasmSymbolTable.Type.CREG:
                dim = int(ctx.NNINTEGER().getText())
                self.creg_map[symbol] = Creg(symbol, dim)
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
        qreglist = [self.qargAsQreg(qarg) for qarg in ctx.arglist().qarg()]
        self.circuit.append(Barrier(qreglist))
