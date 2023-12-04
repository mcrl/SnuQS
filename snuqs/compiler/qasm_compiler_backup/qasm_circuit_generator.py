from .generated.QASMParser import QASMParser
from .qasm_stage import QasmStage
from .qasm_exception import QasmException
from .qasm_context import SymbolType, QasmContext
from .qasm_utils import arglist_to_listarg, goplist_to_listgop, idlist_to_listid, explist_to_listexp
from snuqs.circuit import Circuit, Qop, Cond, Qreg, Creg, Measure, Barrier, Reset
import math


class QasmCircuitGenerator(QasmStage):
    def __init__(self, ctx: QasmContext):
        self.ctx = ctx
        self.qubits = {}
        self.bits = {}
        self.nqubits = 0
        self.nbits = 0
#        for sym, (typ, ctx) in self.ctx.symbols.items():
#            if typ == SymbolType.QREG:
#                dim = int(ctx.NNINTEGER().getText())
#                self.qubits[f"{sym}"] = (self.nqubits, self.nqubits + dim)
#                self.nqubits += dim
#            elif typ == SymbolType.CREG:
#                dim = int(ctx.NNINTEGER().getText())
#                self.bits[f"{sym}"] = (self.nbits, self.nbits + dim)
#                self.nbits += dim
#            else:  # typ == SymbolType.QOP:
#                pass
#
#        self.circ = Circuit(
#            "circ", Qreg(self.nqubits), Creg(self.nbits))
#        self.ctx.circ = self.circ

        self.is_conditional = False
        self.cond_creg = None
        self.cond_base = 0
        self.cond_limit = 0
        self.cond_value = None

    def create_qop(self, typ, qubits, params=None, bits=[]):
        if self.is_conditional:
            self.is_conditional = False
            return Cond(Qop.Type.Cond,
                        self.cond_base,
                        self.cond_limit,
                        self.cond_value,
                        self.create_qop(typ, qubits, params, bits)
                        )
        elif typ == Qop.Type.Barrier:
            return Barrier(typ, qubits)
        elif typ == Qop.Type.Reset:
            return Reset(typ, qubits)
        elif typ == Qop.Type.Measure:
            return Measure(typ, qubits, bits)
        else:
            return Gate(typ, qubits, params=params)

    def exp_to_value(self, exp, par_map=None):
        if exp.REAL():
            return float(exp.REAL().getText())
        elif exp.NNINTEGER():
            return int(exp.NNINTEGER().getText())
        elif exp.ID():
            return par_map[exp.ID().getText()]
        elif exp.binop():
            op = exp.binop().getText()
            lhs = self.exp_to_value(exp.exp()[0], par_map=par_map)
            rhs = self.exp_to_value(exp.exp()[1], par_map=par_map)
            if op == '+':
                return lhs + rhs
            elif op == '-':
                return lhs - rhs
            elif op == '*':
                return lhs * rhs
            elif op == '/':
                return lhs / rhs
        elif exp.negop():
            return -self.exp_to_value(exp.exp()[0], par_map=par_map)
        elif exp.unaryop():
            op = exp.unaryop().getText()
            if op == 'sin':
                return math.sin(self.exp_to_value(exp.exp()[0], par_map=par_map))
            elif op == 'cos':
                return math.cos(self.exp_to_value(exp.exp()[0], par_map=par_map))
            elif op == 'tan':
                return math.tan(self.exp_to_value(exp.exp()[0], par_map=par_map))
            elif op == 'exp':
                return math.exp(self.exp_to_value(exp.exp()[0], par_map=par_map))
            elif op == 'ln':
                return math.ln(self.exp_to_value(exp.exp()[0], par_map=par_map))
            elif op == 'sqrt':
                return math.sqrt(self.exp_to_value(exp.exp()[0], par_map=par_map))
        elif exp.exp():
            return self.exp_to_value(exp.exp()[0], par_map=par_map)
        else:  # 'pi'
            return math.pi

    def arg_to_qubits(self, arg):
        idx = 0
        ident = arg.ID().getText()
        base, limit = self.qubits[f"{ident}"]
        if arg.NNINTEGER():
            base = base + int(arg.NNINTEGER().getText())
            limit = base + 1

        return list(range(base, limit))

    def arg_to_bits(self, arg):
        idx = 0
        ident = arg.ID().getText()
        base, limit = self.bits[f"{ident}"]
        if arg.NNINTEGER():
            base = int(arg.NNINTEGER().getText())
            limit = base + 1

        return list(range(base, limit))

    def inline_UGate(self, ctx, var_map, par_map):
        params = [self.exp_to_value(exp, par_map=par_map)
                  for exp in explist_to_listexp(ctx.explist())]
        q = var_map[ctx.ID().getText()]
        self.circ.push_op(self.create_qop(Qop.Type.UGate, [q], params=params))

    def inline_CXGate(self, ctx, var_map, par_map):
        cident = ctx.ID()[0].getText()
        tident = ctx.ID()[1].getText()
        c = var_map[cident]
        t = var_map[tident]
        self.circ.push_op(self.create_qop(Qop.Type.CXGate, [c, t]))

    def inline_CustomGate(self, ctx, var_map, par_map):
        values = [self.exp_to_value(exp, par_map=par_map)
                  for exp in explist_to_listexp(ctx.explist())]

        gate = self.ctx.get_gate(ctx.ID().getText())
        qubits = idlist_to_listid(ctx.idlist())

        vs = idlist_to_listid(gate.gatedecl().idlist())
        ps = []
        if gate.gatedecl().paramlist():
            ps = idlist_to_listid(gate.gatedecl().paramlist().idlist())

        new_var_map = {v.getText(): var_map[q.getText()]
                       for v, q in zip(vs, qubits)}
        new_par_map = par_map.copy()
        for p, v in zip(ps, values):
            new_par_map[p.getText()] = v
        self.inline_custom(gate, new_var_map, new_par_map)

    def inline_Barrier(self, ctx, var_map, par_map):
        for i in idlist_to_listid(ctx.idlist()):
            q = var_map[i.getText()]
            self.circ.push_op(self.create_qop(Qop.Type.Barrier, [q]))

    def inline_ReservedGate(self, ctx, var_map, par_map):
        params = [self.exp_to_value(exp, par_map=par_map)
                  for exp in explist_to_listexp(ctx.explist())]
        q = var_map[ctx.ID().getText()]
        self.circ.push_op(self.create_qop(Qop.Type.UGate, [q], params=params))

    def inline_custom(self, gate, var_map, par_map):
        for gop in goplist_to_listgop(gate.goplist()):
            if gop.gopUGate():
                self.inline_UGate(gop.gopUGate(), var_map, par_map)
            elif gop.gopCXGate():
                self.inline_CXGate(gop.gopCXGate(), var_map, par_map)
            elif gop.gopReservedGate():
                self.inline_ReservedGate(
                    gop.gopReservedGate(), var_map, par_map)
            elif gop.gopCustomGate():
                self.inline_CustomGate(gop.gopCustomGate(), var_map, par_map)
            else:  # gop.gopBarrier():
                self.inline_Barrier(gop.gopBarrier(), var_map, par_map)

    # Enter a parse tree produced by QASMParser#opaqueStatement.
    def enterOpaqueStatement(self, ctx: QASMParser.OpaqueStatementContext):
        raise QasmException("Opaque statement not supported.")

    # Enter a parse tree produced by QASMParser#qopUGate.
    def enterQopUGate(self, ctx: QASMParser.QopUGateContext):
        params = [self.exp_to_value(exp)
                  for exp in explist_to_listexp(ctx.explist())]

        qlist = self.arg_to_qubits(ctx.argument())
        for q in qlist:
            self.circ.push_op(self.create_qop(
                Qop.Type.UGate, [q], params=params))

    # Enter a parse tree produced by QASMParser#qopCXGate.
    def enterQopCXGate(self, ctx: QASMParser.QopCXGateContext):
        clist = self.arg_to_qubits(ctx.argument()[0])
        tlist = self.arg_to_qubits(ctx.argument()[1])

        mx = max(len(clist), len(tlist))
        clist = clist if len(clist) == mx else clist * mx
        tlist = tlist if len(tlist) == mx else tlist * mx

        for c, t in zip(clist, tlist):
            self.circ.push_op(self.create_qop(Qop.Type.CXGate, [c, t]))

    def addGate(self, qop_type, arglist, explist=None):
        if type(arglist) is not list:
            arglist = [arglist]

        targets = []
        for arg in arglist:
            targets.append(self.arg_to_qubits(arg))

        params = None
        if explist:
            params = [self.exp_to_value(exp)
                      for exp in explist_to_listexp(explist)]

        for qs in zip(*targets):
            self.circ.push_op(self.create_qop(qop_type, qs, params=params))

    # Enter a parse tree produced by QASMParser#qopCustomGate.
    def enterQopCustomGate(self, ctx: QASMParser.QopCustomGateContext):
        list_of_qlist = []
        for arg in arglist_to_listarg(ctx.arglist()):
            list_of_qlist.append(self.arg_to_qubits(arg))

        mx = max([len(lst) for lst in list_of_qlist])
        for i, qlist in enumerate(list_of_qlist):
            list_of_qlist[i] = qlist if len(qlist) == mx else qlist * mx

        values = [self.exp_to_value(exp)
                  for exp in explist_to_listexp(ctx.explist())]
        gate = self.ctx.get_gate(ctx.ID().getText())
        vs = idlist_to_listid(gate.gatedecl().idlist())
        ps = []
        if gate.gatedecl().paramlist():
            ps = idlist_to_listid(gate.gatedecl().paramlist().idlist())

        for qubits in zip(*list_of_qlist):
            args = ",".join([str(q) for q in qubits])
            var_map = {v.getText(): q for v, q in zip(vs, qubits)}
            par_map = {p.getText(): v for p, v in zip(ps, values)}
            self.inline_custom(gate, var_map, par_map)

    # Enter a parse tree produced by QASMParser#qopMeasure.
    def enterQopMeasure(self, ctx: QASMParser.QopMeasureContext):
        qargs = self.arg_to_qubits(ctx.argument()[0])
        cargs = self.arg_to_bits(ctx.argument()[1])
        mx = max(len(qargs), len(cargs))
        qargs = qargs if len(qargs) == mx else qargs * mx
        cargs = cargs if len(cargs) == mx else cargs * mx

        for q, c in zip(qargs, cargs):
            self.circ.push_op(self.create_qop(Qop.Type.Measure, [q], bits=[c]))

    # Enter a parse tree produced by QASMParser#qopReset.
    def enterQopReset(self, ctx: QASMParser.QopResetContext):
        for q in self.arg_to_qubits(ctx.argument()):
            self.circ.push_op(self.create_qop(Qop.Type.Reset, [q]))

    # Enter a parse tree produced by QASMParser#ifStatement.
    def enterIfStatement(self, ctx: QASMParser.IfStatementContext):
        ident = ctx.ID().getText()
        nn = int(ctx.NNINTEGER().getText())
        self.is_conditional = True
        self.cond_base, self.cond_limit = self.bits[ident]
        self.cond_value = nn

    def exitIfStatement(self, ctx: QASMParser.IfStatementContext):
        self.is_conditional = False

    # Enter a parse tree produced by QASMParser#barrierStatement.
    def enterBarrierStatement(self, ctx: QASMParser.BarrierStatementContext):
        for arg in arglist_to_listarg(ctx.arglist()):
            for q in self.arg_to_qubits(arg):
                self.circ.push_op(self.create_qop(Qop.Type.Barrier, [q]))
