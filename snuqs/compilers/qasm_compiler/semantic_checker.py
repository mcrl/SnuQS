from .generated.QASMListener import QASMListener
from .generated.QASMParser import QASMParser
from .qasm_exception import QASMSemanticException
from .qasm_context import SymbolType, ParamType, QASMContext
from .qasm_utils import *

import sys

class SemanticChecker(QASMListener):
    def __init__(self, ctx):
        self.ctx = ctx

    def check_symbol_duplication(self, ident, ctx=None):
        sym = ident.getText()
        if self.ctx.contains_symbol(sym):
            raise QASMSemanticException(f"Duplicated symbol definition ({sym}).")

    def check_param_duplication(self, ident, ctx=None):
        sym = ident.getText()
        if self.ctx.contains_param(sym):
            raise QASMSemanticException(f"Duplicated parameter definition ({sym}).")

    def check_gate_defined(self, ident, ctx=None):
        sym = ident.getText()
        if not self.ctx.contains_gate(sym):
            raise QASMSemanticException(f"Gate undefined ({sym}).")

    def check_qreg_defined(self, ident, ctx=None):
        sym = ident.getText()
        if not self.ctx.contains_qreg(sym):
            raise QASMSemanticException(f"Qreg undefined ({sym}).")

    def check_creg_defined(self, ident, ctx=None):
        sym = ident.getText()
        if not self.ctx.contains_creg(sym):
            raise QASMSemanticException(f"Creg undefined ({sym}).")

    def check_qubit_defined(self, ident, ctx=None):
        sym = ident.getText()
        if not self.ctx.contains_qubitparam(sym):
            raise QASMSemanticException(f"Qubit parameter undefined ({sym}).")

    def check_arg_index(self, arg, ctx=None):
        sym = arg.ID().getText()
        dim = int(self.ctx.get_qreg(sym).NNINTEGER().getText())
        if arg.NNINTEGER():
            idx = int(arg.NNINTEGER().getText())
            if idx >= dim:
                raise QASMSemanticException(f"Illegal indexing ({sym}[{idx}] >= {sym}[{dim}]).")

    def check_num_args(self, ident, listarg, ctx=None):
        if self.ctx.get_num_args(ident.getText()) != len(listarg):
            raise QASMSemanticException(f"Illegal invocation of gate ({ident.getText()}).")

    def check_integer_positivity(self, nninteger, ctx=None):
        i = int(nninteger.getText())
        if i <= 0:
            raise QASMSemanticException(f"Illegal use of 0 or negative integers ({i}).")


    def check_listarg_dims(self, listarg):
        dims = []
        for arg in listarg:
            dim = self.ctx.get_reg_dim(arg.ID().getText())
            dims.append(1 if arg.NNINTEGER() else dim)

        mx = max(dims)
        for dim in dims:
            if dim != 1 and dim != mx:
                raise QASMSemanticException("Illegal argument dimensions ({dim} != {mx}).")

    def check_valid_exp(self, exp, qop=None, ctx=None):
        if not exp:
            return True

        if exp.ID():
            if qop:
                raise QASMSemanticException(f"Invalid use of identifer in parameters ({exp.ID().getText()}).")
                return False
                #return self.ctx.contains_creg(exp.ID().getText())
            else:
                self.ctx.contains_hyperparam(exp.ID().getText())
        elif isinstance(exp.exp(), list):
            for exp in exp.exp():
                self.check_valid_exp(exp)
        else:
            self.check_valid_exp(exp.exp())

    def is_valid_exp(self, exp, qop=False):
        if not exp:
            return True

        ID = exp.ID()
        if ID:
            if qop:
                return self.ctx.contains_creg(ID.getText())
            else:
                return self.ctx.contains_hyperparam(ID.getText())
        elif isinstance(exp.exp(), list):
            check = True
            for exp in exp.exp():
                check = check and self.is_valid_exp(exp)
            return check
        else:
            return self.is_valid_exp(exp.exp())

    # Enter a parse tree produced by QASMParser#mainprogram.
    def enterMainprogram(self, ctx:QASMParser.MainprogramContext):
        self.ctx.create_scope()

    # Exit a parse tree produced by QASMParser#mainprogram.
    def exitMainprogram(self, ctx:QASMParser.MainprogramContext):
        self.ctx.destroy_scope()

    # Enter a parse tree produced by QASMParser#version.
    def enterVersion(self, ctx:QASMParser.VersionContext):
        major, minor = ctx.REAL().getText().split('.')

        if major != '2' or minor != '0':
            raise QASMSemanticException(f"OPENQASM version mismatch (Expected 2.0).")

    # Enter a parse tree produced by QASMParser#qregDeclStatement.
    def enterQregDeclStatement(self, ctx:QASMParser.QregDeclStatementContext):
        self.check_symbol_duplication(ctx.ID())
        self.check_integer_positivity(ctx.NNINTEGER())
        self.ctx.insert_symbol(ctx.ID().getText(), SymbolType.QREG, ctx)

    # Enter a parse tree produced by QASMParser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx:QASMParser.CregDeclStatementContext):
        self.check_symbol_duplication(ctx.ID())
        self.check_integer_positivity(ctx.NNINTEGER())
        self.ctx.insert_symbol(ctx.ID().getText(), SymbolType.CREG, ctx)

    # Enter a parse tree produced by QASMParser#gatedeclStatement.
    def enterGatedeclStatement(self, ctx:QASMParser.GatedeclStatementContext):
        self.check_symbol_duplication(ctx.gatedecl().ID())

        self.ctx.create_scope()

        for p in idlist_to_listid(ctx.gatedecl().idlist()):
            self.check_param_duplication(p)
            self.ctx.insert_param(p.getText(), ParamType.QUBITPARAM, p)

        if ctx.gatedecl().paramlist():
            for p in idlist_to_listid(ctx.gatedecl().paramlist().idlist()):
                self.check_param_duplication(p)
                self.ctx.insert_param(p.getText(), ParamType.HYPERPARAM, p)


    # Exit a parse tree produced by QASMParser#gatedeclStatement.
    def exitGatedeclStatement(self, ctx:QASMParser.GatedeclStatementContext):
        self.ctx.destroy_scope()
        # insert symbol after the semantic check to prevent recursive call.
        self.ctx.insert_symbol(ctx.gatedecl().ID().getText(), SymbolType.GATE, ctx)

    # Enter a parse tree produced by QASMParser#gopUGate.
    def enterGopUGate(self, ctx:QASMParser.GopUGateContext):
        self.check_qubit_defined(ctx.ID())

        for exp in explist_to_listexp(ctx.explist()):
            self.check_valid_exp(exp)

    # Enter a parse tree produced by QASMParser#gopCXGate.
    def enterGopCXGate(self, ctx:QASMParser.GopCXGateContext):
        self.check_qubit_defined(ctx.ID()[0])
        self.check_qubit_defined(ctx.ID()[1])

    # Enter a parse tree produced by QASMParser#gopCustomGate.
    def enterGopCustomGate(self, ctx:QASMParser.GopCustomGateContext):
        self.check_gate_defined(ctx.ID())

        for exp in explist_to_listexp(ctx.explist()):
            self.check_valid_exp(exp)

        for q in idlist_to_listid(ctx.idlist()):
            self.check_qubit_defined(q)

    # Enter a parse tree produced by QASMParser#gopBarrier.
    def enterGopBarrier(self, ctx:QASMParser.GopBarrierContext):
        for q in idlist_to_listid(ctx.idlist()):
            self.check_qubit_defined(q)

    # Enter a parse tree produced by QASMParser#opaqueStatement.
    def enterOpaqueStatement(self, ctx:QASMParser.OpaqueStatementContext):
        raise QASMSemanticException("Not supported.")

    # Enter a parse tree produced by QASMParser#qopUGate.
    def enterQopUGate(self, ctx:QASMParser.QopUGateContext):
        listarg = [ctx.argument()]
        for arg in listarg:
            self.check_qreg_defined(arg.ID())
            self.check_arg_index(arg)

        self.check_listarg_dims(listarg)

        for exp in explist_to_listexp(ctx.explist()):
            self.check_valid_exp(exp, qop=True)


    # Enter a parse tree produced by QASMParser#qopCXGate.
    def enterQopCXGate(self, ctx:QASMParser.QopCXGateContext):
        listarg = ctx.argument()
        for arg in listarg:
            self.check_qreg_defined(arg.ID())
            self.check_arg_index(arg)

        self.check_listarg_dims(listarg)

    # Enter a parse tree produced by QASMParser#qopCustomGate.
    def enterQopCustomGate(self, ctx:QASMParser.QopCustomGateContext):

        self.check_gate_defined(ctx.ID())

        listarg = arglist_to_listarg(ctx.arglist())
        for arg in listarg:
            self.check_qreg_defined(arg.ID())
            self.check_arg_index(arg)

        self.check_num_args(ctx.ID(), listarg)

        self.check_listarg_dims(listarg)

        for exp in explist_to_listexp(ctx.explist()):
            self.check_valid_exp(exp, qop=True)

    # Enter a parse tree produced by QASMParser#qopMeasure.
    def enterQopMeasure(self, ctx:QASMParser.QopMeasureContext):
        listarg = ctx.argument()
        for i, arg in enumerate(listarg):
            if i == 0: self.check_qreg_defined(arg.ID())
            else: self.check_creg_defined(arg.ID())
            self.check_arg_index(arg)

        self.check_listarg_dims(listarg)


    # Enter a parse tree produced by QASMParser#qopReset.
    def enterQopReset(self, ctx:QASMParser.QopResetContext):
        listarg = [ctx.argument()]
        for arg in listarg:
            self.check_qreg_defined(arg.ID())
            self.check_arg_index(arg)

    # Enter a parse tree produced by QASMParser#ifStatement.
    def enterIfStatement(self, ctx:QASMParser.IfStatementContext):
        self.check_creg_defined(ctx.ID())

    # Enter a parse tree produced by QASMParser#barrierStatement.
    def enterBarrierStatement(self, ctx:QASMParser.BarrierStatementContext):
        for arg in arglist_to_listarg(ctx.arglist()):
            self.check_qreg_defined(arg.ID())
            self.check_arg_index(arg)
