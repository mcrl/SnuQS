from .generated.Qasm2Parser import Qasm2Parser
from .qasm2_stage import Qasm2Stage
from .qasm2_scope import Qasm2Scope
from .qasm2_symbol_table import Qasm2SymbolTable


class Qasm2SemanticChecker(Qasm2Stage):
    def __init__(self, symtab: Qasm2SymbolTable):
        super().__init__()
        self.symtab = symtab
        self.major = -1
        self.minor = -1

    # Enter a parse tree produced by Qasm2Parser#version.
    def enterVersion(self, ctx: Qasm2Parser.VersionContext):
        major, minor = ctx.REAL().getText().split('.')
        major, minor = int(major), int(minor)

        if major != 2 or minor != 0:
            raise NotImplementedError(
                f"OPENQASM version must be 2.0: but {major}.{minor} given.")

        self.major = major
        self.minor = minor

    # Enter a parse tree produced by Qasm2Parser#regDeclStatement.
    def enterRegDeclStatement(self, ctx: Qasm2Parser.RegDeclStatementContext):
        decl = ctx.getChild(0)
        symbol = decl.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")

        dim = int(decl.NNINTEGER().getText())
        if dim <= 0:
            raise ValueError(f"dimension {dim} <= 0.")

    def enterQregDeclStatement(self, ctx: Qasm2Parser.QregDeclStatementContext):
        self.symtab.insert(ctx.ID().getText(), Qasm2SymbolTable.Type.QREG, ctx)

    # Enter a parse tree produced by Qasm2Parser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx: Qasm2Parser.CregDeclStatementContext):
        self.symtab.insert(ctx.ID().getText(), Qasm2SymbolTable.Type.CREG, ctx)

    # Enter a parse tree produced by Qasm2Parser#gateDeclStatement.
    def enterGateDeclStatement(self, ctx: Qasm2Parser.GateDeclStatementContext):
        decl = ctx.getChild(0)

        symbol = decl.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")

        self.scope = Qasm2Scope()
        for _id in decl.idlist().ID():
            symbol = _id.getText()
            if self.scope.contains(symbol):
                raise LookupError(f"'{symbol}' is duplicated.")
            self.scope.insert(symbol, Qasm2Scope.Type.TARGET, decl)

        if decl.paramlist():
            for _id in decl.paramlist().ID():
                symbol = _id.getText()
                if self.scope.contains(symbol):
                    raise LookupError(f"'{symbol}' is duplicated.")
                self.scope.insert(symbol, Qasm2Scope.Type.PARAM, decl)

    # Exit a parse tree produced by Qasm2Parser#gateDeclStatement.
    def exitGateDeclStatement(self, ctx: Qasm2Parser.GateDeclStatementContext):
        decl = ctx.getChild(0)
        self.symtab.attach_scope(decl.ID().getText(), self.scope, decl)

    # Enter a parse tree produced by Qasm2Parser#opaqueStatement.
    def enterOpaqueStatement(self, ctx: Qasm2Parser.OpaqueStatementContext):
        self.symtab.insert(ctx.ID().getText(),
                           Qasm2SymbolTable.Type.OPAQUE, ctx)

    # Enter a parse tree produced by Qasm2Parser#gateStatement.
    def enterGateStatement(self, ctx: Qasm2Parser.GateStatementContext):
        self.symtab.insert(ctx.ID().getText(),
                           Qasm2SymbolTable.Type.GATE, ctx)

    # Enter a parse tree produced by Qasm2Parser#gopUGate.
    def enterGopUGate(self, ctx: Qasm2Parser.GopUGateContext):
        symbol = ctx.ID().getText()
        if not self.scope.contains(symbol, Qasm2Scope.Type.TARGET):
            raise LookupError(f"target '{symbol}' has not been defined.")

        num_exprs = len(ctx.explist().exp())
        if num_exprs != 3:
            raise ValueError(f"U needs 3 expressions but {num_exprs} given.")

        for exp in ctx.explist().exp():
            if exp.ID():
                symbol = exp.ID().getText()
                if not self.scope.contains(symbol, Qasm2Scope.Type.PARAM):
                    raise LookupError(
                        f"parameter '{symbol}' has not been defined.")

    # Enter a parse tree produced by Qasm2Parser#gopCXGate.
    def enterGopCXGate(self, ctx: Qasm2Parser.GopCXGateContext):
        _id0, _id1 = ctx.ID()

        symbol0 = _id0.getText()
        symbol1 = _id1.getText()

        if not self.scope.contains(symbol0, Qasm2Scope.Type.TARGET):
            raise LookupError(f"target '{symbol0}' has not been defined.")

        if not self.scope.contains(symbol1, Qasm2Scope.Type.TARGET):
            raise LookupError(f"target '{symbol1}' has not been defined.")

        if symbol0 == symbol1:
            raise ValueError(f"duplicated target '{symbol0}' is not allowed")

    # Enter a parse tree produced by Qasm2Parser#gopBarrier.
    def enterGopBarrier(self, ctx: Qasm2Parser.GopBarrierContext):
        for _id in ctx.idlist().ID():
            symbol = _id.getText()
            if not self.scope.contains(symbol, Qasm2Scope.Type.TARGET):
                raise LookupError(f"target '{symbol}' has not been defined.")

        symbols = [_id.getText() for _id in ctx.idlist().ID()]
        symbols = sorted(symbols)
        for i in range(len(symbols)-1):
            if symbols[i] == symbols[i+1]:
                raise ValueError(
                    f"duplicated target '{symbols[i]}' is not allowed")

    # Enter a parse tree produced by Qasm2Parser#gopCustomGate.
    def enterGopCustomGate(self, ctx: Qasm2Parser.GopCustomGateContext):
        symbol = ctx.ID().getText()
        is_opaque = self.symtab.contains(symbol, Qasm2SymbolTable.Type.OPAQUE)
        is_gate = self.symtab.contains(symbol, Qasm2SymbolTable.Type.GATE)
        if not (is_opaque or is_gate):
            raise LookupError(
                f"gate (or opaque) '{symbol}' has not been defined.")

        num_ids_given = len(ctx.idlist().ID())
        num_ids_required = len(self.symtab.find(symbol)[1].idlist().ID())
        if num_ids_given != num_ids_required:
            raise ValueError(
                f"number of arguments does not match: {num_ids_required} required but {num_ids_given} given")

        for _id in ctx.idlist().ID():
            symbol = _id.getText()
            if not self.scope.contains(symbol, Qasm2Scope.Type.TARGET):
                raise LookupError(f"target '{symbol}' has not been defined.")

        symbols = [_id.getText() for _id in ctx.idlist().ID()]
        symbols = sorted(symbols)
        for i in range(len(symbols)-1):
            if symbols[i] == symbols[i+1]:
                raise ValueError(
                    f"duplicated target '{symbols[i]}' is not allowed")

        if ctx.explist():
            symbol = ctx.ID().getText()
            num_exprs_given = len(ctx.explist().exp())
            decl = self.symtab.find(symbol)[1]
            if decl.paramlist():
                num_exprs_required = len(decl.paramlist().ID())
            if num_exprs_given != num_exprs_required:
                raise ValueError(
                    f"number of expressions does not match: {num_exprs_given} required but {num_exprs_given} given")

            for exp in ctx.explist().exp():
                if exp.ID():
                    symbol = exp.ID().getText()
                    if not self.scope.contains(symbol, Qasm2Scope.Type.PARAM):
                        raise LookupError(
                            f"parameter '{symbol}' has not been defined.")

    # Enter a parse tree produced by Qasm2Parser#gopReset.
    def enterGopReset(self, ctx: Qasm2Parser.GopResetContext):
        symbol = ctx.ID().getText()
        if not self.scope.contains(symbol, Qasm2Scope.Type.TARGET):
            raise LookupError(f"target '{symbol}' has not been defined.")

    # Exit a parse tree produced by Qasm2Parser#qopUGate.
    def exitQopUGate(self, ctx: Qasm2Parser.QopUGateContext):
        num_exprs = len(ctx.explist().exp())
        if num_exprs != 3:
            raise ValueError(f"U needs 3 expressions but {num_exprs} given.")

        for exp in ctx.explist().exp():
            if exp.ID():
                symbol = exp.ID().getText()
                if not self.symtab.contains(symbol, Qasm2SymbolTable.Type.CREG):
                    raise LookupError(
                        f"creg '{symbol}' has not been defined.")

    # Exit a parse tree produced by Qasm2Parser#qopCXGate.
    def exitQopCXGate(self, ctx: Qasm2Parser.QopCXGateContext):
        symbols = [(qarg.ID().getText(), qarg.getText())
                   for qarg in ctx.qarg()]
        symbols = sorted(symbols)
        for i in range(len(symbols)-1):
            if symbols[i][0] == symbols[i][1]:
                if symbols[i][0] == symbols[i+1][0]:
                    raise ValueError(
                        f"duplicated target '{symbols[i][1]}' and '{symbols[i+1][1]}'is not allowed")
            elif symbols[i][1] == symbols[i+1][1]:
                raise ValueError(
                    f"duplicated target '{symbols[i][1]}' is not allowed")

    # Exit a parse tree produced by Qasm2Parser#qopMeasure.
    def exitQopMeasure(self, ctx: Qasm2Parser.QopMeasureContext):
        if ctx.qarg().NNINTEGER():
            qdim = 1
        else:
            qdim = int(self.symtab.find(
                ctx.qarg().ID().getText())[1].NNINTEGER().getText())
        if ctx.carg().NNINTEGER():
            cdim = 1
        else:
            cdim = int(self.symtab.find(
                ctx.carg().ID().getText())[1].NNINTEGER().getText())

        if qdim != cdim:
            raise IndexError("measure dimension mismatch: {qdim} != {cdim}")

    # Enter a parse tree produced by Qasm2Parser#qopReset.
    def exitQopReset(self, ctx: Qasm2Parser.QopResetContext):
        pass

    # Enter a parse tree produced by Qasm2Parser#qopCustomGate.
    def enterQopCustomGate(self, ctx: Qasm2Parser.QopCustomGateContext):
        symbol = ctx.ID().getText()
        if not self.symtab.contains(symbol, Qasm2SymbolTable.Type.OPAQUE) and not self.symtab.contains(symbol, Qasm2SymbolTable.Type.GATE):
            raise LookupError(f"gate '{symbol}' has not been defined.")

        num_ids_given = len(ctx.arglist().qarg())
        num_ids_required = len(self.symtab.find(symbol)[1].idlist().ID())
        if num_ids_given != num_ids_required:
            raise ValueError(
                f"number of arguments does not match: {num_ids_required} required but {num_ids_given} given")

        symbols = []
        for qarg in ctx.arglist().qarg():
            symbol = qarg.ID().getText()
            if qarg.NNINTEGER():
                symbol = f"{symbol}[{int(qarg.NNINTEGER().getText())}]"
            symbols.append(symbol)
        symbols = sorted(symbols)
        for i in range(len(symbols)-1):
            if symbols[i] == symbols[i+1]:
                raise ValueError(
                    f"duplicated target '{symbols[i]}' is not allowed")

        if ctx.explist():
            symbol = ctx.ID().getText()
            num_exprs_given = len(ctx.explist().exp())
            decl = self.symtab.find(symbol)[1]
            if decl.paramlist():
                num_exprs_required = len(decl.paramlist().ID())
            if num_exprs_given != num_exprs_required:
                raise ValueError(
                    f"number of expressions does not match: {num_exprs_given} required but {num_exprs_given} given")

            exprs = ctx.explist().exp()
            while len(exprs) > 0:
                exp = exprs.pop()
                if exp.exp():
                    if type(exp.exp()) is list:
                        exprs = exprs + exp.exp()
                    else:
                        exprs.append(exp.exp())
                elif exp.ID():
                    symbol = exp.ID().getText()
                    if not self.symtab.contains(symbol, Qasm2SymbolTable.Type.CREG):
                        raise LookupError(
                            f"parameter '{symbol}' has not been defined.")

    # Enter a parse tree produced by Qasm2Parser#ifStatement.
    def enterIfStatement(self, ctx: Qasm2Parser.IfStatementContext):
        symbol = ctx.ID().getText()
        if not self.symtab.contains(symbol, Qasm2SymbolTable.Type.CREG):
            raise LookupError(f"creg '{symbol}' has not been defined.")

    # Enter a parse tree produced by Qasm2Parser#barrierStatement.
    def enterBarrierStatement(self, ctx: Qasm2Parser.BarrierStatementContext):
        symbols = [(qarg.ID().getText(), qarg.getText())
                   for qarg in ctx.arglist().qarg()]
        symbols = sorted(symbols)
        for i in range(len(symbols)-1):
            if symbols[i][0] == symbols[i][1]:
                if symbols[i][0] == symbols[i+1][0]:
                    raise ValueError(
                        f"duplicated target '{symbols[i][1]}' and '{symbols[i+1][1]}'is not allowed")
            elif symbols[i][1] == symbols[i+1][1]:
                raise ValueError(
                    f"duplicated target '{symbols[i][1]}' is not allowed")

    # Enter a parse tree produced by Qasm2Parser#qarg.
    def enterQarg(self, ctx: Qasm2Parser.QargContext):
        symbol = ctx.ID().getText()

        if not self.symtab.contains(symbol, Qasm2SymbolTable.Type.QREG):
            raise LookupError(f"'qreg {symbol}' has not been defined.")

        if ctx.NNINTEGER():
            _, decl = self.symtab.find(symbol)
            dim = int(decl.NNINTEGER().getText())
            idx = int(ctx.NNINTEGER().getText())
            if (idx >= dim):
                raise IndexError(
                    f"index {idx} must be less than dimension {dim}.")

    # Enter a parse tree produced by Qasm2Parser#carg.
    def enterCarg(self, ctx: Qasm2Parser.CargContext):
        symbol = ctx.ID().getText()

        if not self.symtab.contains(symbol, Qasm2SymbolTable.Type.CREG):
            raise LookupError(f"'creg {symbol}' has not been defined.")

        if ctx.NNINTEGER():
            _, decl = self.symtab.find(symbol)
            dim = int(decl.NNINTEGER().getText())
            idx = int(ctx.NNINTEGER().getText())
            if (idx >= dim):
                raise IndexError(
                    f"index {idx} must be less than dimension {dim}.")
