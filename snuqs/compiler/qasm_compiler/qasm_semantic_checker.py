from .generated.QASMParser import QASMParser
from .qasm_stage import QasmStage
from .qasm_scope import QasmScope
from .qasm_symbol_table import QasmSymbolTable


class QasmSemanticChecker(QasmStage):
    def __init__(self, symtab: QasmSymbolTable):
        self.symtab = symtab

    # Enter a parse tree produced by QASMParser#version.
    def enterVersion(self, ctx: QASMParser.VersionContext):
        major, minor = ctx.REAL().getText().split('.')
        major, minor = int(major), int(minor)

        if major != 2 or minor != 0:
            raise NotImplementedError(
                f"OPENQASM version must be 2.0: but {major}.{minor} given.")

    # Enter a parse tree produced by QASMParser#regDeclStatement.
    def enterRegDeclStatement(self, ctx: QASMParser.RegDeclStatementContext):
        decl = ctx.getChild(0)
        symbol = decl.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")

        dim = int(decl.NNINTEGER().getText())
        if dim <= 0:
            raise ValueError(f"dimension {dim} <= 0.")

    def enterQregDeclStatement(self, ctx: QASMParser.QregDeclStatementContext):
        self.symtab.insert(ctx.ID().getText(), QasmSymbolTable.Type.QREG, ctx)

    # Enter a parse tree produced by QASMParser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx: QASMParser.CregDeclStatementContext):
        self.symtab.insert(ctx.ID().getText(), QasmSymbolTable.Type.CREG, ctx)

    # Enter a parse tree produced by QASMParser#gateDeclStatement.
    def enterGateDeclStatement(self, ctx: QASMParser.GateDeclStatementContext):
        decl = ctx.getChild(0)

        symbol = decl.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")

        self.scope = QasmScope()
        for _id in decl.idlist().ID():
            symbol = _id.getText()
            if self.scope.contains(symbol):
                raise LookupError(f"'{symbol}' is duplicated.")
            self.scope.insert(symbol, QasmScope.Type.TARGET, decl)

        if decl.paramlist():
            for _id in decl.paramlist().ID():
                symbol = _id.getText()
                if self.scope.contains(symbol):
                    raise LookupError(f"'{symbol}' is duplicated.")
                self.scope.insert(symbol, QasmScope.Type.PARAM, decl)

    # Exit a parse tree produced by QASMParser#gateDeclStatement.
    def exitGateDeclStatement(self, ctx: QASMParser.GateDeclStatementContext):
        decl = ctx.getChild(0)
        self.symtab.attach_scope(decl.ID().getText(), self.scope, decl)

    # Enter a parse tree produced by QASMParser#opaqueStatement.
    def enterOpaqueStatement(self, ctx: QASMParser.OpaqueStatementContext):
        self.symtab.insert(ctx.ID().getText(),
                           QasmSymbolTable.Type.OPAQUE, ctx)

    # Enter a parse tree produced by QASMParser#gateStatement.
    def enterGateStatement(self, ctx: QASMParser.GateStatementContext):
        self.symtab.insert(ctx.ID().getText(),
                           QasmSymbolTable.Type.GATE, ctx)

    # Enter a parse tree produced by QASMParser#gopUGate.
    def enterGopUGate(self, ctx: QASMParser.GopUGateContext):
        symbol = ctx.ID().getText()
        if not self.scope.contains(symbol, QasmScope.Type.TARGET):
            raise LookupError(f"target '{symbol}' has not been defined.")

        num_exprs = len(ctx.explist().exp())
        if num_exprs != 3:
            raise ValueError(f"U needs 3 expressions but {num_exprs} given.")

        for exp in ctx.explist().exp():
            if exp.ID():
                symbol = exp.ID().getText()
                if not self.scope.contains(symbol, QasmScope.Type.PARAM):
                    raise LookupError(
                        f"parameter '{symbol}' has not been defined.")

    # Enter a parse tree produced by QASMParser#gopCXGate.
    def enterGopCXGate(self, ctx: QASMParser.GopCXGateContext):
        _id0, _id1 = ctx.ID()

        symbol0 = _id0.getText()
        symbol1 = _id1.getText()

        if not self.scope.contains(symbol0, QasmScope.Type.TARGET):
            raise LookupError(f"target '{symbol0}' has not been defined.")

        if not self.scope.contains(symbol1, QasmScope.Type.TARGET):
            raise LookupError(f"target '{symbol1}' has not been defined.")

        if symbol0 == symbol1:
            raise ValueError(f"duplicated target '{symbol0}' is not allowed")

    # Enter a parse tree produced by QASMParser#gopBarrier.
    def enterGopBarrier(self, ctx: QASMParser.GopBarrierContext):
        for _id in ctx.idlist().ID():
            symbol = _id.getText()
            if not self.scope.contains(symbol, QasmScope.Type.TARGET):
                raise LookupError(f"target '{symbol}' has not been defined.")

        symbols = [_id.getText() for _id in ctx.idlist().ID()]
        symbols = sorted(symbols)
        for i in range(len(symbols)-1):
            if symbols[i] == symbols[i+1]:
                raise ValueError(
                    f"duplicated target '{symbols[i]}' is not allowed")

    # Enter a parse tree produced by QASMParser#gopCustomGate.
    def enterGopCustomGate(self, ctx: QASMParser.GopCustomGateContext):
        symbol = ctx.ID().getText()
        is_opaque = self.symtab.contains(symbol, QasmSymbolTable.Type.OPAQUE)
        is_gate = self.symtab.contains(symbol, QasmSymbolTable.Type.GATE)
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
            if not self.scope.contains(symbol, QasmScope.Type.TARGET):
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
                    if not self.scope.contains(symbol, QasmScope.Type.PARAM):
                        raise LookupError(
                            f"parameter '{symbol}' has not been defined.")

    # Exit a parse tree produced by QASMParser#qopUGate.
    def exitQopUGate(self, ctx: QASMParser.QopUGateContext):
        num_exprs = len(ctx.explist().exp())
        if num_exprs != 3:
            raise ValueError(f"U needs 3 expressions but {num_exprs} given.")

        for exp in ctx.explist().exp():
            if exp.ID():
                symbol = exp.ID().getText()
                if not self.symtab.contains(symbol, QasmSymbolTable.Type.CREG):
                    raise LookupError(
                        f"creg '{symbol}' has not been defined.")

    # Exit a parse tree produced by QASMParser#qopCXGate.
    def exitQopCXGate(self, ctx: QASMParser.QopCXGateContext):
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

    # Exit a parse tree produced by QASMParser#qopMeasure.
    def exitQopMeasure(self, ctx: QASMParser.QopMeasureContext):
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

    # Enter a parse tree produced by QASMParser#qopReset.
    def exitQopReset(self, ctx: QASMParser.QopResetContext):
        pass

    # Enter a parse tree produced by QASMParser#qopCustomGate.
    def enterQopCustomGate(self, ctx: QASMParser.QopCustomGateContext):
        symbol = ctx.ID().getText()
        if not self.symtab.contains(symbol, QasmSymbolTable.Type.OPAQUE) and not self.symtab.contains(symbol, QasmSymbolTable.Type.GATE):
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
                    if not self.symtab.contains(symbol, QasmSymbolTable.Type.CREG):
                        raise LookupError(
                            f"parameter '{symbol}' has not been defined.")

    # Enter a parse tree produced by QASMParser#ifStatement.
    def enterIfStatement(self, ctx: QASMParser.IfStatementContext):
        symbol = ctx.ID().getText()
        if not self.symtab.contains(symbol, QasmSymbolTable.Type.CREG):
            raise LookupError(f"creg '{symbol}' has not been defined.")

    # Enter a parse tree produced by QASMParser#barrierStatement.
    def enterBarrierStatement(self, ctx: QASMParser.BarrierStatementContext):
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

    # Enter a parse tree produced by QASMParser#qarg.
    def enterQarg(self, ctx: QASMParser.QargContext):
        symbol = ctx.ID().getText()

        if not self.symtab.contains(symbol, QasmSymbolTable.Type.QREG):
            raise LookupError(f"'qreg {symbol}' has not been defined.")

        if ctx.NNINTEGER():
            _, decl = self.symtab.find(symbol)
            dim = int(decl.NNINTEGER().getText())
            idx = int(ctx.NNINTEGER().getText())
            if (idx >= dim):
                raise IndexError(
                    f"index {idx} must be less than dimension {dim}.")

    # Enter a parse tree produced by QASMParser#carg.
    def enterCarg(self, ctx: QASMParser.CargContext):
        symbol = ctx.ID().getText()

        if not self.symtab.contains(symbol, QasmSymbolTable.Type.CREG):
            raise LookupError(f"'creg {symbol}' has not been defined.")

        if ctx.NNINTEGER():
            _, decl = self.symtab.find(symbol)
            dim = int(decl.NNINTEGER().getText())
            idx = int(ctx.NNINTEGER().getText())
            if (idx >= dim):
                raise IndexError(
                    f"index {idx} must be less than dimension {dim}.")
