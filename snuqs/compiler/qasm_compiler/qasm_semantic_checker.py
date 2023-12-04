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

        if major != '2' or minor != '0':
            raise NotImplementedError(
                f"OPENQASM version mismatch: expected 2.0, but {major}.{minor} given.")

    def enterQregDeclStatement(self, ctx: QASMParser.QregDeclStatementContext):
        symbol = ctx.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")
        self.symtab.insert(symbol, QasmSymbolTable.Type.QREG, ctx)

        dim = int(ctx.NNINTEGER().getText())
        if dim <= 0:
            raise ValueError(
                f"dimension must be positive but {dim}.")

    # Enter a parse tree produced by QASMParser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx: QASMParser.CregDeclStatementContext):
        symbol = ctx.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")
        self.symtab.insert(symbol, QasmSymbolTable.Type.CREG, ctx)

        dim = int(ctx.NNINTEGER().getText())
        if dim <= 0:
            raise ValueError(
                f"dimension must be positive but {dim}.")

    # Enter a parse tree produced by QASMParser#opaqueDeclStatement.
    def enterOpaqueDeclStatement(self, ctx: QASMParser.OpaqueDeclStatementContext):
        symbol = ctx.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")
        self.symtab.insert(symbol, QasmSymbolTable.Type.OPAQUE, ctx)

        self.scope = QasmScope()
        for _id in ctx.idlist().ID():
            symbol = _id.getText()

            if self.scope.contains(symbol):
                raise LookupError(f"'{symbol}' already defined in the scope.")

            self.scope.insert(symbol, QasmScope.Type.TARGET, ctx)

        if ctx.paramlist():
            for _id in ctx.paramlist().ID():
                symbol = _id.getText()

                if self.scope.contains(symbol):
                    raise LookupError(
                        f"'{symbol}' already defined in the scope.")

                self.scope.insert(symbol, QasmScope.Type.PARAM, ctx)

    # Exit a parse tree produced by QASMParser#opaqueDeclStatement.
    def exitOpaqueDeclStatement(self, ctx: QASMParser.OpaqueDeclStatementContext):
        symbol = ctx.ID().getText()
        self.symtab.attach_scope(symbol, self.scope, ctx)

    # Enter a parse tree produced by QASMParser#gatedeclStatement.
    def enterGatedeclStatement(self, ctx: QASMParser.GatedeclStatementContext):
        symbol = ctx.ID().getText()
        if self.symtab.contains(symbol):
            raise LookupError(f"'{symbol}' already defined.")
        self.symtab.insert(symbol, QasmSymbolTable.Type.GATE, ctx)

        self.scope = QasmScope()
        for _id in ctx.idlist().ID():
            symbol = _id.getText()

            if self.scope.contains(symbol):
                raise LookupError(f"'{symbol}' already defined in the scope.")

            self.scope.insert(symbol, QasmScope.Type.TARGET, ctx)

        if ctx.paramlist():
            for _id in ctx.paramlist().ID():
                symbol = _id.getText()

                if self.scope.contains(symbol):
                    raise LookupError(
                        f"'{symbol}' already defined in the scope.")

                self.scope.insert(symbol, QasmScope.Type.PARAM, ctx)

    # Exit a parse tree produced by QASMParser#gatedeclStatement.
    def exitGatedeclStatement(self, ctx: QASMParser.GatedeclStatementContext):
        symbol = ctx.ID().getText()
        self.symtab.attach_scope(symbol, self.scope, ctx)

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
        for _id in ctx.ID():
            symbol = _id.getText()
            if not self.scope.contains(symbol, QasmScope.Type.TARGET):
                raise LookupError(f"target '{symbol}' has not been defined.")

        ids = ctx.ID()
        if ids[0].getText() == ids[1].getText():
            raise ValueError(f"duplicated target '{symbol}' is not allowed")

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
        if not self.symtab.contains(symbol, QasmSymbolTable.Type.OPAQUE) and not self.symtab.contains(symbol, QasmSymbolTable.Type.GATE):
            raise LookupError(f"gate '{symbol}' has not been defined.")

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

    # Enter a parse tree produced by QASMParser#qopUGate.
    def enterQopUGate(self, ctx: QASMParser.QopUGateContext):
        num_exprs = len(ctx.explist().exp())
        if num_exprs != 3:
            raise ValueError(f"U needs 3 expressions but {num_exprs} given.")

        for exp in ctx.explist().exp():
            if exp.ID():
                symbol = exp.ID().getText()
                if not self.symtab.contains(symbol, QasmSymbolTable.Type.CREG):
                    raise LookupError(
                        f"creg '{symbol}' has not been defined.")

    # Enter a parse tree produced by QASMParser#qopCXGate.
    def enterQopCXGate(self, ctx: QASMParser.QopCXGateContext):
        pass

    # Enter a parse tree produced by QASMParser#qopMeasure.
    def enterQopMeasure(self, ctx: QASMParser.QopMeasureContext):
        pass

    # Enter a parse tree produced by QASMParser#qopReset.
    def enterQopReset(self, ctx: QASMParser.QopResetContext):
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

            for exp in ctx.explist().exp():
                if exp.ID():
                    symbol = exp.ID().getText()
                    if not self.scope.contains(symbol, QasmScope.Type.PARAM):
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
