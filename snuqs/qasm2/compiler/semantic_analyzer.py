from .generated import qasm2Listener, qasm2Parser
from .symbol_table import SymbolTable
from .error import CompileError


class SemanticAnalyzer(qasm2Listener):
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.opaque_list = ['test_opaque']

    def assertNotDefined(self, name: str, token, msg: str):
        symbol = self.symbol_table.find(name)
        if symbol is not None:  # Not found
            symbol_file_name = symbol.ctx.start.getInputStream().fileName
            symbol_line = symbol.ctx.start.line
            symbol_column = symbol.ctx.start.column
            raise LookupError(CompileError(
                token.getInputStream().fileName,
                token.line,
                token.column,
                f"{msg}\n"
                f"\tDefined here: {symbol_file_name}:{symbol_line}:{symbol_column}"))

    # Enter a parse tree produced by qasm2Parser#mainprogram.
    def enterMainprogram(self, ctx: qasm2Parser.MainprogramContext):
        pass

    # Exit a parse tree produced by qasm2Parser#mainprogram.
    def exitMainprogram(self, ctx: qasm2Parser.MainprogramContext):
        pass

    # Enter a parse tree produced by qasm2Parser#version.
    def enterVersion(self, ctx: qasm2Parser.VersionContext):
        version = ctx.REAL().getText()
        if version != "2.0":
            raise ValueError(CompileError(
                ctx.start.getInputStream().fileName,
                ctx.start.line,
                ctx.start.column,
                f"Invalid version {version}"))

    # Exit a parse tree produced by qasm2Parser#version.
    def exitVersion(self, ctx: qasm2Parser.VersionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#program.
    def enterProgram(self, ctx: qasm2Parser.ProgramContext):
        pass

    # Exit a parse tree produced by qasm2Parser#program.
    def exitProgram(self, ctx: qasm2Parser.ProgramContext):
        pass

    # Enter a parse tree produced by qasm2Parser#statement.
    def enterStatement(self, ctx: qasm2Parser.StatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#statement.
    def exitStatement(self, ctx: qasm2Parser.StatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#includeStatement.
    def enterIncludeStatement(self, ctx: qasm2Parser.IncludeStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#includeStatement.
    def exitIncludeStatement(self, ctx: qasm2Parser.IncludeStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#declStatement.
    def enterDeclStatement(self, ctx: qasm2Parser.DeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#declStatement.
    def exitDeclStatement(self, ctx: qasm2Parser.DeclStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#regDeclStatement.
    def enterRegDeclStatement(self, ctx: qasm2Parser.RegDeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#regDeclStatement.
    def exitRegDeclStatement(self, ctx: qasm2Parser.RegDeclStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qregDeclStatement.
    def enterQregDeclStatement(self, ctx: qasm2Parser.QregDeclStatementContext):
        name = ctx.ID().getText()
        nninteger = int(ctx.NNINTEGER().getText())

        if nninteger == 0:
            raise ValueError(CompileError(
                ctx.start.getInputStream().fileName,
                ctx.start.line,
                ctx.start.column,
                f"Qreg dimension must be larger than {nninteger}"))

        self.assertNotDefined(name, ctx.start, f"Creg '{name}' has already been defined")
        self.symbol_table.insert(name, SymbolTable.Type.QREG, ctx)

    # Exit a parse tree produced by qasm2Parser#qregDeclStatement.
    def exitQregDeclStatement(self, ctx: qasm2Parser.QregDeclStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx: qasm2Parser.CregDeclStatementContext):
        name = ctx.ID().getText()
        nninteger = int(ctx.NNINTEGER().getText())

        if nninteger == 0:
            raise ValueError(CompileError(
                ctx.NNINTEGER().getInputStream().fileName,
                ctx.NNINTEGER().line,
                ctx.NNINTEGER().column,
                f"Qreg dimension must be larger than {nninteger}"))

        self.assertNotDefined(name, ctx.start, f"Creg '{name}' has already been defined")
        self.symbol_table.insert(name, SymbolTable.Type.CREG, ctx)

    # Exit a parse tree produced by qasm2Parser#cregDeclStatement.
    def exitCregDeclStatement(self, ctx: qasm2Parser.CregDeclStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gateDeclStatement.
    def enterGateDeclStatement(self, ctx: qasm2Parser.GateDeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gateDeclStatement.
    def exitGateDeclStatement(self, ctx: qasm2Parser.GateDeclStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#opaqueStatement.
    def enterOpaqueStatement(self, ctx: qasm2Parser.OpaqueStatementContext):
        name = ctx.ID().getText()

        self.assertNotDefined(name, ctx.start, f"Opaque gate '{name}' has already been defined")
        self.symbol_table.insert(name, SymbolTable.Type.OPAQUE, ctx)

        paramlist = ctx.paramlist()
        if paramlist is not None:
            param_ids = sorted([_id.getText() for _id in paramlist.ID()])
            for i in range(len(param_ids)-1):
                if param_ids[i] == param_ids[i+1]:
                    raise LookupError(CompileError(
                        ctx.start.getInputStream().fileName,
                        ctx.start.line,
                        ctx.start.column,
                        f"Opaque gate has a duplicated parameter '{param_ids[i]}'"))

        idlist = ctx.idlist()
        if idlist is not None:
            ids = sorted([_id.getText() for _id in idlist.ID()])
            for i in range(len(ids)-1):
                if ids[i] == ids[i+1]:
                    raise LookupError(CompileError(
                        ctx.start.getInputStream().fileName,
                        ctx.start.line,
                        ctx.start.column,
                        f"Opaque gate has a duplicated id '{ids[i]}'"))

        # TODO: Check number of parameters and ids for supported opaque gates
        if name not in self.opaque_list:
            raise ValueError(CompileError(
                ctx.start.getInputStream().fileName,
                ctx.start.line,
                ctx.start.column,
                f"Opaque '{name}' is not supported"))

    # Exit a parse tree produced by qasm2Parser#opaqueStatement.

    def exitOpaqueStatement(self, ctx: qasm2Parser.OpaqueStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gateStatement.
    def enterGateStatement(self, ctx: qasm2Parser.GateStatementContext):
        name = ctx.ID().getText()

        self.assertNotDefined(name, ctx.start, f"Gate '{name}' has already been defined")
        self.symbol_table.insert(name, SymbolTable.Type.GATE, ctx)

        paramlist = ctx.paramlist()
        if paramlist is not None:
            param_ids = sorted([_id.getText() for _id in paramlist.ID()])
            for i in range(len(param_ids)-1):
                if param_ids[i] == param_ids[i+1]:
                    raise LookupError(CompileError(
                        ctx.start.getInputStream().fileName,
                        ctx.start.line,
                        ctx.start.column,
                        f"Gate has a duplicated parameter '{param_ids[i]}'"))

        idlist = ctx.idlist()
        if idlist is not None:
            ids = sorted([_id.getText() for _id in idlist.ID()])
            for i in range(len(ids)-1):
                if ids[i] == ids[i+1]:
                    raise LookupError(CompileError(
                        ctx.start.getInputStream().fileName,
                        ctx.start.line,
                        ctx.start.column,
                        f"Gate has a duplicated id '{ids[i]}'"))

    # Exit a parse tree produced by qasm2Parser#gateStatement.
    def exitGateStatement(self, ctx: qasm2Parser.GateStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#goplist.
    def enterGoplist(self, ctx: qasm2Parser.GoplistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#goplist.
    def exitGoplist(self, ctx: qasm2Parser.GoplistContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gop.
    def enterGop(self, ctx: qasm2Parser.GopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gop.
    def exitGop(self, ctx: qasm2Parser.GopContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gopUGate.
    def enterGopUGate(self, ctx: qasm2Parser.GopUGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopUGate.
    def exitGopUGate(self, ctx: qasm2Parser.GopUGateContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gopCXGate.
    def enterGopCXGate(self, ctx: qasm2Parser.GopCXGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopCXGate.
    def exitGopCXGate(self, ctx: qasm2Parser.GopCXGateContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gopBarrier.
    def enterGopBarrier(self, ctx: qasm2Parser.GopBarrierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopBarrier.
    def exitGopBarrier(self, ctx: qasm2Parser.GopBarrierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gopCustomGate.
    def enterGopCustomGate(self, ctx: qasm2Parser.GopCustomGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopCustomGate.
    def exitGopCustomGate(self, ctx: qasm2Parser.GopCustomGateContext):
        pass

    # Enter a parse tree produced by qasm2Parser#gopReset.
    def enterGopReset(self, ctx: qasm2Parser.GopResetContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopReset.
    def exitGopReset(self, ctx: qasm2Parser.GopResetContext):
        pass

    # Enter a parse tree produced by qasm2Parser#idlist.
    def enterIdlist(self, ctx: qasm2Parser.IdlistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#idlist.
    def exitIdlist(self, ctx: qasm2Parser.IdlistContext):
        pass

    # Enter a parse tree produced by qasm2Parser#paramlist.
    def enterParamlist(self, ctx: qasm2Parser.ParamlistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#paramlist.
    def exitParamlist(self, ctx: qasm2Parser.ParamlistContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qopStatement.
    def enterQopStatement(self, ctx: qasm2Parser.QopStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopStatement.
    def exitQopStatement(self, ctx: qasm2Parser.QopStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qopUGate.
    def enterQopUGate(self, ctx: qasm2Parser.QopUGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopUGate.
    def exitQopUGate(self, ctx: qasm2Parser.QopUGateContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qopCXGate.
    def enterQopCXGate(self, ctx: qasm2Parser.QopCXGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopCXGate.
    def exitQopCXGate(self, ctx: qasm2Parser.QopCXGateContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qopMeasure.
    def enterQopMeasure(self, ctx: qasm2Parser.QopMeasureContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopMeasure.
    def exitQopMeasure(self, ctx: qasm2Parser.QopMeasureContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qopReset.
    def enterQopReset(self, ctx: qasm2Parser.QopResetContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopReset.
    def exitQopReset(self, ctx: qasm2Parser.QopResetContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qopCustomGate.
    def enterQopCustomGate(self, ctx: qasm2Parser.QopCustomGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopCustomGate.
    def exitQopCustomGate(self, ctx: qasm2Parser.QopCustomGateContext):
        pass

    # Enter a parse tree produced by qasm2Parser#ifStatement.
    def enterIfStatement(self, ctx: qasm2Parser.IfStatementContext):
        name = ctx.ID().getText()

        symbol = self.symbol_table.find(name)
        if symbol is None:  # Not found
            raise LookupError(CompileError(
                ctx.start.getInputStream().fileName,
                ctx.start.line,
                ctx.start.column,
                f"Creg '{name}' is undefined"))
        elif symbol.type != SymbolTable.Type.CREG:
            raise TypeError(CompileError(
                ctx.start.getInputStream().fileName,
                ctx.start.line,
                ctx.start.column,
                f"Symbol '{name}' does not refer to a creg"))

    # Exit a parse tree produced by qasm2Parser#ifStatement.
    def exitIfStatement(self, ctx: qasm2Parser.IfStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#barrierStatement.
    def enterBarrierStatement(self, ctx: qasm2Parser.BarrierStatementContext):
        qarg_map = {}
        for qarg in ctx.arglist().qarg():
            name = qarg.ID().getText()
            symbol = self.symbol_table.find(name)
            if symbol is None:  # Not found
                raise LookupError(CompileError(
                    qarg.start.getInputStream().fileName,
                    qarg.start.line,
                    qarg.start.column,
                    f"Reference to an undefined symbol '{name}'"))
            elif symbol.type != SymbolTable.Type.QREG:
                raise TypeError(CompileError(
                    qarg.start.getInputStream().fileName,
                    qarg.start.line,
                    qarg.start.column,
                    f"Symbol '{name}' does not refer to a qreg"))
            elif qarg.NNINTEGER() is not None:
                qdim = int(symbol.ctx.NNINTEGER().getText())
                qidx = None if qarg.NNINTEGER() is None else int(qarg.NNINTEGER().getText())
                if qidx >= qdim:
                    raise IndexError(CompileError(
                        qarg.start.getInputStream().fileName,
                        qarg.start.line,
                        qarg.start.column,
                        f"illegal indexing '{qarg.getText()}': the index {qidx} is >= than the dimension {qdim}"))

            qarg_dim = -1 if qarg.NNINTEGER() is None else int(qarg.NNINTEGER().getText())
            if name in qarg_map:
                if qarg_map[name] == -1 or qarg_dim == -1 or qarg_map[name] == qarg_dim:
                    ref_msg_0 = f"{name}" if qarg_map[name] == -1 else f"{name}{qarg_map[name]}"
                    ref_msg_1 = f"{name}" if qarg_dim == -1 else f"{name}{qarg_dim}"
                    raise ValueError(CompileError(
                        qarg.start.getInputStream().fileName,
                        qarg.start.line,
                        qarg.start.column,
                        f"Barrier does not allow duplicated qargs: '{ref_msg_0}' and '{ref_msg_1}'"))
            else:
                qarg_map[name] = qarg_dim

    # Exit a parse tree produced by qasm2Parser#barrierStatement.
    def exitBarrierStatement(self, ctx: qasm2Parser.BarrierStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#arglist.
    def enterArglist(self, ctx: qasm2Parser.ArglistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#arglist.
    def exitArglist(self, ctx: qasm2Parser.ArglistContext):
        pass

    # Enter a parse tree produced by qasm2Parser#qarg.
    def enterQarg(self, ctx: qasm2Parser.QargContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qarg.
    def exitQarg(self, ctx: qasm2Parser.QargContext):
        pass

    # Enter a parse tree produced by qasm2Parser#carg.
    def enterCarg(self, ctx: qasm2Parser.CargContext):
        pass

    # Exit a parse tree produced by qasm2Parser#carg.
    def exitCarg(self, ctx: qasm2Parser.CargContext):
        pass

    # Enter a parse tree produced by qasm2Parser#explist.
    def enterExplist(self, ctx: qasm2Parser.ExplistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#explist.
    def exitExplist(self, ctx: qasm2Parser.ExplistContext):
        pass

    # Enter a parse tree produced by qasm2Parser#exp.
    def enterExp(self, ctx: qasm2Parser.ExpContext):
        pass

    # Exit a parse tree produced by qasm2Parser#exp.
    def exitExp(self, ctx: qasm2Parser.ExpContext):
        pass

    # Enter a parse tree produced by qasm2Parser#binop.
    def enterBinop(self, ctx: qasm2Parser.BinopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#binop.
    def exitBinop(self, ctx: qasm2Parser.BinopContext):
        pass

    # Enter a parse tree produced by qasm2Parser#negop.
    def enterNegop(self, ctx: qasm2Parser.NegopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#negop.
    def exitNegop(self, ctx: qasm2Parser.NegopContext):
        pass

    # Enter a parse tree produced by qasm2Parser#unaryop.
    def enterUnaryop(self, ctx: qasm2Parser.UnaryopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#unaryop.
    def exitUnaryop(self, ctx: qasm2Parser.UnaryopContext):
        pass
