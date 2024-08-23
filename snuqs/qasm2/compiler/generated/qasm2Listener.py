# Generated from qasm2.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .qasm2Parser import qasm2Parser
else:
    from qasm2Parser import qasm2Parser

# This class defines a complete listener for a parse tree produced by qasm2Parser.
class qasm2Listener(ParseTreeListener):

    # Enter a parse tree produced by qasm2Parser#mainprogram.
    def enterMainprogram(self, ctx:qasm2Parser.MainprogramContext):
        pass

    # Exit a parse tree produced by qasm2Parser#mainprogram.
    def exitMainprogram(self, ctx:qasm2Parser.MainprogramContext):
        pass


    # Enter a parse tree produced by qasm2Parser#version.
    def enterVersion(self, ctx:qasm2Parser.VersionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#version.
    def exitVersion(self, ctx:qasm2Parser.VersionContext):
        pass


    # Enter a parse tree produced by qasm2Parser#program.
    def enterProgram(self, ctx:qasm2Parser.ProgramContext):
        pass

    # Exit a parse tree produced by qasm2Parser#program.
    def exitProgram(self, ctx:qasm2Parser.ProgramContext):
        pass


    # Enter a parse tree produced by qasm2Parser#statement.
    def enterStatement(self, ctx:qasm2Parser.StatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#statement.
    def exitStatement(self, ctx:qasm2Parser.StatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#includeStatement.
    def enterIncludeStatement(self, ctx:qasm2Parser.IncludeStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#includeStatement.
    def exitIncludeStatement(self, ctx:qasm2Parser.IncludeStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#declStatement.
    def enterDeclStatement(self, ctx:qasm2Parser.DeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#declStatement.
    def exitDeclStatement(self, ctx:qasm2Parser.DeclStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#regDeclStatement.
    def enterRegDeclStatement(self, ctx:qasm2Parser.RegDeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#regDeclStatement.
    def exitRegDeclStatement(self, ctx:qasm2Parser.RegDeclStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qregDeclStatement.
    def enterQregDeclStatement(self, ctx:qasm2Parser.QregDeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qregDeclStatement.
    def exitQregDeclStatement(self, ctx:qasm2Parser.QregDeclStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx:qasm2Parser.CregDeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#cregDeclStatement.
    def exitCregDeclStatement(self, ctx:qasm2Parser.CregDeclStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gateDeclStatement.
    def enterGateDeclStatement(self, ctx:qasm2Parser.GateDeclStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gateDeclStatement.
    def exitGateDeclStatement(self, ctx:qasm2Parser.GateDeclStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#opaqueStatement.
    def enterOpaqueStatement(self, ctx:qasm2Parser.OpaqueStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#opaqueStatement.
    def exitOpaqueStatement(self, ctx:qasm2Parser.OpaqueStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gateStatement.
    def enterGateStatement(self, ctx:qasm2Parser.GateStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gateStatement.
    def exitGateStatement(self, ctx:qasm2Parser.GateStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#goplist.
    def enterGoplist(self, ctx:qasm2Parser.GoplistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#goplist.
    def exitGoplist(self, ctx:qasm2Parser.GoplistContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gop.
    def enterGop(self, ctx:qasm2Parser.GopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gop.
    def exitGop(self, ctx:qasm2Parser.GopContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gopUGate.
    def enterGopUGate(self, ctx:qasm2Parser.GopUGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopUGate.
    def exitGopUGate(self, ctx:qasm2Parser.GopUGateContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gopCXGate.
    def enterGopCXGate(self, ctx:qasm2Parser.GopCXGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopCXGate.
    def exitGopCXGate(self, ctx:qasm2Parser.GopCXGateContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gopBarrier.
    def enterGopBarrier(self, ctx:qasm2Parser.GopBarrierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopBarrier.
    def exitGopBarrier(self, ctx:qasm2Parser.GopBarrierContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gopCustomGate.
    def enterGopCustomGate(self, ctx:qasm2Parser.GopCustomGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopCustomGate.
    def exitGopCustomGate(self, ctx:qasm2Parser.GopCustomGateContext):
        pass


    # Enter a parse tree produced by qasm2Parser#gopReset.
    def enterGopReset(self, ctx:qasm2Parser.GopResetContext):
        pass

    # Exit a parse tree produced by qasm2Parser#gopReset.
    def exitGopReset(self, ctx:qasm2Parser.GopResetContext):
        pass


    # Enter a parse tree produced by qasm2Parser#idlist.
    def enterIdlist(self, ctx:qasm2Parser.IdlistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#idlist.
    def exitIdlist(self, ctx:qasm2Parser.IdlistContext):
        pass


    # Enter a parse tree produced by qasm2Parser#paramlist.
    def enterParamlist(self, ctx:qasm2Parser.ParamlistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#paramlist.
    def exitParamlist(self, ctx:qasm2Parser.ParamlistContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qopStatement.
    def enterQopStatement(self, ctx:qasm2Parser.QopStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopStatement.
    def exitQopStatement(self, ctx:qasm2Parser.QopStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qopUGate.
    def enterQopUGate(self, ctx:qasm2Parser.QopUGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopUGate.
    def exitQopUGate(self, ctx:qasm2Parser.QopUGateContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qopCXGate.
    def enterQopCXGate(self, ctx:qasm2Parser.QopCXGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopCXGate.
    def exitQopCXGate(self, ctx:qasm2Parser.QopCXGateContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qopMeasure.
    def enterQopMeasure(self, ctx:qasm2Parser.QopMeasureContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopMeasure.
    def exitQopMeasure(self, ctx:qasm2Parser.QopMeasureContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qopReset.
    def enterQopReset(self, ctx:qasm2Parser.QopResetContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopReset.
    def exitQopReset(self, ctx:qasm2Parser.QopResetContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qopCustomGate.
    def enterQopCustomGate(self, ctx:qasm2Parser.QopCustomGateContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qopCustomGate.
    def exitQopCustomGate(self, ctx:qasm2Parser.QopCustomGateContext):
        pass


    # Enter a parse tree produced by qasm2Parser#ifStatement.
    def enterIfStatement(self, ctx:qasm2Parser.IfStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#ifStatement.
    def exitIfStatement(self, ctx:qasm2Parser.IfStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#barrierStatement.
    def enterBarrierStatement(self, ctx:qasm2Parser.BarrierStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#barrierStatement.
    def exitBarrierStatement(self, ctx:qasm2Parser.BarrierStatementContext):
        pass


    # Enter a parse tree produced by qasm2Parser#arglist.
    def enterArglist(self, ctx:qasm2Parser.ArglistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#arglist.
    def exitArglist(self, ctx:qasm2Parser.ArglistContext):
        pass


    # Enter a parse tree produced by qasm2Parser#qarg.
    def enterQarg(self, ctx:qasm2Parser.QargContext):
        pass

    # Exit a parse tree produced by qasm2Parser#qarg.
    def exitQarg(self, ctx:qasm2Parser.QargContext):
        pass


    # Enter a parse tree produced by qasm2Parser#carg.
    def enterCarg(self, ctx:qasm2Parser.CargContext):
        pass

    # Exit a parse tree produced by qasm2Parser#carg.
    def exitCarg(self, ctx:qasm2Parser.CargContext):
        pass


    # Enter a parse tree produced by qasm2Parser#explist.
    def enterExplist(self, ctx:qasm2Parser.ExplistContext):
        pass

    # Exit a parse tree produced by qasm2Parser#explist.
    def exitExplist(self, ctx:qasm2Parser.ExplistContext):
        pass


    # Enter a parse tree produced by qasm2Parser#exp.
    def enterExp(self, ctx:qasm2Parser.ExpContext):
        pass

    # Exit a parse tree produced by qasm2Parser#exp.
    def exitExp(self, ctx:qasm2Parser.ExpContext):
        pass


    # Enter a parse tree produced by qasm2Parser#binop.
    def enterBinop(self, ctx:qasm2Parser.BinopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#binop.
    def exitBinop(self, ctx:qasm2Parser.BinopContext):
        pass


    # Enter a parse tree produced by qasm2Parser#negop.
    def enterNegop(self, ctx:qasm2Parser.NegopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#negop.
    def exitNegop(self, ctx:qasm2Parser.NegopContext):
        pass


    # Enter a parse tree produced by qasm2Parser#unaryop.
    def enterUnaryop(self, ctx:qasm2Parser.UnaryopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#unaryop.
    def exitUnaryop(self, ctx:qasm2Parser.UnaryopContext):
        pass



del qasm2Parser