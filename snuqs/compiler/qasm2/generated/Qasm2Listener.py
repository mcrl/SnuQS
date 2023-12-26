# Generated from Qasm2.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .Qasm2Parser import Qasm2Parser
else:
    from Qasm2Parser import Qasm2Parser

# This class defines a complete listener for a parse tree produced by Qasm2Parser.
class Qasm2Listener(ParseTreeListener):

    # Enter a parse tree produced by Qasm2Parser#mainprogram.
    def enterMainprogram(self, ctx:Qasm2Parser.MainprogramContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#mainprogram.
    def exitMainprogram(self, ctx:Qasm2Parser.MainprogramContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#version.
    def enterVersion(self, ctx:Qasm2Parser.VersionContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#version.
    def exitVersion(self, ctx:Qasm2Parser.VersionContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#program.
    def enterProgram(self, ctx:Qasm2Parser.ProgramContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#program.
    def exitProgram(self, ctx:Qasm2Parser.ProgramContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#statement.
    def enterStatement(self, ctx:Qasm2Parser.StatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#statement.
    def exitStatement(self, ctx:Qasm2Parser.StatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#includeStatement.
    def enterIncludeStatement(self, ctx:Qasm2Parser.IncludeStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#includeStatement.
    def exitIncludeStatement(self, ctx:Qasm2Parser.IncludeStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#declStatement.
    def enterDeclStatement(self, ctx:Qasm2Parser.DeclStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#declStatement.
    def exitDeclStatement(self, ctx:Qasm2Parser.DeclStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#regDeclStatement.
    def enterRegDeclStatement(self, ctx:Qasm2Parser.RegDeclStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#regDeclStatement.
    def exitRegDeclStatement(self, ctx:Qasm2Parser.RegDeclStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qregDeclStatement.
    def enterQregDeclStatement(self, ctx:Qasm2Parser.QregDeclStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qregDeclStatement.
    def exitQregDeclStatement(self, ctx:Qasm2Parser.QregDeclStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx:Qasm2Parser.CregDeclStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#cregDeclStatement.
    def exitCregDeclStatement(self, ctx:Qasm2Parser.CregDeclStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gateDeclStatement.
    def enterGateDeclStatement(self, ctx:Qasm2Parser.GateDeclStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gateDeclStatement.
    def exitGateDeclStatement(self, ctx:Qasm2Parser.GateDeclStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#opaqueStatement.
    def enterOpaqueStatement(self, ctx:Qasm2Parser.OpaqueStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#opaqueStatement.
    def exitOpaqueStatement(self, ctx:Qasm2Parser.OpaqueStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gateStatement.
    def enterGateStatement(self, ctx:Qasm2Parser.GateStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gateStatement.
    def exitGateStatement(self, ctx:Qasm2Parser.GateStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#goplist.
    def enterGoplist(self, ctx:Qasm2Parser.GoplistContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#goplist.
    def exitGoplist(self, ctx:Qasm2Parser.GoplistContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gop.
    def enterGop(self, ctx:Qasm2Parser.GopContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gop.
    def exitGop(self, ctx:Qasm2Parser.GopContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gopUGate.
    def enterGopUGate(self, ctx:Qasm2Parser.GopUGateContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gopUGate.
    def exitGopUGate(self, ctx:Qasm2Parser.GopUGateContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gopCXGate.
    def enterGopCXGate(self, ctx:Qasm2Parser.GopCXGateContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gopCXGate.
    def exitGopCXGate(self, ctx:Qasm2Parser.GopCXGateContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gopBarrier.
    def enterGopBarrier(self, ctx:Qasm2Parser.GopBarrierContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gopBarrier.
    def exitGopBarrier(self, ctx:Qasm2Parser.GopBarrierContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gopCustomGate.
    def enterGopCustomGate(self, ctx:Qasm2Parser.GopCustomGateContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gopCustomGate.
    def exitGopCustomGate(self, ctx:Qasm2Parser.GopCustomGateContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#gopReset.
    def enterGopReset(self, ctx:Qasm2Parser.GopResetContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#gopReset.
    def exitGopReset(self, ctx:Qasm2Parser.GopResetContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#idlist.
    def enterIdlist(self, ctx:Qasm2Parser.IdlistContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#idlist.
    def exitIdlist(self, ctx:Qasm2Parser.IdlistContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#paramlist.
    def enterParamlist(self, ctx:Qasm2Parser.ParamlistContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#paramlist.
    def exitParamlist(self, ctx:Qasm2Parser.ParamlistContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qopStatement.
    def enterQopStatement(self, ctx:Qasm2Parser.QopStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qopStatement.
    def exitQopStatement(self, ctx:Qasm2Parser.QopStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qopUGate.
    def enterQopUGate(self, ctx:Qasm2Parser.QopUGateContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qopUGate.
    def exitQopUGate(self, ctx:Qasm2Parser.QopUGateContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qopCXGate.
    def enterQopCXGate(self, ctx:Qasm2Parser.QopCXGateContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qopCXGate.
    def exitQopCXGate(self, ctx:Qasm2Parser.QopCXGateContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qopMeasure.
    def enterQopMeasure(self, ctx:Qasm2Parser.QopMeasureContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qopMeasure.
    def exitQopMeasure(self, ctx:Qasm2Parser.QopMeasureContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qopReset.
    def enterQopReset(self, ctx:Qasm2Parser.QopResetContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qopReset.
    def exitQopReset(self, ctx:Qasm2Parser.QopResetContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qopCustomGate.
    def enterQopCustomGate(self, ctx:Qasm2Parser.QopCustomGateContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qopCustomGate.
    def exitQopCustomGate(self, ctx:Qasm2Parser.QopCustomGateContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#ifStatement.
    def enterIfStatement(self, ctx:Qasm2Parser.IfStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#ifStatement.
    def exitIfStatement(self, ctx:Qasm2Parser.IfStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#barrierStatement.
    def enterBarrierStatement(self, ctx:Qasm2Parser.BarrierStatementContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#barrierStatement.
    def exitBarrierStatement(self, ctx:Qasm2Parser.BarrierStatementContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#arglist.
    def enterArglist(self, ctx:Qasm2Parser.ArglistContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#arglist.
    def exitArglist(self, ctx:Qasm2Parser.ArglistContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#qarg.
    def enterQarg(self, ctx:Qasm2Parser.QargContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#qarg.
    def exitQarg(self, ctx:Qasm2Parser.QargContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#carg.
    def enterCarg(self, ctx:Qasm2Parser.CargContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#carg.
    def exitCarg(self, ctx:Qasm2Parser.CargContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#explist.
    def enterExplist(self, ctx:Qasm2Parser.ExplistContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#explist.
    def exitExplist(self, ctx:Qasm2Parser.ExplistContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#exp.
    def enterExp(self, ctx:Qasm2Parser.ExpContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#exp.
    def exitExp(self, ctx:Qasm2Parser.ExpContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#complex.
    def enterComplex(self, ctx:Qasm2Parser.ComplexContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#complex.
    def exitComplex(self, ctx:Qasm2Parser.ComplexContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#addsub.
    def enterAddsub(self, ctx:Qasm2Parser.AddsubContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#addsub.
    def exitAddsub(self, ctx:Qasm2Parser.AddsubContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#binop.
    def enterBinop(self, ctx:Qasm2Parser.BinopContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#binop.
    def exitBinop(self, ctx:Qasm2Parser.BinopContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#negop.
    def enterNegop(self, ctx:Qasm2Parser.NegopContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#negop.
    def exitNegop(self, ctx:Qasm2Parser.NegopContext):
        pass


    # Enter a parse tree produced by Qasm2Parser#unaryop.
    def enterUnaryop(self, ctx:Qasm2Parser.UnaryopContext):
        pass

    # Exit a parse tree produced by Qasm2Parser#unaryop.
    def exitUnaryop(self, ctx:Qasm2Parser.UnaryopContext):
        pass



del Qasm2Parser