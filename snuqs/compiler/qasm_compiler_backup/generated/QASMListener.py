# Generated from QASM.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .QASMParser import QASMParser
else:
    from QASMParser import QASMParser

# This class defines a complete listener for a parse tree produced by QASMParser.
class QASMListener(ParseTreeListener):

    # Enter a parse tree produced by QASMParser#mainprogram.
    def enterMainprogram(self, ctx:QASMParser.MainprogramContext):
        pass

    # Exit a parse tree produced by QASMParser#mainprogram.
    def exitMainprogram(self, ctx:QASMParser.MainprogramContext):
        pass


    # Enter a parse tree produced by QASMParser#version.
    def enterVersion(self, ctx:QASMParser.VersionContext):
        pass

    # Exit a parse tree produced by QASMParser#version.
    def exitVersion(self, ctx:QASMParser.VersionContext):
        pass


    # Enter a parse tree produced by QASMParser#program.
    def enterProgram(self, ctx:QASMParser.ProgramContext):
        pass

    # Exit a parse tree produced by QASMParser#program.
    def exitProgram(self, ctx:QASMParser.ProgramContext):
        pass


    # Enter a parse tree produced by QASMParser#statement.
    def enterStatement(self, ctx:QASMParser.StatementContext):
        pass

    # Exit a parse tree produced by QASMParser#statement.
    def exitStatement(self, ctx:QASMParser.StatementContext):
        pass


    # Enter a parse tree produced by QASMParser#declStatement.
    def enterDeclStatement(self, ctx:QASMParser.DeclStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#declStatement.
    def exitDeclStatement(self, ctx:QASMParser.DeclStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#qregDeclStatement.
    def enterQregDeclStatement(self, ctx:QASMParser.QregDeclStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#qregDeclStatement.
    def exitQregDeclStatement(self, ctx:QASMParser.QregDeclStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#cregDeclStatement.
    def enterCregDeclStatement(self, ctx:QASMParser.CregDeclStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#cregDeclStatement.
    def exitCregDeclStatement(self, ctx:QASMParser.CregDeclStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#gatedeclStatement.
    def enterGatedeclStatement(self, ctx:QASMParser.GatedeclStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#gatedeclStatement.
    def exitGatedeclStatement(self, ctx:QASMParser.GatedeclStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#gatedecl.
    def enterGatedecl(self, ctx:QASMParser.GatedeclContext):
        pass

    # Exit a parse tree produced by QASMParser#gatedecl.
    def exitGatedecl(self, ctx:QASMParser.GatedeclContext):
        pass


    # Enter a parse tree produced by QASMParser#goplist.
    def enterGoplist(self, ctx:QASMParser.GoplistContext):
        pass

    # Exit a parse tree produced by QASMParser#goplist.
    def exitGoplist(self, ctx:QASMParser.GoplistContext):
        pass


    # Enter a parse tree produced by QASMParser#gop.
    def enterGop(self, ctx:QASMParser.GopContext):
        pass

    # Exit a parse tree produced by QASMParser#gop.
    def exitGop(self, ctx:QASMParser.GopContext):
        pass


    # Enter a parse tree produced by QASMParser#gopUGate.
    def enterGopUGate(self, ctx:QASMParser.GopUGateContext):
        pass

    # Exit a parse tree produced by QASMParser#gopUGate.
    def exitGopUGate(self, ctx:QASMParser.GopUGateContext):
        pass


    # Enter a parse tree produced by QASMParser#gopCXGate.
    def enterGopCXGate(self, ctx:QASMParser.GopCXGateContext):
        pass

    # Exit a parse tree produced by QASMParser#gopCXGate.
    def exitGopCXGate(self, ctx:QASMParser.GopCXGateContext):
        pass


    # Enter a parse tree produced by QASMParser#gopCustomGate.
    def enterGopCustomGate(self, ctx:QASMParser.GopCustomGateContext):
        pass

    # Exit a parse tree produced by QASMParser#gopCustomGate.
    def exitGopCustomGate(self, ctx:QASMParser.GopCustomGateContext):
        pass


    # Enter a parse tree produced by QASMParser#gopBarrier.
    def enterGopBarrier(self, ctx:QASMParser.GopBarrierContext):
        pass

    # Exit a parse tree produced by QASMParser#gopBarrier.
    def exitGopBarrier(self, ctx:QASMParser.GopBarrierContext):
        pass


    # Enter a parse tree produced by QASMParser#paramlist.
    def enterParamlist(self, ctx:QASMParser.ParamlistContext):
        pass

    # Exit a parse tree produced by QASMParser#paramlist.
    def exitParamlist(self, ctx:QASMParser.ParamlistContext):
        pass


    # Enter a parse tree produced by QASMParser#idlist.
    def enterIdlist(self, ctx:QASMParser.IdlistContext):
        pass

    # Exit a parse tree produced by QASMParser#idlist.
    def exitIdlist(self, ctx:QASMParser.IdlistContext):
        pass


    # Enter a parse tree produced by QASMParser#opaqueStatement.
    def enterOpaqueStatement(self, ctx:QASMParser.OpaqueStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#opaqueStatement.
    def exitOpaqueStatement(self, ctx:QASMParser.OpaqueStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#qopStatement.
    def enterQopStatement(self, ctx:QASMParser.QopStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#qopStatement.
    def exitQopStatement(self, ctx:QASMParser.QopStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#qopUGate.
    def enterQopUGate(self, ctx:QASMParser.QopUGateContext):
        pass

    # Exit a parse tree produced by QASMParser#qopUGate.
    def exitQopUGate(self, ctx:QASMParser.QopUGateContext):
        pass


    # Enter a parse tree produced by QASMParser#qopCXGate.
    def enterQopCXGate(self, ctx:QASMParser.QopCXGateContext):
        pass

    # Exit a parse tree produced by QASMParser#qopCXGate.
    def exitQopCXGate(self, ctx:QASMParser.QopCXGateContext):
        pass


    # Enter a parse tree produced by QASMParser#qopCustomGate.
    def enterQopCustomGate(self, ctx:QASMParser.QopCustomGateContext):
        pass

    # Exit a parse tree produced by QASMParser#qopCustomGate.
    def exitQopCustomGate(self, ctx:QASMParser.QopCustomGateContext):
        pass


    # Enter a parse tree produced by QASMParser#qopMeasure.
    def enterQopMeasure(self, ctx:QASMParser.QopMeasureContext):
        pass

    # Exit a parse tree produced by QASMParser#qopMeasure.
    def exitQopMeasure(self, ctx:QASMParser.QopMeasureContext):
        pass


    # Enter a parse tree produced by QASMParser#qopReset.
    def enterQopReset(self, ctx:QASMParser.QopResetContext):
        pass

    # Exit a parse tree produced by QASMParser#qopReset.
    def exitQopReset(self, ctx:QASMParser.QopResetContext):
        pass


    # Enter a parse tree produced by QASMParser#ifStatement.
    def enterIfStatement(self, ctx:QASMParser.IfStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#ifStatement.
    def exitIfStatement(self, ctx:QASMParser.IfStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#barrierStatement.
    def enterBarrierStatement(self, ctx:QASMParser.BarrierStatementContext):
        pass

    # Exit a parse tree produced by QASMParser#barrierStatement.
    def exitBarrierStatement(self, ctx:QASMParser.BarrierStatementContext):
        pass


    # Enter a parse tree produced by QASMParser#arglist.
    def enterArglist(self, ctx:QASMParser.ArglistContext):
        pass

    # Exit a parse tree produced by QASMParser#arglist.
    def exitArglist(self, ctx:QASMParser.ArglistContext):
        pass


    # Enter a parse tree produced by QASMParser#argument.
    def enterArgument(self, ctx:QASMParser.ArgumentContext):
        pass

    # Exit a parse tree produced by QASMParser#argument.
    def exitArgument(self, ctx:QASMParser.ArgumentContext):
        pass


    # Enter a parse tree produced by QASMParser#explist.
    def enterExplist(self, ctx:QASMParser.ExplistContext):
        pass

    # Exit a parse tree produced by QASMParser#explist.
    def exitExplist(self, ctx:QASMParser.ExplistContext):
        pass


    # Enter a parse tree produced by QASMParser#exp.
    def enterExp(self, ctx:QASMParser.ExpContext):
        pass

    # Exit a parse tree produced by QASMParser#exp.
    def exitExp(self, ctx:QASMParser.ExpContext):
        pass


    # Enter a parse tree produced by QASMParser#binop.
    def enterBinop(self, ctx:QASMParser.BinopContext):
        pass

    # Exit a parse tree produced by QASMParser#binop.
    def exitBinop(self, ctx:QASMParser.BinopContext):
        pass


    # Enter a parse tree produced by QASMParser#negop.
    def enterNegop(self, ctx:QASMParser.NegopContext):
        pass

    # Exit a parse tree produced by QASMParser#negop.
    def exitNegop(self, ctx:QASMParser.NegopContext):
        pass


    # Enter a parse tree produced by QASMParser#unaryop.
    def enterUnaryop(self, ctx:QASMParser.UnaryopContext):
        pass

    # Exit a parse tree produced by QASMParser#unaryop.
    def exitUnaryop(self, ctx:QASMParser.UnaryopContext):
        pass



del QASMParser
