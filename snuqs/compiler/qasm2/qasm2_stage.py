from .generated.Qasm2Parser import Qasm2Parser
from .generated.Qasm2Listener import Qasm2Listener
from antlr4 import ParseTreeWalker


class Qasm2Stage(Qasm2Listener):
    def __init__(self):
        self.walker = ParseTreeWalker()

    # Enter a parse tree produced by Qasm2Parser#includeStatement.
    def enterIncludeStatement(self, ctx: Qasm2Parser.IncludeStatementContext):
        self.walker.walk(self, ctx.tree)
