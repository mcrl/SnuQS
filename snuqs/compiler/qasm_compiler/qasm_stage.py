from .generated.QASMParser import QASMParser
from .generated.QASMListener import QASMListener
from antlr4 import ParseTreeWalker


class QasmStage(QASMListener):
    def __init__(self):
        self.walker = ParseTreeWalker()

    # Enter a parse tree produced by QASMParser#includeStatement.
    def enterIncludeStatement(self, ctx: QASMParser.IncludeStatementContext):
        self.walker.walk(self, ctx.tree)
