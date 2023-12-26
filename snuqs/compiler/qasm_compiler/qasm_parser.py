from .generated.QASMLexer import QASMLexer
from .generated.QASMParser import QASMParser
from .generated.QASMListener import QASMListener
from antlr4 import ParseTreeWalker, FileStream, CommonTokenStream, ParserRuleContext


import snuqs
import os

PREFIXES = [
    f"{snuqs.__path__[0]}/compiler/qasm_compiler",
]

class QasmParser(QASMListener):
    def parse(self, file_name: str) -> ParserRuleContext:
        stream = FileStream(file_name)
        lexer = QASMLexer(stream)
        stream = CommonTokenStream(lexer)
        parser = QASMParser(stream)
        tree = parser.mainprogram()

        if parser.getNumberOfSyntaxErrors() > 0:
            raise SyntaxError()

        walker = ParseTreeWalker()
        walker.walk(self, tree)

        return tree

    def getPath(self, file_name):
        if file_name[0] == '/' or file_name[0:2] == './':
            return file_name
        else:
            for prefix in PREFIXES:
                path = f"{prefix}/{file_name}"
                if os.path.exists(path):
                    return path

            raise FileNotFoundError(
                f"'{file_name}' does not exists")

    # Enter a parse tree produced by QASMParser#includeStatement.
    def enterIncludeStatement(self, ctx: QASMParser.IncludeStatementContext):
        file_name = ctx.STRING().getText()[1:-1]
        path = self.getPath(file_name)
        ctx.path = path
        ctx.tree = QasmParser().parse(path)
