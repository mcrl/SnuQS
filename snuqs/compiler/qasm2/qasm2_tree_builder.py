from .generated.Qasm2Lexer import Qasm2Lexer
from .generated.Qasm2Parser import Qasm2Parser
from .generated.Qasm2Listener import Qasm2Listener
from antlr4 import ParseTreeWalker
from antlr4 import FileStream, CommonTokenStream, ParserRuleContext


import snuqs
import os

PREFIXES = [
    f"{snuqs.__path__[0]}/compiler/qasm2",
]


class Qasm2TreeBuilder(Qasm2Listener):
    def parse(self, file_name: str) -> ParserRuleContext:
        stream = FileStream(file_name)
        lexer = Qasm2Lexer(stream)
        stream = CommonTokenStream(lexer)
        parser = Qasm2Parser(stream)
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

    # Enter a parse tree produced by Qasm2Parser#includeStatement.
    def enterIncludeStatement(self, ctx: Qasm2Parser.IncludeStatementContext):
        file_name = ctx.STRING().getText()[1:-1]
        path = self.getPath(file_name)
        ctx.path = path
        ctx.tree = Qasm2TreeBuilder().parse(path)
