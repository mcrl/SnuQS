from .generated import qasm2Lexer, qasm2Parser, qasm2Listener
from .error import CompileError
from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
import os
from typing import List, Optional


def is_double_quoted(text):
    return len(text) >= 2 and text[0] == '"' and text[-1] == '"'


def undoublequote(text):
    return text[1:-1]


class ParsingErrorListener(object):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        file_name = offendingSymbol.getInputStream().fileName
        raise SyntaxError(
            CompileError(file_name, line, column, f"Invalid syntax for preprocessor: {msg}"))

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        pass

    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        pass

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        pass


class Parser(qasm2Listener):
    def __init__(self, include_dirs: Optional[List[str]] = ["."]):
        import snuqs
        self.include_dirs = include_dirs
        include_dirs.append(f"{snuqs.__path__[0]}/qasm2/headers")

    def find_path(self, file_name: str) -> Optional[str]:
        for include_dir in self.include_dirs:
            path = f"{include_dir}/{file_name}"
            if os.path.isfile(path):
                return path

        return None

    def build_tree(self, file_name: str):
        input_stream = FileStream(file_name)
        lexer = qasm2Lexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = qasm2Parser(stream)
        # parser.addErrorListener(DiagnosticErrorListener())
        parser.removeErrorListeners()
        parser.addErrorListener(ParsingErrorListener())
        tree = parser.mainprogram()

        return tree

    def parse(self, path: str) -> str:
        tree = self.build_tree(path)
        walker = ParseTreeWalker()

        while True:
            self.running = False
            walker.walk(self, tree)
            if not self.running:
                break

        return tree

    # Enter a parse tree produced by qasm2Parser#includeStatement.
    def enterIncludeStatement(self, ctx: qasm2Parser.IncludeStatementContext):
        include_name = undoublequote(ctx.STRING().getText())
        path = self.find_path(include_name)
        if path is None:
            raise FileNotFoundError(CompileError(
                ctx.STRING().symbol.getInputStream().fileName,
                ctx.STRING().symbol.line,
                ctx.STRING().symbol.column,
                f"No such a file: '{include_name}'"))

        tree = self.build_tree(path)
        for i in range(len(ctx.parentCtx.children)):
            if ctx.parentCtx.children[i] == ctx:
                tree.parentCtx = ctx.parentCtx
                ctx.parentCtx.children[i] = tree
                self.running = True
