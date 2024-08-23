from antlr4 import ParseTreeWalker
from .parser import Parser
from .semantic_analyzer import SemanticAnalyzer
from .symbol_table import SymbolTable

from typing import List, Optional


class Compiler:
    def compile(self, file_name: str, include_dirs: Optional[List[str]] = ["."]):
        import snuqs
        include_dirs.append(f"{snuqs.__path__[0]}/qasm2/headers")
        parser = Parser(include_dirs)
        tree = parser.parse(file_name)

        symbol_table = SymbolTable()
        walker = ParseTreeWalker()
        walker.walk(SemanticChecker(symbol_table), tree)
