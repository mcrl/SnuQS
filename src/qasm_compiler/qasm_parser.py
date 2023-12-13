from .generated.QASMLexer import QASMLexer
from .generated.QASMParser import QASMParser
from antlr4 import InputStream, CommonTokenStream,  ParserRuleContext

from .qasm_exception import QasmException


class QasmParser:
    def parse(self, qasm: str) -> ParserRuleContext:
        try:
            stream = InputStream(qasm)
            lexer = QASMLexer(stream)
            stream = CommonTokenStream(lexer)
            parser = QASMParser(stream)
            tree = parser.mainprogram()
            return tree
        except Exception:
            print("Parsing error")
            raise QasmException("parsing error")
