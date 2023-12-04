from .qasm_preprocessor import QasmPreprocessor
from .qasm_semantic_checker import QasmSemanticChecker
from .qasm_context import QasmContext
from .qasm_circuit_generator import QasmCircuitGenerator
from .qasm_parser import QasmParser
from .qasm_walker import QasmWalker
from .qasm_exception import QasmException

class QasmCompiler:
    """
    OpenQASM Compiler for SnuQS.
    Only OpenQASM 2.0 is supported for now.
    """

    def compile(self, qasm: str):
        """
        compile *qasm* string.
        ----------------
        qasm: OpenQASM string
        """
        try:
            pp = QasmPreprocessor()
            qasm_preprocessed = pp.preprocess(qasm)

            parser = QasmParser()
            tree = parser.parse(qasm_preprocessed)

            ctx = QasmContext()
            stages = [
                QasmSemanticChecker(ctx),
                QasmCircuitGenerator(ctx),
            ]

            walker = QasmWalker()
            for stage in stages:
                walker.walk(stage, tree)
            return ctx.circ

        except QasmException as e:
            raise e

        return None
