from .qasm_preprocessor import QasmPreprocessor
from .qasm_semantic_checker import QasmSemanticChecker
from .qasm_opaque_gate_checker import QasmOpaqueGateChecker
from .qasm_symbol_table import QasmSymbolTable
from .qasm_circuit_generator import QasmCircuitGenerator
from .qasm_parser import QasmParser
from .qasm_walker import QasmWalker
from .qasm_exception import QasmException
from snuqs.circuit import Circuit


class QasmCompiler:
    """
    OpenQASM Compiler for SnuQS.
    Only OpenQASM 2.0 is supported for now.
    """

    def compile(self, file_name: str):
        """
        compile OpenQASM file *file_name*.
        ----------------
        file_name: OpenQASM file name
        """
        try:
            circuit = Circuit(file_name)
            pp = QasmPreprocessor()
            qasm_preprocessed = pp.preprocess(file_name)

            parser = QasmParser()
            tree = parser.parse(qasm_preprocessed)

            symtab = QasmSymbolTable()
            stages = [
                QasmSemanticChecker(symtab),
                QasmOpaqueGateChecker(symtab),
                QasmCircuitGenerator(circuit, symtab),
            ]

            walker = QasmWalker()
            for stage in stages:
                walker.walk(stage, tree)
            return circuit

        except QasmException as e:
            raise e

        return None
