from .qasm2_semantic_checker import Qasm2SemanticChecker
from .qasm2_opaque_gate_checker import Qasm2OpaqueGateChecker
from .qasm2_symbol_table import Qasm2SymbolTable
from .qasm2_circuit_generator import Qasm2CircuitGenerator
from .qasm2_tree_builder import Qasm2TreeBuilder
from .qasm2_walker import Qasm2Walker
from snuqs.circuit import Circuit


class Qasm2Compiler:
    """
    OpenQASM2 Compiler for SnuQS.
    """

    def compile(self, file_name: str):
        """
        compile OpenQASM2 file *file_name*.
        ----------------
        file_name: OpenQASM2 file name
        """
        try:
            circuit = Circuit(file_name)
            parser = Qasm2TreeBuilder()
            tree = parser.parse(file_name)

            symtab = Qasm2SymbolTable()
            stages = [
                Qasm2SemanticChecker(symtab),
                Qasm2OpaqueGateChecker(symtab),
                Qasm2CircuitGenerator(circuit, symtab),
            ]
            walker = Qasm2Walker()
            for stage in stages:
                walker.walk(stage, tree)
            return circuit

        except Exception as e:
            raise e

        return None
