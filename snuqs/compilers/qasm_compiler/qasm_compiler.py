from .generated.QASMLexer import QASMLexer
from .generated.QASMListener import QASMListener
from .generated.QASMParser import QASMParser

from .preprocessor import Preprocessor
from .semantic_checker import SemanticChecker
from .circuit_generator import CircuitGenerator
from .qasm_exception import QASMException
from .qasm_context import QASMContext

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
import os

class QASMCompiler:
    IN_FILE_NAME = '.tmp.in.qasm'
    OUT_FILE_NAME = '.tmp.out.qasm'

    def __init__(self):
        pass

    @classmethod
    def to_qasm(cls, qc):
        import qiskit
        qc_str = ""

        if isinstance(qc, str):
            qc_str = open(qc).read()
        elif isinstance(qc, qiskit.circuit.quantumcircuit.QuantumCircuit):
            qc_str = qc.qasm()
        else:
            raise QASMException(f"Not supported quantum circuit type {type(qc)}")

        temp = open(f'/tmp/{cls.IN_FILE_NAME}', 'w')
        temp.write(qc_str)
        temp.close()
        return f'/tmp/{cls.IN_FILE_NAME}'

    def compile(self, fname):
        pp = Preprocessor()
        
        tmp_file = f"/tmp/{self.OUT_FILE_NAME}"
        tmp_file = pp.run(fname, out=tmp_file)


        input_stream = FileStream(tmp_file)
        lexer = QASMLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = QASMParser(stream)
        tree = parser.mainprogram()

        # FIXME: Parsing error

        walker = ParseTreeWalker()

        ctx = QASMContext()
        stages = [ 
          SemanticChecker(ctx),
        ]

        for stage in stages:
            walker.walk(stage, tree)



        gen = CircuitGenerator(ctx)
        walker.walk(gen, tree)

        os.remove(tmp_file)

        circ = gen.get_circuit()
        qreg = gen.qreg
        creg = gen.creg

        return circ, qreg, creg
