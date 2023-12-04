import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmCGTest(unittest.TestCase):
    def test_cg_uop(self):
        with open("qasm/compiler/cg_uop.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_cg_cx(self):
        with open("qasm/compiler/cg_cx.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_cg_measure(self):
        with open("qasm/compiler/cg_measure.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_cg_reset(self):
        with open("qasm/compiler/cg_reset.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_cg_barrier(self):
        with open("qasm/compiler/cg_barrier.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_cg_if(self):
        with open("qasm/compiler/cg_if.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_cg_custom(self):
        with open("qasm/compiler/cg_custom.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)


if __name__ == '__main__':
    unittest.main()
