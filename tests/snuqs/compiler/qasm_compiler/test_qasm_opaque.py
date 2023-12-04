import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmOpaqueTest(unittest.TestCase):
    def test_opaque_invalid_gate(self):
        with open("qasm/compiler/opaque_invalid_gate.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_opaque_gate_calls(self):
        with open("qasm/compiler/opaque_gate_calls.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)


if __name__ == '__main__':
    unittest.main()
