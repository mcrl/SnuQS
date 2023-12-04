import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmOpaqueTest(unittest.TestCase):
    def test_invalid_gate_call(self):
        file_name = "qasm/opaque/invalid_gate_call.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_valid_gate_calls(self):
        file_name = "qasm/opaque/valid_gate_calls.qasm"
        QasmCompiler().compile(file_name)


if __name__ == '__main__':
    unittest.main()
