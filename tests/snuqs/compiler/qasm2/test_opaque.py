import unittest

from snuqs.compiler.qasm2 import Qasm2Compiler


class QasmOpaqueTest(unittest.TestCase):
    def test_invalid_gate_call(self):
        file_name = "qasm2/opaque/invalid_gate_call.qasm"
        self.assertRaises(
            LookupError, lambda: Qasm2Compiler().compile(file_name))

    def test_valid_gate_calls(self):
        file_name = "qasm2/opaque/valid_gate_calls.qasm"
        Qasm2Compiler().compile(file_name)


if __name__ == '__main__':
    unittest.main()
