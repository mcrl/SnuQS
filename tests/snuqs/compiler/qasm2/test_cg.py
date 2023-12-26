import unittest

from snuqs.compiler.qasm2 import Qasm2Compiler


class QasmCGTest(unittest.TestCase):
    def test_valid_u(self):
        file_name = "qasm2/cg/valid_u.qasm"
        Qasm2Compiler().compile(file_name)

    def test_valid_cx(self):
        file_name = "qasm2/cg/valid_cx.qasm"
        Qasm2Compiler().compile(file_name)

    def test_valid_measure(self):
        file_name = "qasm2/cg/valid_measure.qasm"
        Qasm2Compiler().compile(file_name)

    def test_valid_reset(self):
        file_name = "qasm2/cg/valid_reset.qasm"
        Qasm2Compiler().compile(file_name)

    def test_valid_barrier(self):
        file_name = "qasm2/cg/valid_barrier.qasm"
        Qasm2Compiler().compile(file_name)

    def test_valid_if(self):
        file_name = "qasm2/cg/valid_if.qasm"
        Qasm2Compiler().compile(file_name)

    def test_valid_custom(self):
        file_name = "qasm2/cg/valid_custom.qasm"
        Qasm2Compiler().compile(file_name)


if __name__ == '__main__':
    unittest.main()
