import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmCGTest(unittest.TestCase):
#    def test_valid_u(self):
#        file_name = "qasm/cg/valid_u.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_valid_cx(self):
#        file_name = "qasm/cg/valid_cx.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_valid_measure(self):
#        file_name = "qasm/cg/valid_measure.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_valid_reset(self):
#        file_name = "qasm/cg/valid_reset.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_valid_barrier(self):
#        file_name = "qasm/cg/valid_barrier.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_valid_if(self):
#        file_name = "qasm/cg/valid_if.qasm"
#        QasmCompiler().compile(file_name)

    def test_valid_custom(self):
        file_name = "qasm/cg/valid_custom.qasm"
        QasmCompiler().compile(file_name)


if __name__ == '__main__':
    unittest.main()
