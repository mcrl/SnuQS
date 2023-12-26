import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class BenchmarkTest(unittest.TestCase):
#    def test_adder(self):
#        file_name = "qasm/benchmark/adder.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_bigadder(self):
#        file_name = "qasm/benchmark/bigadder.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_entangled_registers(self):
#        file_name = "qasm/benchmark/entangled_registers.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_invalid_gate_no_found(self):
#        file_name = "qasm/benchmark/invalid_gate_no_found.qasm"
#        self.assertRaises(
#            LookupError, lambda: QasmCompiler().compile(file_name))
#
#    def test_invalid_missing_semicolon(self):
#        file_name = "qasm/benchmark/invalid_missing_semicolon.qasm"
#        self.assertRaises(SyntaxError, lambda: QasmCompiler().compile(file_name))
#
#    def test_inverseqft1(self):
#        file_name = "qasm/benchmark/inverseqft1.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_inverseqft2(self):
#        file_name = "qasm/benchmark/inverseqft2.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_ipea_3_pi_8(self):
#        file_name = "qasm/benchmark/ipea_3_pi_8.qasm"
#        self.assertRaises(
#            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_pea_3_pi_8(self):
        file_name = "qasm/benchmark/pea_3_pi_8.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

#    def test_plaquette_check(self):
#        file_name = "qasm/benchmark/plaquette_check.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_qec(self):
#        file_name = "qasm/benchmark/qec.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_qft(self):
#        file_name = "qasm/benchmark/qft.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_qpt(self):
#        file_name = "qasm/benchmark/qpt.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_rb(self):
#        file_name = "qasm/benchmark/rb.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_teleport(self):
#        file_name = "qasm/benchmark/teleport.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_teleportv2(self):
#        file_name = "qasm/benchmark/teleportv2.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_test(self):
#        file_name = "qasm/benchmark/test.qasm"
#        QasmCompiler().compile(file_name)
#
#    def test_W(self):
#        file_name = "qasm/benchmark/W-state.qasm"
#        QasmCompiler().compile(file_name)
#
#
if __name__ == '__main__':
    unittest.main()
