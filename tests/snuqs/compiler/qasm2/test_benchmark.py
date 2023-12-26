import unittest

from snuqs.compiler.qasm2 import Qasm2Compiler


class BenchmarkTest(unittest.TestCase):
    def test_adder(self):
        file_name = "qasm2/benchmark/adder.qasm"
        Qasm2Compiler().compile(file_name)

    def test_bigadder(self):
        file_name = "qasm2/benchmark/bigadder.qasm"
        Qasm2Compiler().compile(file_name)

    def test_entangled_registers(self):
        file_name = "qasm2/benchmark/entangled_registers.qasm"
        Qasm2Compiler().compile(file_name)

    def test_invalid_gate_no_found(self):
        file_name = "qasm2/benchmark/invalid_gate_no_found.qasm"
        self.assertRaises(
            LookupError, lambda: Qasm2Compiler().compile(file_name))

    def test_invalid_missing_semicolon(self):
        file_name = "qasm2/benchmark/invalid_missing_semicolon.qasm"
        self.assertRaises(SyntaxError, lambda: Qasm2Compiler().compile(file_name))

    def test_inverseqft1(self):
        file_name = "qasm2/benchmark/inverseqft1.qasm"
        Qasm2Compiler().compile(file_name)

    def test_inverseqft2(self):
        file_name = "qasm2/benchmark/inverseqft2.qasm"
        Qasm2Compiler().compile(file_name)

    def test_ipea_3_pi_8(self):
        file_name = "qasm2/benchmark/ipea_3_pi_8.qasm"
        self.assertRaises(
            LookupError, lambda: Qasm2Compiler().compile(file_name))

    def test_pea_3_pi_8(self):
        file_name = "qasm2/benchmark/pea_3_pi_8.qasm"
        self.assertRaises(
            LookupError, lambda: Qasm2Compiler().compile(file_name))

    def test_plaquette_check(self):
        file_name = "qasm2/benchmark/plaquette_check.qasm"
        Qasm2Compiler().compile(file_name)

    def test_qec(self):
        file_name = "qasm2/benchmark/qec.qasm"
        Qasm2Compiler().compile(file_name)

    def test_qft(self):
        file_name = "qasm2/benchmark/qft.qasm"
        Qasm2Compiler().compile(file_name)

    def test_qpt(self):
        file_name = "qasm2/benchmark/qpt.qasm"
        Qasm2Compiler().compile(file_name)

    def test_rb(self):
        file_name = "qasm2/benchmark/rb.qasm"
        Qasm2Compiler().compile(file_name)

    def test_teleport(self):
        file_name = "qasm2/benchmark/teleport.qasm"
        Qasm2Compiler().compile(file_name)

    def test_teleportv2(self):
        file_name = "qasm2/benchmark/teleportv2.qasm"
        Qasm2Compiler().compile(file_name)

    def test_test(self):
        file_name = "qasm2/benchmark/test.qasm"
        Qasm2Compiler().compile(file_name)

    def test_W(self):
        file_name = "qasm2/benchmark/W-state.qasm"
        Qasm2Compiler().compile(file_name)


if __name__ == '__main__':
    unittest.main()
