import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class BnechmarkTest(unittest.TestCase):

    def test_adder(self):
        with open("qasm/benchmark/adder.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_bigadder(self):
        with open("qasm/benchmark/bigadder.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_entangled_registers(self):
        with open("qasm/benchmark/entangled_registers.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_invalid_gate_no_found(self):
        with open("qasm/benchmark/invalid_gate_no_found.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_invalid_missing_semicolon(self):
        with open("qasm/benchmark/invalid_missing_semicolon.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(Exception, lambda: comp.compile(qasm))

    def test_inverseqft1(self):
        with open("qasm/benchmark/inverseqft1.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_inverseqft2(self):
        with open("qasm/benchmark/inverseqft2.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_ipea_3_pi_8(self):
        with open("qasm/benchmark/ipea_3_pi_8.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_pea_3_pi_8(self):
        with open("qasm/benchmark/pea_3_pi_8.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_plaquette_check(self):
        with open("qasm/benchmark/plaquette_check.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_qec(self):
        with open("qasm/benchmark/qec.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_qft(self):
        with open("qasm/benchmark/qft.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_qpt(self):
        with open("qasm/benchmark/qpt.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_rb(self):
        with open("qasm/benchmark/rb.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_teleport(self):
        with open("qasm/benchmark/teleport.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_teleportv2(self):
        with open("qasm/benchmark/teleportv2.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_test(self):
        with open("qasm/benchmark/test.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)

    def test_W(self):
        with open("qasm/benchmark/W-state.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        comp.compile(qasm)


if __name__ == '__main__':
    unittest.main()
