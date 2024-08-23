import unittest
from snuqs import QasmCompiler, StatevectorSimulator

import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
import qiskit.quantum_info as qi
import qiskit
from qiskit.providers.aer import Aer

import warnings


def run_snuqs(file_name):
    compiler = QasmCompiler()
    circ = compiler.compile(file_name)
    sim = StatevectorSimulator()
    result = sim.run(circ)
    result.wait()
    state = result.get_statevector()
    return state


def run_qiskit(file_name):
    warnings.filterwarnings("ignore")
    circ = QuantumCircuit.from_qasm_file(file_name)
    circ.save_statevector()

    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(circ, simulator)
    result = simulator.run(circ).result()
    statevector = result.get_statevector(circ)
    return statevector.data


class OpaqueTest(unittest.TestCase):
    def test_id(self):
        state_snuqs = run_snuqs('qasm/opaque/id.qasm')
        state_qiskit = run_qiskit('qasm/opaque/id.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_x(self):
        state_snuqs = run_snuqs('qasm/opaque/x.qasm')
        state_qiskit = run_qiskit('qasm/opaque/x.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_y(self):
        state_snuqs = run_snuqs('qasm/opaque/y.qasm')
        state_qiskit = run_qiskit('qasm/opaque/y.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_z(self):
        state_snuqs = run_snuqs('qasm/opaque/z.qasm')
        state_qiskit = run_qiskit('qasm/opaque/z.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_h(self):
        state_snuqs = run_snuqs('qasm/opaque/h.qasm')
        state_qiskit = run_qiskit('qasm/opaque/h.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_s(self):
        state_snuqs = run_snuqs('qasm/opaque/s.qasm')
        state_qiskit = run_qiskit('qasm/opaque/s.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_sdg(self):
        state_snuqs = run_snuqs('qasm/opaque/sdg.qasm')
        state_qiskit = run_qiskit('qasm/opaque/sdg.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_t(self):
        state_snuqs = run_snuqs('qasm/opaque/t.qasm')
        state_qiskit = run_qiskit('qasm/opaque/t.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_tdg(self):
        state_snuqs = run_snuqs('qasm/opaque/tdg.qasm')
        state_qiskit = run_qiskit('qasm/opaque/tdg.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_sx(self):
        state_snuqs = run_snuqs('qasm/opaque/sx.qasm')
        state_qiskit = run_qiskit('qasm/opaque/sx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_sxdg(self):
        state_snuqs = run_snuqs('qasm/opaque/sxdg.qasm')
        state_qiskit = run_qiskit('qasm/opaque/sxdg.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_p(self):
        state_snuqs = run_snuqs('qasm/opaque/p.qasm')
        state_qiskit = run_qiskit('qasm/opaque/p.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rx(self):
        state_snuqs = run_snuqs('qasm/opaque/rx.qasm')
        state_qiskit = run_qiskit('qasm/opaque/rx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_ry(self):
        state_snuqs = run_snuqs('qasm/opaque/ry.qasm')
        state_qiskit = run_qiskit('qasm/opaque/ry.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rz(self):
        state_snuqs = run_snuqs('qasm/opaque/rz.qasm')
        state_qiskit = run_qiskit('qasm/opaque/rz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u0(self):
        state_snuqs = run_snuqs('qasm/opaque/u0.qasm')
        state_qiskit = run_qiskit('qasm/opaque/u0.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u1(self):
        state_snuqs = run_snuqs('qasm/opaque/u1.qasm')
        state_qiskit = run_qiskit('qasm/opaque/u1.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u2(self):
        state_snuqs = run_snuqs('qasm/opaque/u2.qasm')
        state_qiskit = run_qiskit('qasm/opaque/u2.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u3(self):
        state_snuqs = run_snuqs('qasm/opaque/u3.qasm')
        state_qiskit = run_qiskit('qasm/opaque/u3.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u(self):
        state_snuqs = run_snuqs('qasm/opaque/u.qasm')
        state_qiskit = run_qiskit('qasm/opaque/u.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cx(self):
        state_snuqs = run_snuqs('qasm/opaque/cx.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cy(self):
        state_snuqs = run_snuqs('qasm/opaque/cy.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cy.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cz(self):
        state_snuqs = run_snuqs('qasm/opaque/cz.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_swap(self):
        state_snuqs = run_snuqs('qasm/opaque/swap.qasm')
        state_qiskit = run_qiskit('qasm/opaque/swap.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_ch(self):
        state_snuqs = run_snuqs('qasm/opaque/ch.qasm')
        state_qiskit = run_qiskit('qasm/opaque/ch.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_csx(self):
        state_snuqs = run_snuqs('qasm/opaque/csx.qasm')
        state_qiskit = run_qiskit('qasm/opaque/csx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_crx(self):
        state_snuqs = run_snuqs('qasm/opaque/crx.qasm')
        state_qiskit = run_qiskit('qasm/opaque/crx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cry(self):
        state_snuqs = run_snuqs('qasm/opaque/cry.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cry.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_crz(self):
        state_snuqs = run_snuqs('qasm/opaque/crz.qasm')
        state_qiskit = run_qiskit('qasm/opaque/crz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cp(self):
        state_snuqs = run_snuqs('qasm/opaque/cp.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cp.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cu1(self):
        state_snuqs = run_snuqs('qasm/opaque/cu1.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cu1.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rxx(self):
        state_snuqs = run_snuqs('qasm/opaque/rxx.qasm')
        state_qiskit = run_qiskit('qasm/opaque/rxx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rzz(self):
        state_snuqs = run_snuqs('qasm/opaque/rzz.qasm')
        state_qiskit = run_qiskit('qasm/opaque/rzz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cu3(self):
        state_snuqs = run_snuqs('qasm/opaque/cu3.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cu3.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cu(self):
        state_snuqs = run_snuqs('qasm/opaque/cu.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cu.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_ccx(self):
        state_snuqs = run_snuqs('qasm/opaque/ccx.qasm')
        state_qiskit = run_qiskit('qasm/opaque/ccx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cswap(self):
        state_snuqs = run_snuqs('qasm/opaque/cswap.qasm')
        state_qiskit = run_qiskit('qasm/opaque/cswap.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()
