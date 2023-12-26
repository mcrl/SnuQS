import unittest
from snuqs import Qasm2Compiler, StatevectorSimulator

import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
import qiskit.quantum_info as qi
import qiskit
from qiskit.providers.aer import Aer

import warnings


def run_snuqs(file_name):
    compiler = Qasm2Compiler()
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
        state_snuqs = run_snuqs('qasm2/opaque/id.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/id.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_x(self):
        state_snuqs = run_snuqs('qasm2/opaque/x.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/x.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_y(self):
        state_snuqs = run_snuqs('qasm2/opaque/y.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/y.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_z(self):
        state_snuqs = run_snuqs('qasm2/opaque/z.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/z.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_h(self):
        state_snuqs = run_snuqs('qasm2/opaque/h.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/h.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_s(self):
        state_snuqs = run_snuqs('qasm2/opaque/s.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/s.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_sdg(self):
        state_snuqs = run_snuqs('qasm2/opaque/sdg.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/sdg.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_t(self):
        state_snuqs = run_snuqs('qasm2/opaque/t.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/t.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_tdg(self):
        state_snuqs = run_snuqs('qasm2/opaque/tdg.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/tdg.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_sx(self):
        state_snuqs = run_snuqs('qasm2/opaque/sx.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/sx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_sxdg(self):
        state_snuqs = run_snuqs('qasm2/opaque/sxdg.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/sxdg.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_p(self):
        state_snuqs = run_snuqs('qasm2/opaque/p.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/p.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rx(self):
        state_snuqs = run_snuqs('qasm2/opaque/rx.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/rx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_ry(self):
        state_snuqs = run_snuqs('qasm2/opaque/ry.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/ry.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rz(self):
        state_snuqs = run_snuqs('qasm2/opaque/rz.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/rz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u0(self):
        state_snuqs = run_snuqs('qasm2/opaque/u0.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/u0.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u1(self):
        state_snuqs = run_snuqs('qasm2/opaque/u1.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/u1.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u2(self):
        state_snuqs = run_snuqs('qasm2/opaque/u2.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/u2.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u3(self):
        state_snuqs = run_snuqs('qasm2/opaque/u3.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/u3.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_u(self):
        state_snuqs = run_snuqs('qasm2/opaque/u.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/u.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cx(self):
        state_snuqs = run_snuqs('qasm2/opaque/cx.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cy(self):
        state_snuqs = run_snuqs('qasm2/opaque/cy.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cy.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cz(self):
        state_snuqs = run_snuqs('qasm2/opaque/cz.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_swap(self):
        state_snuqs = run_snuqs('qasm2/opaque/swap.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/swap.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_ch(self):
        state_snuqs = run_snuqs('qasm2/opaque/ch.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/ch.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_csx(self):
        state_snuqs = run_snuqs('qasm2/opaque/csx.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/csx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_crx(self):
        state_snuqs = run_snuqs('qasm2/opaque/crx.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/crx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cry(self):
        state_snuqs = run_snuqs('qasm2/opaque/cry.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cry.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_crz(self):
        state_snuqs = run_snuqs('qasm2/opaque/crz.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/crz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cp(self):
        state_snuqs = run_snuqs('qasm2/opaque/cp.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cp.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cu1(self):
        state_snuqs = run_snuqs('qasm2/opaque/cu1.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cu1.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rxx(self):
        state_snuqs = run_snuqs('qasm2/opaque/rxx.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/rxx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rzz(self):
        state_snuqs = run_snuqs('qasm2/opaque/rzz.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/rzz.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cu3(self):
        state_snuqs = run_snuqs('qasm2/opaque/cu3.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cu3.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cu(self):
        state_snuqs = run_snuqs('qasm2/opaque/cu.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cu.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_ccx(self):
        state_snuqs = run_snuqs('qasm2/opaque/ccx.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/ccx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_cswap(self):
        state_snuqs = run_snuqs('qasm2/opaque/cswap.qasm')
        state_qiskit = run_qiskit('qasm2/opaque/cswap.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()
