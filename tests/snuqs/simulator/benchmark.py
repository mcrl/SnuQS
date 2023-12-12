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


class BenchmarkTest(unittest.TestCase):
    def test_adder(self):
        state_snuqs = run_snuqs('qasm/benchmark/adder.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/adder.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_bigadder(self):
        state_snuqs = run_snuqs('qasm/benchmark/bigadder.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/bigadder.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_entangled_registers(self):
        state_snuqs = run_snuqs('qasm/benchmark/entangled_registers.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/entangled_registers.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_inverseqft1(self):
        state_snuqs = run_snuqs('qasm/benchmark/inverseqft1.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/inverseqft1.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_inverseqft2(self):
        state_snuqs = run_snuqs('qasm/benchmark/inverseqft2.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/inverseqft2.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_plaquette_check(self):
        state_snuqs = run_snuqs('qasm/benchmark/plaquette_check.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/plaquette_check.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_qec(self):
        state_snuqs = run_snuqs('qasm/benchmark/qec.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/qec.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_qft(self):
        state_snuqs = run_snuqs('qasm/benchmark/qft.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/qft.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_qpt(self):
        state_snuqs = run_snuqs('qasm/benchmark/qpt.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/qpt.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rb(self):
        state_snuqs = run_snuqs('qasm/benchmark/rb.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/rb.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_teleport(self):
        state_snuqs = run_snuqs('qasm/benchmark/teleport.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/teleport.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_teleportv2(self):
        state_snuqs = run_snuqs('qasm/benchmark/teleportv2.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/teleportv2.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_W(self):
        state_snuqs = run_snuqs('qasm/benchmark/W-state.qasm')
        state_qiskit = run_qiskit('qasm/benchmark/W-state.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()
