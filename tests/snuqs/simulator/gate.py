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


class GateTest(unittest.TestCase):
    def test_rccx(self):
        state_snuqs = run_snuqs('qasm2/gate/rccx.qasm')
        state_qiskit = run_qiskit('qasm2/gate/rccx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_rc3x(self):
        state_snuqs = run_snuqs('qasm2/gate/rc3x.qasm')
        state_qiskit = run_qiskit('qasm2/gate/rc3x.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_c3x(self):
        state_snuqs = run_snuqs('qasm2/gate/c3x.qasm')
        state_qiskit = run_qiskit('qasm2/gate/c3x.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_c3sqrtx(self):
        state_snuqs = run_snuqs('qasm2/gate/c3sqrtx.qasm')
        state_qiskit = run_qiskit('qasm2/gate/c3sqrtx.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)

    def test_c4x(self):
        state_snuqs = run_snuqs('qasm2/gate/c4x.qasm')
        state_qiskit = run_qiskit('qasm2/gate/c4x.qasm')
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y)



if __name__ == '__main__':
    unittest.main()
