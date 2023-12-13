import unittest
from snuqs import StatevectorSimulator

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit.providers.aer import Aer
from qiskit.circuit.library.standard_gates import RYGate, MCXGate, RXGate, SGate, XGate
import qiskit

import warnings


def run_snuqs(circ):
    sim = StatevectorSimulator()
    result = sim.run(circ)
    result.wait()
    state = result.get_statevector()
    return np.array(state, copy=False)


def run_qiskit(circ):
    circ.save_statevector()
    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(circ, simulator)
    result = simulator.run(circ).result()
    statevector = result.get_statevector(circ)
    return statevector.data


class MyInit(qiskit.circuit.Gate):
    def __init__(self, name, num_qubits, params):
        self.name = name
        self.num_qubits = num_qubits
        self.params = params
        self._num_clbits = 0
        self._condition = None

    def validate_parameter(self, parameter):
        return True


class BenchmarkTest(unittest.TestCase):
    def test_init(self):
        total_qubit = 16
        qc = QuantumCircuit(total_qubit)

        np.random.seed(123)
        # init = np.random.rand(2 ** total_qubit)
        init = np.random.rand(2 ** total_qubit) + 1j * \
            np.random.rand(2 ** total_qubit)
        init = init / np.sqrt(np.sum(np.square(np.abs(init))))
        qc.initialize(init)

        state_snuqs = run_snuqs(qc)
        state_qiskit = run_qiskit(qc)
        phase = state_snuqs[0] / state_qiskit[0]
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y * phase)


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(10**7)
    unittest.main()
