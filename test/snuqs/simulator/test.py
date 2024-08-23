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


class TestTest(unittest.TestCase):
    def test_test(self):
        state_qiskit = run_qiskit('test.qasm')
        print(state_qiskit)

if __name__ == '__main__':
    unittest.main()
