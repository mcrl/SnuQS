import unittest
from snuqs import QasmCompiler, StatevectorSimulator

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.providers.aer import Aer

import random
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


class LargeTest(unittest.TestCase):
#    def test_q28_g300(self):
#        state_snuqs = run_snuqs('qasm/large/q28_g300.qasm')
#        state_qiskit = run_qiskit('qasm/large/q28_g300.qasm')
#        random_indices = [random.randint(
#            0, len(state_qiskit)-1) for _ in range(1000)]
#        for i in random_indices:
#            x, y = state_snuqs[i], state_qiskit[i]
#            self.assertAlmostEqual(x, y)
#
#    def test_q30_g300(self):
#        state_snuqs = run_snuqs('qasm/large/q30_g300.qasm')
#        state_qiskit = run_qiskit('qasm/large/q30_g300.qasm')
#        random_indices = [random.randint(
#            0, len(state_qiskit)-1) for _ in range(1000)]
#        for i in random_indices:
#            x, y = state_snuqs[i], state_qiskit[i]
#            self.assertAlmostEqual(x, y)

    def test_q32_g300(self):
        state_snuqs = run_snuqs('qasm/large/q32_g300.qasm')
        state_qiskit = run_qiskit('qasm/large/q32_g300.qasm')
        random_indices = [random.randint(
            0, len(state_qiskit)-1) for _ in range(1000)]
        for i in random_indices:
            x, y = state_snuqs[i], state_qiskit[i]
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()
