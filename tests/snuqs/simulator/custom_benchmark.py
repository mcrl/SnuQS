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


class CustomBenchmarkTest(unittest.TestCase):
    def test_init7(self):
        state_snuqs = run_snuqs('qasm/custom_benchmark/init7.qasm')
#        state_qiskit = run_qiskit('qasm/custom_benchmark/init7.qasm')
#        for x, y in zip(state_snuqs, state_qiskit):
#            self.assertAlmostEqual(x, y)

#    def test_test5(self):
#        state_snuqs = run_snuqs('qasm/custom_benchmark/test5.qasm')
#        state_qiskit = run_qiskit('qasm/custom_benchmark/test5.qasm')
#        for x, y in zip(state_snuqs, state_qiskit):
#            self.assertAlmostEqual(x, y)

#    def test_test7(self):
#        state_snuqs = run_snuqs('qasm/custom_benchmark/test7.qasm')
#        state_qiskit = run_qiskit('qasm/custom_benchmark/test7.qasm')
#        for x, y in zip(state_snuqs, state_qiskit):
#            self.assertAlmostEqual(x, y)

#    def test_test9(self):
#        state_snuqs = run_snuqs('qasm/custom_benchmark/test9.qasm')
#        state_qiskit = run_qiskit('qasm/custom_benchmark/test9.qasm')
#        for x, y in zip(state_snuqs, state_qiskit):
#            self.assertAlmostEqual(x, y)
#
#    def test_test11(self):
#        state_snuqs = run_snuqs('qasm/custom_benchmark/test11.qasm')
#        state_qiskit = run_qiskit('qasm/custom_benchmark/test11.qasm')
#        for x, y in zip(state_snuqs, state_qiskit):
#            self.assertAlmostEqual(x, y)
#
#    def test_test13(self):
#        state_snuqs = run_snuqs('qasm/custom_benchmark/test13.qasm')
#        state_qiskit = run_qiskit('qasm/custom_benchmark/test13.qasm')
#        for x, y in zip(state_snuqs, state_qiskit):
#            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()
