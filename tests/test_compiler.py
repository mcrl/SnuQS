import unittest

import random
import math
from snuqs.quantum_circuit import QgateDictionary
from snuqs.compilers import QASMCompiler

MAX_QUBITS = 8


def random_max_qubits():
    return random.randint(5, MAX_QUBITS)


def random_distinct_qubits(num_qubits, max_qubits):
    qubits = set()
    while len(qubits) < num_qubits:
        qubits.add(random.randint(0, max_qubits-1))
    return list(qubits)


def random_params(num_params):
    return [4 * math.pi * random.random() for _ in range(num_params)]


def random_qasm():
    max_qubits = random_max_qubits()
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[%d];
creg c[%d];""" % (max_qubits, max_qubits)

    for name, (num_qubits, num_params) in QgateDictionary.items():
        qubits = [str(v) for v in random_distinct_qubits(num_qubits, max_qubits)]
        qasm += f"{name.lower()} "
        if num_params > 0:
            params = [str(v) for v in random_params(num_params)]
            qasm += f"({','.join(params)}) "
        qasm += f"{','.join(qubits)}\n"
    return qasm


class CompilerTest(unittest.TestCase):
    def test_random_circuit(self):
        qasm = random_qasm()
        qasm_compiler = QASMCompiler()
        circ = qasm_compiler.compile(qasm)
        self.assertIsNotNone(circ, 'Random OpenQASM Compilation Test')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(CompilerTest('test_random_circuit'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
