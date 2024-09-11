import unittest
import numpy as np

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction

MIN_QUBIT = 4
MAX_QUBIT = 20
MAX_GATE = 100
NGATE_KIND = 31


class RandomInstruction:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.gates = [
            Gate.CCNot,
            Gate.CPhaseShift10,
            Gate.I,
            Gate.Ry,
            Gate.T,
            Gate.X,
            Gate.Z,
            Gate.CNot,
            Gate.CSwap,
            Gate.ISwap,
            Gate.Rz,
            Gate.Ti,
            Gate.XX,
            Gate.ZZ,
            Gate.CPhaseShift,
            Gate.CY,
            Gate.PSwap,
            Gate.S,
            Gate.XY,
            Gate.CPhaseShift00,
            Gate.CZ,
            Gate.PhaseShift,
            Gate.Si,
            Gate.V,
            Gate.Y,
            Gate.CPhaseShift01,
            Gate.H,
            Gate.Rx,
            Gate.Swap,
            Gate.Vi,
            Gate.YY,
        ]

        self.gate_maps = [
            lambda x: Gate.CCNot(),
            lambda x: Gate.CPhaseShift10(angle=x),
            lambda x: Gate.I(),
            lambda x: Gate.Ry(angle=x),
            lambda x: Gate.T(),
            lambda x: Gate.X(),
            lambda x: Gate.Z(),
            lambda x: Gate.CNot(),
            lambda x: Gate.CSwap(),
            lambda x: Gate.ISwap(),
            lambda x: Gate.Rz(angle=x),
            lambda x: Gate.Ti(),
            lambda x: Gate.XX(angle=x),
            lambda x: Gate.ZZ(angle=x),
            lambda x: Gate.CPhaseShift(angle=x),
            lambda x: Gate.CY(),
            lambda x: Gate.PSwap(angle=x),
            lambda x: Gate.S(),
            lambda x: Gate.XY(angle=x),
            lambda x: Gate.CPhaseShift00(angle=x),
            lambda x: Gate.CZ(),
            lambda x: Gate.PhaseShift(angle=x),
            lambda x: Gate.Si(),
            lambda x: Gate.V(),
            lambda x: Gate.Y(),
            lambda x: Gate.CPhaseShift01(angle=x),
            lambda x: Gate.H(),
            lambda x: Gate.Rx(angle=x),
            lambda x: Gate.Swap(),
            lambda x: Gate.Vi(),
            lambda x: Gate.YY(angle=x),
        ]

    def get_qubits(self, count):
        qubits = set()
        while len(qubits) < count:
            qubits.add(np.random.randint(0, self.nqubits))
        return qubits

    def get(self):
        idx = np.random.randint(0, len(self.gates))
        param = np.random.rand() * np.pi * 2
        print(self.gates[idx].fixed_qubit_count())
        return Instruction(self.gate_maps[idx](param), self.get_qubits(self.gates[idx].fixed_qubit_count()))


class BraketTest(unittest.TestCase):
    def random_circuit(self):
        nqubits = np.random.randint(MIN_QUBIT, MAX_QUBIT+1)
        ngates = np.random.randint(1, MAX_GATE+1)
        print("nqubits: ", nqubits)
        circ = Circuit([RandomInstruction(nqubits).get() for _ in range(ngates)])
        return circ

    def run_braket(self, circ):
        sim = LocalSimulator()
        task = sim.run(circ)
        return task

    def run_snuqs(self, circ):
        sim = LocalSimulator(backend="snuqs")
        task = sim.run(circ)
        return task

    def test_braket_snuqs(self):
        for i in range(10000):
            circ = self.random_circuit()
            circ.state_vector()

            print(f"#{i} Running random circuit test...")
            print(circ)

            task_braket = self.run_braket(circ)
            task_snuqs = self.run_snuqs(circ)

            print(task_braket.result().values)
            print(task_snuqs.result().values)
            self.assertTrue(np.allclose(
                task_braket.result().values,
                task_snuqs.result().values
            ))


if __name__ == '__main__':
    unittest.main()
