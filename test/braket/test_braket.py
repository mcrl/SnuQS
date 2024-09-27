import unittest
import numpy as np

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction

MIN_QUBIT = 5
MAX_QUBIT = 15
MAX_GATE = 200
NGATE_KIND = 31
NUM_ITER = 30000


class RandomInstruction:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.gates = [
            Gate.I,
            Gate.H,
            Gate.X,
            Gate.Y,
            Gate.Z,
            Gate.CNot,
            Gate.CY,
            Gate.CZ,
            Gate.S,
            Gate.Si,
            Gate.T,
            Gate.Ti,
            Gate.V,
            Gate.Vi,
            Gate.PhaseShift,
            Gate.CPhaseShift,
            Gate.CPhaseShift00,
            Gate.CPhaseShift01,
            Gate.CPhaseShift10,
            Gate.Rx,
            Gate.Ry,
            Gate.Rz,
            Gate.Swap,
            Gate.ISwap,
            Gate.PSwap,
            Gate.XY,
            Gate.XX,
            Gate.YY,
            Gate.ZZ,
            Gate.CCNot,
            Gate.CSwap,
        ]

        self.gate_maps = [
            lambda x: Gate.I(),
            lambda x: Gate.H(),
            lambda x: Gate.X(),
            lambda x: Gate.Y(),
            lambda x: Gate.Z(),
            lambda x: Gate.CNot(),
            lambda x: Gate.CY(),
            lambda x: Gate.CZ(),
            lambda x: Gate.S(),
            lambda x: Gate.Si(),
            lambda x: Gate.T(),
            lambda x: Gate.Ti(),
            lambda x: Gate.V(),
            lambda x: Gate.Vi(),
            lambda x: Gate.PhaseShift(angle=x),
            lambda x: Gate.CPhaseShift(angle=x),
            lambda x: Gate.CPhaseShift00(angle=x),
            lambda x: Gate.CPhaseShift01(angle=x),
            lambda x: Gate.CPhaseShift10(angle=x),
            lambda x: Gate.Rx(angle=x),
            lambda x: Gate.Ry(angle=x),
            lambda x: Gate.Rz(angle=x),
            lambda x: Gate.Swap(),
            lambda x: Gate.ISwap(),
            lambda x: Gate.PSwap(angle=x),
            lambda x: Gate.XY(angle=x),
            lambda x: Gate.XX(angle=x),
            lambda x: Gate.YY(angle=x),
            lambda x: Gate.ZZ(angle=x),
            lambda x: Gate.CCNot(),
            lambda x: Gate.CSwap(),
            # Unitary
            # U
            # GPhase
        ]

    def get_qubits(self, count):
        qubits = set()
        while len(qubits) < count:
            qubits.add(np.random.randint(0, self.nqubits))
        return np.random.permutation(list(qubits))

    def get(self):
        idx = np.random.randint(0, len(self.gate_maps))
        param = np.random.rand() * np.pi * 2
        return Instruction(self.gate_maps[idx](param), self.get_qubits(self.gates[idx].fixed_qubit_count()))


def repeat(times):
    def repeat_helper(f):
        def call_helper(*args):
            for i in range(0, times):
                f(*args)
        return call_helper

    return repeat_helper


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
        option = {
            'path': [
                '/dev/nvme0n1',
                '/dev/nvme1n1',
                '/dev/nvme2n1',
                '/dev/nvme3n1',
                '/dev/nvme4n1',
                '/dev/nvme5n1',
                '/dev/nvme6n1',
                '/dev/nvme7n1',
            ],
        }
        sim = LocalSimulator(backend="snuqs")
        task = sim.run(circ, **option)
        return task

    @repeat(NUM_ITER)
    def test_braket_snuqs(self):
        circ = self.random_circuit()
        circ.state_vector()

        print(f"Running random circuit test...")
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
