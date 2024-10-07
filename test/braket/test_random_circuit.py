import unittest
import numpy as np

from braket.devices import LocalSimulator
from braket.circuits import Circuit
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction

MIN_QUBIT = 20
MAX_QUBIT = 20
MAX_GATE = 100
NGATE_KIND = 31
NUM_ITER = 1000


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
            Gate.U,
            Gate.GPhase,
        ]

        self.gate_maps = [
            lambda x, y, z: Gate.I(),
            lambda x, y, z: Gate.H(),
            lambda x, y, z: Gate.X(),
            lambda x, y, z: Gate.Y(),
            lambda x, y, z: Gate.Z(),
            lambda x, y, z: Gate.CNot(),
            lambda x, y, z: Gate.CY(),
            lambda x, y, z: Gate.CZ(),
            lambda x, y, z: Gate.S(),
            lambda x, y, z: Gate.Si(),
            lambda x, y, z: Gate.T(),
            lambda x, y, z: Gate.Ti(),
            lambda x, y, z: Gate.V(),
            lambda x, y, z: Gate.Vi(),
            lambda x, y, z: Gate.PhaseShift(angle=x),
            lambda x, y, z: Gate.CPhaseShift(angle=x),
            lambda x, y, z: Gate.CPhaseShift00(angle=x),
            lambda x, y, z: Gate.CPhaseShift01(angle=x),
            lambda x, y, z: Gate.CPhaseShift10(angle=x),
            lambda x, y, z: Gate.Rx(angle=x),
            lambda x, y, z: Gate.Ry(angle=x),
            lambda x, y, z: Gate.Rz(angle=x),
            lambda x, y, z: Gate.Swap(),
            lambda x, y, z: Gate.ISwap(),
            lambda x, y, z: Gate.PSwap(angle=x),
            lambda x, y, z: Gate.XY(angle=x),
            lambda x, y, z: Gate.XX(angle=x),
            lambda x, y, z: Gate.YY(angle=x),
            lambda x, y, z: Gate.ZZ(angle=x),
            lambda x, y, z: Gate.CCNot(),
            lambda x, y, z: Gate.CSwap(),
            lambda x, y, z: Gate.U(angle_1=x,
                                   angle_2=y,
                                   angle_3=z),
            lambda x, y, z: Gate.GPhase(angle=x),
            # Unitary
        ]

    def get_qubits(self, count):
        qubits = set()
        while len(qubits) < count:
            qubits.add(np.random.randint(0, self.nqubits))
        return np.random.permutation(list(qubits))

    def get(self):
        idx = np.random.randint(0, len(self.gate_maps))
        param1 = np.random.rand() * np.pi * 2
        param2 = np.random.rand() * np.pi * 2
        param3 = np.random.rand() * np.pi * 2
        return Instruction(self.gate_maps[idx](param1, param2, param3), self.get_qubits(self.gates[idx].fixed_qubit_count()))


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
        circ = Circuit([Instruction(Gate.I(), [q]) for q in range(nqubits)])
        for _ in range(ngates):
            circ.add_instruction(RandomInstruction(nqubits).get())
        return circ

    def run_braket(self, circ):
        sim = LocalSimulator()
        task = sim.run(circ)
        return task

    def run_snuqs(self, circ):
        option = {
            'device': 'cuda',
            #'offload': 'cpu',
            # 'path': [ '/dev/nvme0n1', '/dev/nvme1n1', '/dev/nvme2n1', '/dev/nvme3n1', '/dev/nvme4n1', '/dev/nvme5n1', '/dev/nvme6n1', '/dev/nvme7n1', ],
        }
        sim = LocalSimulator(backend="snuqs")
        task = sim.run(circ, **option)
        return task

    def test_braket_snuqs(self):
        for i in range(NUM_ITER):
            circ = self.random_circuit()
            circ.state_vector()

            print(f"Running random circuit test #{i}... ")

            print("\tRunning braket")
            task_braket = self.run_braket(circ)
            print("\tRunning snuqs")
            task_snuqs = self.run_snuqs(circ)

            result_braket = task_braket.result().values
            result_snuqs = task_snuqs.result().values

            self.assertTrue(np.allclose(
                result_braket,
                result_snuqs,
            ))


if __name__ == '__main__':
    unittest.main()
