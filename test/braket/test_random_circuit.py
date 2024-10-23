import unittest
import numpy as np

from braket.devices import LocalSimulator
from braket.circuits import Circuit
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction

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
    def random_circuit(self, nqubits, ngates):
        circ = Circuit([Instruction(Gate.H(), [q]) for q in range(nqubits)])
        for _ in range(ngates):
            circ.add_instruction(RandomInstruction(nqubits).get())
        return circ

    def run_braket(self, circ):
        sim = LocalSimulator()
        task = sim.run(circ)
        return task

    def run_snuqs(self, circ, **option):
        sim = LocalSimulator(backend="snuqs")
        task = sim.run(circ, **option)
        return task

    def run_benchmark(self, nqubits, ngates, **option):
        circ = self.random_circuit(nqubits, ngates)
        circ.state_vector()

        print("\tRunning snuqs")
        task_snuqs = self.run_snuqs(circ, **option)
        result_snuqs = task_snuqs.result().values
        print("\t\t=> Done")

        print("\tRunning braket")
        task_braket = self.run_braket(circ)
        result_braket = task_braket.result().values
        print("\t\t=> Done")

        print(result_snuqs[0][2:])
        print(result_braket[0][2:])
        self.assertTrue(np.allclose(
            result_snuqs[0][2:],
            result_braket[0][2:],
        ))

#    def test_1_braket_snuqs_cpu(self):
#        print("Testing Braket-SnuQS CPU")
#        self.run_benchmark(15, 2000, accelerator='cpu')
#
#    def test_2_braket_snuqs_cuda(self):
#        print("Testing Braket-SnuQS CUDA")
#        self.run_benchmark(15, 2000, accelerator='cuda')
#
#    def test_3_braket_snuqs_hybrid(self):
#        print("Testing Braket-SnuQS Hybrid")
#        self.run_benchmark(15, 2000,
#                           qubit_count_cpu=15,
#                           qubit_count_cuda=14,
#                           qubit_count_slice=13,
#                           accelerator='hybrid')
#
#    def test_4_braket_snuqs_cpu_offload_cpu(self):
#        print("Testing Braket-SnuQS CPU-Offload CPU")
#        self.run_benchmark(15, 2000,
#                           qubit_count_cpu=15,
#                           qubit_count_cuda=14,
#                           qubit_count_slice=13,
#                           accelerator='cpu',
#                           offload='cpu')
#
#    def test_5_braket_snuqs_cpu_offload_cuda(self):
#        print("Testing Braket-SnuQS CPU-Offload CUDA")
#        self.run_benchmark(15, 2000,
#                           qubit_count_cpu=15,
#                           qubit_count_cuda=14,
#                           qubit_count_slice=13,
#                           accelerator='cuda',
#                           offload='cpu')
#
#    def test_6_braket_snuqs_cpu_offload_hybrid(self):
#        print("Testing Braket-SnuQS CPU-Offload Hybrid")
#        self.run_benchmark(15, 1000,
#                           qubit_count_cpu=15,
#                           qubit_count_cuda=14,
#                           qubit_count_slice=13,
#                           accelerator='hybrid',
#                           offload='cpu')
#
    def test_7_braket_snuqs_storage_offload_cpu(self):
        print("Testing Braket-SnuQS Stroage-Offload Hybrid")
        for _ in range(100):
            self.run_benchmark(16, 0,
                               qubit_count_cpu=15,
                               qubit_count_cuda=14,
                               qubit_count_slice=13,
                               accelerator='cpu',
                               offload='storage',
                               path=['/dev/nvme0n1p1', '/dev/nvme1n1p1', '/dev/nvme2n1p1', '/dev/nvme3n1p1',
                                     '/dev/nvme4n1p1', '/dev/nvme5n1p1', '/dev/nvme6n1p1', '/dev/nvme7n1p1'],
                               block_count=2**12
                               )

#    def test_7_braket_snuqs_storage_offload_cuda(self):
#        print("Testing Braket-SnuQS Stroage-Offload Hybrid")
#        self.run_benchmark(35, 5, accelerator='cuda', offload='storage',
#                           path=['/dev/nvme0n1p1', '/dev/nvme1n1p1', '/dev/nvme2n1p1', '/dev/nvme3n1p1',
#                                 '/dev/nvme4n1p1', '/dev/nvme5n1p1', '/dev/nvme6n1p1', '/dev/nvme7n1p1'],
#                           count=2**38,
#                           block_count=2**34
#                           )
#
#    def test_7_braket_snuqs_storage_offload_hybrid(self):
#        print("Testing Braket-SnuQS Stroage-Offload Hybrid")
#        self.run_benchmark(35, 5, accelerator='hybrid', offload='storage',
#                           path=['/dev/nvme0n1p1', '/dev/nvme1n1p1', '/dev/nvme2n1p1', '/dev/nvme3n1p1',
#                                 '/dev/nvme4n1p1', '/dev/nvme5n1p1', '/dev/nvme6n1p1', '/dev/nvme7n1p1'],
#                           count=2**38,
#                           block_count=2**34
#                           )


if __name__ == '__main__':
    unittest.main()
