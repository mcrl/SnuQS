import unittest
from snuqs import StatevectorSimulator

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit.providers.aer import Aer
from qiskit.circuit.library.standard_gates import RYGate, MCXGate, RXGate, SGate, XGate

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


class BenchmarkTest(unittest.TestCase):
    def test_init(self):
        qubit_n_H = 6
        A = np.random.rand(4, 4)
        b_vect = np.random.rand(4)
        N = len(A)
        k = max(np.linalg.eig(A)[0])/min(abs(np.linalg.eig(A)[0]))
        eip = 0.001
        b = int(k**2 * np.log2(k/eip))
        j0 = int(np.sqrt(b*np.log2(4*b/eip)))
        print("j0:", j0)
        j0 = 511
        qubit_n = int(np.log2(N))
        qubit_ancila = 1
        k_lo = 0
        k0_lo = qubit_n
        j_lo = qubit_n + 1
        j0_lo = 2*qubit_n + 1
        ancila_lo = qubit_n + 1 + qubit_n + 1
        total_qubit = qubit_n + 1 + qubit_n + 1 + qubit_ancila
        qcKr = QuantumRegister(2*qubit_n + 2)
        qcK = QuantumCircuit(qcKr)
        for i in range(qubit_n):
            qcK.h(qcKr[i])
        for Kj in range(N):
            for Kk in range(N):
                qcK_target = []
                qcK_target.append(j0_lo)
                qcK_target.extend(range(j_lo, j0_lo))
                qcK_target.extend(range(k_lo, k0_lo))
                qcK_target.append(k0_lo)

                Kk_list = np.zeros(qubit_n)
                Kk_bin = [int(i) for i in bin(Kk)[2:]]
                Kk_list[qubit_n-len(Kk_bin):] = Kk_bin
                Kk_list = [int(i) for i in Kk_list]
                Kk_list = ''.join(str(i) for i in Kk_list)

                Kj_list = np.zeros(qubit_n)
                Kj_bin = [int(i) for i in bin(Kj)[2:]]
                Kj_list[qubit_n-len(Kj_bin):] = Kj_bin
                Kj_list = [int(i) for i in Kj_list]
                Kj_list = ''.join(str(i) for i in Kj_list)

                ctrl_target = Kk_list + Kj_list+'0'

                if A[Kj, Kk] >= 0:
                    theta = 2 * np.arccos((A[Kj, Kk])**0.5)
                    Tmaker = RYGate(theta).control(
                        2*qubit_n+1, ctrl_state=ctrl_target)
                else:

                    phase_min = XGate().control(2*qubit_n+1, ctrl_state=ctrl_target)
                    qcK.append(phase_min, qcK_target)

                    phase_min2 = SGate().control(2*qubit_n+1, ctrl_state=ctrl_target)
                    qcK.append(phase_min2, qcK_target)

                    phase_min3 = XGate().control(2*qubit_n+1, ctrl_state=ctrl_target)
                    qcK.append(phase_min3, qcK_target)

                    theta = 2 * np.arccos((-A[Kj, Kk])**0.5)
                    Tmaker = RXGate(theta).control(
                        2*qubit_n+1, ctrl_state=ctrl_target)
                qcK.append(Tmaker, qcK_target)
        K_circ = qcK
        qcSr = QuantumRegister(2*qubit_n + 2)
        qcS = QuantumCircuit(qcSr)
        for i in range(qubit_n + 1):
            qcS.swap(qcSr[i], qcSr[i+qubit_n+1])

        # S_circ = qcS.to_instruction()
        S_circ = qcS
        qc = QuantumCircuit(total_qubit)

        init = np.random.rand(2 ** total_qubit) + 1j *  np.random.rand(2 ** total_qubit)
        init = init / np.sqrt(np.sum(np.square(np.abs(init))))
        qc.initialize(init)

        qc.append(K_circ.inverse(), range(2*qubit_n + 2))

        qc.x(ancila_lo)
        qc.h(ancila_lo)

        qc.x(ancila_lo)
        mcxg = MCXGate(qubit_n + 2, ctrl_state=('0' * (qubit_n + 2)))
        mxxg_target = []
        mxxg_target.extend(range(qubit_n))
        mxxg_target.extend([k0_lo, j0_lo, ancila_lo])
        qc.append(mcxg, mxxg_target)
        qc.h(ancila_lo)
        qc.x(ancila_lo)

        qc.append(K_circ, range(2*qubit_n + 2))
        qc.append(S_circ, range(2*qubit_n + 2))

        state_snuqs = run_snuqs(qc)
        state_qiskit = run_qiskit(qc)
        phase = state_qiskit[0] / state_snuqs[0]
        for x, y in zip(state_snuqs, state_qiskit):
            self.assertAlmostEqual(x, y * phase)


if __name__ == '__main__':
    unittest.main()
