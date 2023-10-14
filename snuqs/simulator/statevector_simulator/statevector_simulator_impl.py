import numpy as np
from abc import *
from snuqs.quantum_circuit import QuantumCircuit, QopType

class StatevectorSimulatorImpl:
    @abstractmethod
    def run(self, circ: QuantumCircuit, **kwargs):
        for op in circ.ops:
            self.run_op(op, circ.qreg, circ.creg)

    def run_op(self, op, qreg, creg):
        if op.typ == QopType.Barrier:
            # Do nothing
            pass
        elif op.typ == QopType.Reset:
            self.op_reset(qreg, op.qubits)
        elif op.typ == QopType.Measure:
            self.op_measure(qreg, creg, op.qubits, op.bits)
        elif op.typ == QopType.Cond:
            if op.eval(creg):
                self.run_op(op.op, qreg, creg)
        elif op.typ == QopType.UGate:
            self.op_ugate(qreg, op.qubits, op.params)
        elif op.typ == QopType.CXGate:
            self.op_cxgate(qreg, op.qubits)

    def op_reset(self, qreg, qubits):
        nqubits = qreg.nqubits
        target = qubits[0]
        st = 2**target
        for i in range(0, 2**nqubits, 2*st):
            for j in range(i, i+st):
                amp = np.sqrt(np.square(np.abs(qreg[j])) + np.square(np.abs(qreg[j+st])))
                qreg[j] = amp
                qreg[j+st] = 0

    def op_measure(self, qreg, creg, qubits, bits):
        nqubits = qreg.nqubits
        target = qubits[0]
        tbit = bits[0]
        st = 2**target

        threshold = np.random.rand()
        prob = 0
        collapse_to_zero = False
        for i in range(0, 2**nqubits, 2*st):
            for j in range(i, i+st):
                prob += np.square(np.abs(qreg[j]))

        if threshold <= prob:
            collapse_to_zero = True

        for i in range(0, 2**nqubits, 2*st):
            for j in range(i, i+st):
                x = qreg[j]
                y = qreg[j+st]
                prob_amp = np.sqrt(np.square(np.abs(x)) + np.square(np.abs(y)))

                if np.isclose(prob_amp, 0):
                    qreg[j] = qreg[j+st] = 0
                else:
                    if collapse_to_zero:
                        qreg[j] = qreg[j] / np.sqrt(prob)
                        qreg[j+st] = 0
                        creg[tbit] = 0
                    else:
                        qreg[j] = 0
                        qreg[j+st] = qreg[j+st] / np.sqrt(1-prob)
                        creg[tbit] = 1

    def op_ugate(self, qreg, qubits, params):
        nqubits = qreg.nqubits
        target = qubits[0]
        st = 2**target

        theta, phi, lmbda = params
        a00 = np.cos(theta/2)
        a01 = -np.exp(1j*lmbda) * np.sin(theta/2)
        a10 = np.exp(1j*phi) * np.sin(theta/2)
        a11 = np.exp(1j*(phi+lmbda)) * np.cos(theta/2)

        for i in range(0, 2**nqubits, 2*st):
            for j in range(i, i+st):
                v0 = a00 * qreg[j] + a01 * qreg[j+st]
                v1 = a10 * qreg[j] + a11 * qreg[j+st]
                qreg[j] = v0
                qreg[j+st] = v1

    def op_cxgate(self, qreg, qubits):
        nqubits = qreg.nqubits
        ctrl, target = qubits

        t0, t1 = (ctrl, target) if ctrl < target else (target, ctrl)

        st0 = 2**t0 
        st1 = 2**t1 

        cst = 2**ctrl
        tst = 2**target

        for i in range(0, 2**nqubits, 2*st1):
            for j in range(i, i+st1, 2*st0):
                for k in range(j, j+st0):
                    qreg[k+cst], qreg[k+cst+tst] = qreg[k+cst+tst], qreg[k+cst]

