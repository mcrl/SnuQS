import unittest
import numpy as np

from braket.devices import LocalSimulator
from braket.circuits import Circuit
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction

MIN_QUBIT = 15
MAX_QUBIT = 20
MAX_GATE = 10
MAX_ITER = 10
NGATE_KIND = 31
NUM_ITER = 1000


# initial=0, other = CZ?_lastSingleGate
# 0=x_none, 1=x_X, 2=x_Y, 3=x_T
# 4=o_none, 5=o_X, 6=o_Y, 7=o_T

def RandomSingleQubitGate(state, idx):
    gates = [
        None,
        Gate.Rx(angle=3.14159 / 2),
        Gate.Ry(angle=3.14159 / 2),
        Gate.T,
    ]
    pos = [1, 2, 3]
    pos.remove(state[idx]-4)
    state[idx] = np.random.choice(pos)
    return gates[state[idx]]

def single_PT_circuit(nqubits, state, d):
    circ = Circuit()
    bias = [[0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1],
            [1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0],
            [-6, 0, -6, 0, -6, 0, 0, 6, 0, 6, 0, 6,
             0, -6, 0, -6, 0, -6, 6, 0, 6, 0, 6, 0],
            [0, -6, 0, -6, 0, -6, 6, 0, 6, 0, 6, 0,
             -6, 0, -6, 0, -6, 0, 0, 6, 0, 6, 0, 6],
            [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, 0],
            [6, 0, 6, 0, 6, 0, -6, 0, -6, 0, -6, 0,
             0, 6, 0, 6, 0, 6, 0, -6, 0, -6, 0, -6],
            [0, 6, 0, 6, 0, 6, 0, -6, 0, -6, 0, -6,
             6, 0, 6, 0, 6, 0, -6, 0, -6, 0, -6, 0]]
    rand_idx = d % 8  # np.random.randint(0, 8)
    period = len(bias[rand_idx])

    for q in range(0, nqubits):
        b = bias[rand_idx][q % period]
        if b > 0 and q+b < nqubits:
            circ.add_instruction(Instruction(Gate.CZ, [q, q+b]))
            if state[q] < 4:
                state[q] += 4
        elif b < 0 and q+b >= 0:
            if state[q] < 4:
                state[q] += 4
            pass
        else:
            if state[q] < 4:
                pass
            elif state[q] == 4:
                circ.add_instruction(Instruction(Gate.T, [q]))
                state[q] = 3
            else:  # state =5,6,7
                circ.add_instruction(Instruction(
                    RandomSingleQubitGate(state, q), [q]))
    return circ

def PT_circuit(nqubits, d):
    circ = Circuit([Instruction(Gate.H(), [q]) for q in range(nqubits)])
    state = [0 for _ in range(nqubits)]
    for _ in range(d):
        circ.add_circuit(single_PT_circuit(nqubits, state, _))
        print(circ)
    return circ


nqubits = np.random.randint(MIN_QUBIT, MAX_QUBIT+1)
d = MAX_ITER  # np.random.randint(1, MAX_ITER)
circ = PT_circuit(nqubits, d)

instructions = list(circ.instructions)
for e in instructions:
    print(e)
