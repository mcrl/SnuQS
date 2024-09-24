import numpy as np
from braket.snuqs._C import StateVector
from braket.snuqs._C import (
    Identity, Hadamard, PauliX, PauliY, PauliZ,
    CX, CY, CZ, S, Si, T, Ti, V, Vi,
    PhaseShift, CPhaseShift, CPhaseShift00, CPhaseShift01, CPhaseShift10,
    RotX, RotY, RotZ,
    Swap, ISwap, PSwap, XY, XX, YY, ZZ,
    CCNot, CSwap
)

from typing import Optional

linalg = np.linalg


class ndarray(np.ndarray):
    def __new__(cls, buffer):
        obj = np.array(buffer, copy=False).view(cls)
        obj.buffer = buffer
        return obj


def state_vector(qubit_count: int, init: Optional[bool] = True):
    arr = ndarray(StateVector(qubit_count))
    if init:
        arr.fill(0)
        arr[0] = 1
    return arr


def multiply_matrix(
        state: ndarray,
        matrix: ndarray,
        targets: tuple[int, ...],
) -> ndarray:
    gate_matrix = np.reshape(matrix, [2] * len(targets) * 2)
    axes = (
        np.arange(len(targets), 2 * len(targets)),
        targets,
    )
    product = np.tensordot(gate_matrix, state, axes=axes)

    # Axes given in `operation.targets` are in the first positions.
    unused_idxs = [idx for idx in range(len(state.shape)) if idx not in targets]
    permutation = list(targets) + unused_idxs
    # Invert the permutation to put the indices in the correct place
    inverse_permutation = np.argsort(permutation)
    return np.transpose(product, inverse_permutation)


def evolve(state_vector: ndarray, qubit_count: int, operations) -> None:
    state_vector = np.reshape(state_vector, [2] * qubit_count)
    supported_gates = [
        Identity,
        # Hadamard, PauliX, PauliY, PauliZ,
        # CX, CY, CZ, S, Si, T, Ti, V, Vi,
        # PhaseShift, CPhaseShift, CPhaseShift00, CPhaseShift01, CPhaseShift10,
        # RotX, RotY, RotZ,
        # Swap, ISwap, PSwap, XY, XX, YY, ZZ,
        # CCNot, CSwap
    ]
    for op in operations:
        if any(isinstance(op.matrix.buffer, g) for g in supported_gates):
            op.matrix.buffer.evolve(state_vector, op.targets)
        else:
            state_vector=multiply_matrix(state_vector, op.matrix, op.targets)
    return np.reshape(state_vector, 2**qubit_count)


def eye(*args, **kwargs):
    return np.eye(*args, **kwargs)


def allclose(*args, **kwargs):
    return np.allclose(*args, **kwargs)


def diag(*args, **kwargs):
    return np.diag(*args, **kwargs)


def identity():
    return ndarray(Identity())


def hadamard():
    return ndarray(Hadamard())


def paulix():
    return ndarray(PauliX())


def pauliy():
    return ndarray(PauliY())


def pauliz():
    return ndarray(PauliZ())


def cx():
    return ndarray(CX())


def cy():
    return ndarray(CY())


def cz():
    return ndarray(CZ())


def s():
    return ndarray(S())


def si():
    return ndarray(Si())


def t():
    return ndarray(T())


def ti():
    return ndarray(Ti())


def v():
    return ndarray(V())


def vi():
    return ndarray(Vi())


def phase_shift(angle: float):
    return ndarray(PhaseShift(angle))


def cphase_shift(angle: float):
    return ndarray(CPhaseShift(angle))


def cphase_shift00(angle: float):
    return ndarray(CPhaseShift00(angle))


def cphase_shift01(angle: float):
    return ndarray(CPhaseShift01(angle))


def cphase_shift10(angle: float):
    return ndarray(CPhaseShift10(angle))


def rot_x(angle: float):
    return ndarray(RotX(angle))


def rot_y(angle: float):
    return ndarray(RotY(angle))


def rot_z(angle: float):
    return ndarray(RotZ(angle))


def swap():
    return ndarray(Swap())


def iswap():
    return ndarray(ISwap())


def pswap(angle: float):
    return ndarray(PSwap(angle))


def xy(angle: float):
    return ndarray(XY(angle))


def xx(angle: float):
    return ndarray(XX(angle))


def yy(angle: float):
    return ndarray(YY(angle))


def zz(angle: float):
    return ndarray(ZZ(angle))


def ccnot():
    return ndarray(CCNot())


def cswap():
    return ndarray(CSwap())
