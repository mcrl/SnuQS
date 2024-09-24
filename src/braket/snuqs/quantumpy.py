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

from braket.snuqs._C import tensordot

from typing import Optional

ndarray = np.ndarray
linalg = np.linalg


def statevector(qubit_count: int, init: Optional[bool] = True):
    sv = np.array(StateVector(qubit_count), copy=False)
    if init:
        sv.fill(0)
        sv[0] = 1
    return sv


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
    # FIXME
    state = tensordot(gate_matrix, state, axes)
    product = np.tensordot(gate_matrix, state, axes=axes)

    # Axes given in `operation.targets` are in the first positions.
    unused_idxs = [idx for idx in range(len(state.shape)) if idx not in targets]
    permutation = list(targets) + unused_idxs
    # Invert the permutation to put the indices in the correct place
    inverse_permutation = np.argsort(permutation)
    return np.transpose(product, inverse_permutation)


def evolve(state_vector: ndarray, qubit_count: int, operations) -> None:
    for op in operations:
        multiply_matrix(state_vector, op.matrix, op.targets)
    #state_vector = np.reshape(state_vector, [2] * qubit_count)
    # for op in operations:
    #state_vector = multiply_matrix(state_vector, op.matrix, op.targets)
    # return np.reshape(state_vector, 2**qubit_count)


def eye(*args, **kwargs):
    return np.eye(*args, **kwargs)


def allclose(*args, **kwargs):
    return np.allclose(*args, **kwargs)


def diag(*args, **kwargs):
    return np.diag(*args, **kwargs)


def identity():
    return np.array(Identity(), copy=False)


def hadamard():
    return np.array(Hadamard(), copy=False)


def paulix():
    return np.array(PauliX(), copy=False)


def pauliy():
    return np.array(PauliY(), copy=False)


def pauliz():
    return np.array(PauliZ(), copy=False)


def cx():
    return np.array(CX(), copy=False)


def cy():
    return np.array(CY(), copy=False)


def cz():
    return np.array(CZ(), copy=False)


def s():
    return np.array(S(), copy=False)


def si():
    return np.array(Si(), copy=False)


def t():
    return np.array(T(), copy=False)


def ti():
    return np.array(Ti(), copy=False)


def v():
    return np.array(V(), copy=False)


def vi():
    return np.array(Vi(), copy=False)


def phase_shift(angle: float):
    return np.array(PhaseShift(angle), copy=False)


def cphase_shift(angle: float):
    return np.array(CPhaseShift(angle), copy=False)


def cphase_shift00(angle: float):
    return np.array(CPhaseShift00(angle), copy=False)


def cphase_shift01(angle: float):
    return np.array(CPhaseShift01(angle), copy=False)


def cphase_shift10(angle: float):
    return np.array(CPhaseShift10(angle), copy=False)


def rot_x(angle: float):
    return np.array(RotX(angle), copy=False)


def rot_y(angle: float):
    return np.array(RotY(angle), copy=False)


def rot_z(angle: float):
    return np.array(RotZ(angle), copy=False)


def swap():
    return np.array(Swap(), copy=False)


def iswap():
    return np.array(ISwap(), copy=False)


def pswap(angle: float):
    return np.array(PSwap(angle), copy=False)


def xy(angle: float):
    return np.array(XY(angle), copy=False)


def xx(angle: float):
    return np.array(XX(angle), copy=False)


def yy(angle: float):
    return np.array(YY(angle), copy=False)


def zz(angle: float):
    return np.array(ZZ(angle), copy=False)


def ccnot():
    return np.array(CCNot(), copy=False)


def cswap():
    return np.array(CSwap(), copy=False)
