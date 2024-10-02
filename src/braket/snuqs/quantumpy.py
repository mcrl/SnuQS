import numpy as np
from braket.snuqs._C import (
    Identity, Hadamard, PauliX, PauliY, PauliZ,
    CX, CY, CZ, S, Si, T, Ti, V, Vi,
    PhaseShift, CPhaseShift, CPhaseShift00, CPhaseShift01, CPhaseShift10,
    RotX, RotY, RotZ,
    Swap, ISwap, PSwap, XY, XX, YY, ZZ,
    CCNot, CSwap,
    U, GPhase,
)


class ndarray(np.ndarray):
    def __new__(cls, obj):
        arr = np.array(obj, copy=False).view(cls)
        arr.obj = obj
        return arr


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


def u(angle_1: float, angle_2: float, angle_3: float):
    return ndarray(U(angle_1, angle_2, angle_3))


def gphase(angle: float):
    return ndarray(GPhase(angle))
