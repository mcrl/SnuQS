import numpy as np
from braket.snuqs._C import StateVector, Identity, Hadamard
from typing import Optional

ndarray = np.ndarray
linalg = np.linalg


def statevector(qubit_count: int, init: Optional[bool] = True):
    sv = np.array(StateVector(qubit_count), copy=False)
    if init:
        sv.fill(0)
        sv[0] = 1
    return sv


def eye(*args, **kwargs):
    return np.eye(*args, **kwargs)


def reshape(*args, **kwargs):
    return np.reshape(*args, **kwargs)


def arange(*args, **kwargs):
    return np.arange(*args, **kwargs)


def tensordot(*args, **kwargs):
    return np.tensordot(*args, **kwargs)


def argsort(*args, **kwargs):
    return np.argsort(*args, **kwargs)


def transpose(*args, **kwargs):
    return np.transpose(*args, **kwargs)


def allclose(*args, **kwargs):
    return np.allclose(*args, **kwargs)


def identity():
    return np.array(Identity(), copy=False)


def hadamard():
    return np.array(Hadamard(), copy=False)
