import numpy as np 
import ctypes

class Qreg:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.buf = np.zeros(2**nqubits, dtype=np.complex128)

    def __repr__(self):
        return f"Qreg {self.nqubits}\n"

    def __getitem__(self, key):
        return self.buf[key]

    def __setitem__(self, key, val):
        self.buf[key] = val

    def nqubits(self):
        return self.nqubits

    def numpy(self):
        return self.buf
