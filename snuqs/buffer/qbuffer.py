import numpy as np
from abc import ABC, abstractmethod

NUM_QMEMORY_MAX_QUBITS = 30
NUM_QBUFFER_STAGING_BUFFER_QUBITS = 22


class Qbuffer(ABC):
    @abstractmethod
    def __init__(self, nqubits: int):
        self.nqubits = nqubits

    @abstractmethod
    def __getitem__(self, key: int):
        pass

    @abstractmethod
    def __setitem__(self, key: int, val):
        pass


class Qmemory(Qbuffer):
    def __init__(self, nqubits: int):
        assert nqubits < NUM_QMEMORY_MAX_QUBITS

        super().__init__(nqubits)
        self.buf = np.zeros(2**nqubits, dtype=np.complex128)

    def __getitem__(self, key: int):
        return self.buf[key]

    def __setitem__(self, key: int, val):
        self.buf[key] = val


class Qstorage(Qbuffer):
    def __init__(self, nqubits: int):
        super()
        self.nqubits = nqubits
        self.buf = np.zeros(
            2**min(nqubits, NUM_QBUFFER_STAGING_BUFFER_QUBITS),
            dtype=np.complex128)

    def __getitem__(self, key):
        return self.buf[key]

    def __setitem__(self, key, val):
        self.buf[key] = val
