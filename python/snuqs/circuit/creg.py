import numpy as np

class Creg:
    def __init__(self, num_bits):
        self.nbits = num_bits
        self.buf = np.zeros(num_bits, dtype=np.int32)
        self.buf[:] = -1

    def __repr__(self):
        return f"Creg[{self.nbits}] = {self.buf}"

    def __getitem__(self, key):
        return self.buf[key]

    def __setitem__(self, key, val):
        self.buf[key] = val

    def num_bits(self):
        return self.num_bits

    def numpy(self):
        return self.buf
