from __future__ import annotations


class Qreg:
    def __init__(self, name: str, dim: int):
        if dim == 0:
            raise ValueError("Qreg of dim 0 is not allowed")

        self.name = name
        self.dim = dim
        self.qbits = [Qbit(self, i) for i in range(dim)]

    def __getitem__(self, key: int):
        return self.qbits[key]

    def __repr__(self):
        return f"{self.name}"


class Qbit(Qreg):
    def __init__(self, qreg: Qreg, index: int):
        self.qreg = qreg
        self.index = index
        self.dim = 1

    def __repr__(self):
        return f"{self.qreg.name}[{self.index}]"
