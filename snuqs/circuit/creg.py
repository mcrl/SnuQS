from __future__ import annotations


class Creg:
    def __init__(self, name: str, dim: int):
        if dim == 0:
            raise ValueError("Creg of dim 0 is not allowed")

        self.name = name
        self.dim = dim
        self.cbits = [Cbit(self, i) for i in range(dim)]

    def __getitem__(self, key: int):
        return self.cbits[key]

    def __repr__(self):
        return f"{self.name}"


class Cbit:
    def __init__(self, creg: Creg, index: int):
        self.creg = creg
        self.index = index
        self.value = 0
        self.dim = 1

    def __repr__(self):
        return f"{self.creg.name}[{self.index}]"
