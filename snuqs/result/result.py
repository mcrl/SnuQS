from snuqs.circuit import Circuit

from dataclasses import dataclass


@dataclass
class Result:
    circ: Circuit

    def __repr__(self):
        return "Result"

    def get_statevector(self):
        return self.circ.qreg.numpy()
