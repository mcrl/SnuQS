import snuqs._C
from snuqs.circuit import Circuit


class StatevectorSimulator:
    def __init__(self):
        self.sim = snuqs._C.StatevectorSimulator()

    def _circuit_transform(self, circ: Circuit):
        # TODO: How to interface qops with C/C++
        # 1. How to deal with Qreg and Creg
        #   -- 1 qreg, many creg
        # 2. How to represent qops
        # 3. How to represent customggate, in particular
        print(circ)

    def run(self, circ: Circuit):
        circ = self._circuit_transform(circ)
        raise NotImplementedError("Not implemeted yet")
        self.sim.run(circ)
