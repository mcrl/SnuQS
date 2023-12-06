from abc import ABCMeta, abstractmethod
from typing import List
from .arg import Qarg, Carg
from .parameter import Parameter


class Qop(metaclass=ABCMeta):
    def __init__(self, qbits: List[Qarg], params: List[Parameter] = []):
        if len(qbits) == 0:
            raise ValueError("List of qbits must not be empty")

        self.qbits = qbits
        self.params = params


class Barrier(Qop):
    def __init__(self, qbits: List[Qarg]):
        super().__init__(qbits)

    def __repr__(self):
        q = ", ".join(str(qb) for qb in self.qbits)
        return f"BARRIER {q}"


class Reset(Qop):
    def __init__(self, qbits: List[Qarg]):
        super().__init__(qbits)

    def __repr__(self):
        q = ", ".join(str(qb) for qb in self.qbits)
        return f"RESET {q}"


class Measure(Qop):
    def __init__(self, qbits: List[Qarg], cbits: List[Carg]):
        super().__init__(qbits)

        if len(qbits) != len(cbits):
            raise ValueError(
                f"Number of qbits must be equal to number of cbits: {len(qbits)} qbits, but {len(cbits)} cbits"
            )

        for q, c in zip(qbits, cbits):
            if q.dim() != c.dim():
                raise ValueError(
                    f"Dimension of qbits and cbits mismatch: {q.dim()} != {c.dim()}"
                )

        self.cbits = cbits

    def __repr__(self):
        q = ", ".join(str(qb) for qb in self.qbits)
        c = ", ".join(str(cb) for cb in self.cbits)
        return f"MEASURE {q} -> {c}"


class Cond(Qop):
    def __init__(self, op: Qop, creg: Carg, val: int):
        super().__init__(op.qbits)
        self.op = op
        self.creg = creg
        self.val = val

    def __repr__(self):
        return f"IF ({self.creg} == {self.val}) {self.op}"


class Custom(Qop):
    def __init__(self, name: str, qops: List[Qop], qbits: List[Qarg], params: List[Parameter] = []):
        super().__init__(qbits, params)
        self.name = name
        self.qops = qops

    def __repr__(self):
        q = ", ".join(str(qb) for qb in self.qbits)
        p = ", ".join(str(pm) for pm in self.params)
        if len(self.params) > 0:
            p = f"({p})"
        rp = f"CUSTOM {self.name}{p} {q}"
        rp += " {\n"
        for qop in self.qops:
            rp += f"  {qop.__repr__()}\n"
        rp += "}"
        return rp


class Qgate(Qop, metaclass=ABCMeta):
    def __init__(self, qbits: List[Qarg], params: List[Parameter] = []):
        super().__init__(qbits, params)

        if len(qbits) != self.num_target_qubits():
            raise ValueError(
                f"Number of targets does not match: {self.num_target_qubits()} required, but given {len(qbits)}")

        if len(params) != self.num_params():
            raise ValueError(
                f"Number of parameters does not match: {self.num_params()} required, but given {len(params)}")

    def __repr__(self):
        q = ", ".join(str(qb) for qb in self.qbits)
        p = ", ".join(str(pm) for pm in self.params)
        if len(self.params) > 0:
            p = f"({p})"
        return f"{self.__class__.__name__}{p} {q}"

    @abstractmethod
    def num_target_qubits(self):
        pass

    @abstractmethod
    def num_params(self):
        pass


class ID(Qgate):
    """
    ID gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class X(Qgate):
    """
    X gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class Y(Qgate):
    """
    Y gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class Z(Qgate):
    """
    Z gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class H(Qgate):
    """
    H gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class S(Qgate):
    """
    S gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class SDG(Qgate):
    """
    SDG gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class T(Qgate):
    """
    T gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class TDG(Qgate):
    """
    TDG gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class SX(Qgate):
    """
    SX gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class SXDG(Qgate):
    """
    SXDG gate
    ----------------
    num_target_qubits: 1
    num_params: 0
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 0


class P(Qgate):
    """
    P gate
    ----------------
    num_target_qubits: 1
    num_params: 1
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 1


class RX(Qgate):
    """
    RX gate
    ----------------
    num_target_qubits: 1
    num_params: 1
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 1


class RY(Qgate):
    """
    RY gate
    ----------------
    num_target_qubits: 1
    num_params: 1
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 1


class RZ(Qgate):
    """
    RZ gate
    ----------------
    num_target_qubits: 1
    num_params: 1
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 1


class U0(Qgate):
    """
    U0 gate
    ----------------
    num_target_qubits: 1
    num_params: 1
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 1


class U1(Qgate):
    """
    U1 gate
    ----------------
    num_target_qubits: 1
    num_params: 1
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 1


class U2(Qgate):
    """
    U2 gate
    ----------------
    num_target_qubits: 1
    num_params: 2
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 2


class U3(Qgate):
    """
    U3 gate
    ----------------
    num_target_qubits: 1
    num_params: 3
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 3


class U(Qgate):
    """
    U gate
    ----------------
    num_target_qubits: 1
    num_params: 3
    """

    def num_target_qubits(self):
        return 1

    def num_params(self):
        return 3


class CX(Qgate):
    """
    CX gate
    ----------------
    num_target_qubits: 2
    num_params: 0
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 0


class CZ(Qgate):
    """
    CZ gate
    ----------------
    num_target_qubits: 2
    num_params: 0
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 0


class CY(Qgate):
    """
    CY gate
    ----------------
    num_target_qubits: 2
    num_params: 0
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 0


class SWAP(Qgate):
    """
    SWAP gate
    ----------------
    num_target_qubits: 2
    num_params: 0
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 0


class CH(Qgate):
    """
    CH gate
    ----------------
    num_target_qubits: 2
    num_params: 0
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 0


class CSX(Qgate):
    """
    CSX gate
    ----------------
    num_target_qubits: 2
    num_params: 0
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 0


class CRX(Qgate):
    """
    CRX gate
    ----------------
    num_target_qubits: 2
    num_params: 1
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 1


class CRY(Qgate):
    """
    CRY gate
    ----------------
    num_target_qubits: 2
    num_params: 1
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 1


class CRZ(Qgate):
    """
    CRZ gate
    ----------------
    num_target_qubits: 2
    num_params: 1
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 1


class CU1(Qgate):
    """
    CU1 gate
    ----------------
    num_target_qubits: 2
    num_params: 1
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 1


class CP(Qgate):
    """
    CP gate
    ----------------
    num_target_qubits: 2
    num_params: 1
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 1


class RXX(Qgate):
    """
    RXX gate
    ----------------
    num_target_qubits: 2
    num_params: 1
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 1


class RZZ(Qgate):
    """
    RZZ gate
    ----------------
    num_target_qubits: 2
    num_params: 1
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 1


class CU3(Qgate):
    """
    CU3 gate
    ----------------
    num_target_qubits: 2
    num_params: 3
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 3


class CU(Qgate):
    """
    CU gate
    ----------------
    num_target_qubits: 2
    num_params: 4
    """

    def num_target_qubits(self):
        return 2

    def num_params(self):
        return 4


class CCX(Qgate):
    """
    CCX gate
    ----------------
    num_target_qubits: 3
    num_params: 0
    """

    def num_target_qubits(self):
        return 3

    def num_params(self):
        return 0


class CSWAP(Qgate):
    """
    CSWAP gate
    ----------------
    num_target_qubits: 3
    num_params: 0
    """

    def num_target_qubits(self):
        return 3

    def num_params(self):
        return 0


class RCCX(Qgate):
    """
    RCCX gate
    ----------------
    num_target_qubits: 3
    num_params: 0
    """

    def num_target_qubits(self):
        return 3

    def num_params(self):
        return 0


class RC3X(Qgate):
    """
    RC3X gate
    ----------------
    num_target_qubits: 4
    num_params: 0
    """

    def num_target_qubits(self):
        return 4

    def num_params(self):
        return 0


class C3X(Qgate):
    """
    C3X gate
    ----------------
    num_target_qubits: 4
    num_params: 0
    """

    def num_target_qubits(self):
        return 4

    def num_params(self):
        return 0


class C3SQRTX(Qgate):
    """
    C3SQRTX gate
    ----------------
    num_target_qubits: 4
    num_params: 0
    """

    def num_target_qubits(self):
        return 4

    def num_params(self):
        return 0


class C4X(Qgate):
    """
    C4X gate
    ----------------
    num_target_qubits: 5
    num_params: 0
    """

    def num_target_qubits(self):
        return 5

    def num_params(self):
        return 0
