from enum import Enum

class QopType(Enum):
    Init = 0
    Fini = 1
    Cond = 2
    Measure = 3
    Reset = 4
    Barrier = 5
    UGate = 6
    CXGate = 7

class Qop:
    def __init__(self, typ: QopType, qubits: int):
        self.typ = typ
        self.qubits = qubits

    def __repr__(self):
        t = self.typ
        q = self.qubits
        return f"{t} {q}"

class Barrier(Qop):
    def __init__(self, typ: QopType, qubits: int):
        super().__init__(typ, qubits)

class Reset(Qop):
    def __init__(self, typ: QopType, qubits: int):
        super().__init__(typ, qubits)

class Measure(Qop):
    def __init__(self, typ: QopType, qubits: int, bits: int):
        super().__init__(typ, qubits)
        self.bits = bits

    def __repr__(self):
        t = self.typ
        q = self.qubits
        c = f"-> {self.bits}" if self.bits else ""
        return f"{t} {q} {c}"

class Cond(Qop):
    def __init__(self, typ: QopType, base: int, limit: int, val: int, op: int):
        super().__init__(typ, op.qubits)
        self.base = base
        self.limit = limit
        self.val = val
        self.op = op

    def eval(self, creg):
        val = 0
        k = 0
        for i in range(self.base, self.limit):
            if creg[i] == 1:
                val += 2 ** k
            k += 1
        return val == self.val

    def __repr__(self):
        return f"IF [{self.base}:{self.limit}] == {self.val} {self.op}"

class Init(Qop):
    def __init__(self, typ, qubits):
        super().__init__(typ, qubits)

class Fini(Qop):
    def __init__(self, typ, qubits):
        super().__init__(typ, qubits)

class Gate(Qop):
    def __init__(self, typ, qubits, params=[]):
        super().__init__(typ, qubits)
        self.params = params

    def __repr__(self):
        t = self.typ
        p = f"({self.params})" if self.params else ""
        q = self.qubits
        return f"{t}{p} {q}"
