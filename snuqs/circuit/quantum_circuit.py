class QuantumCircuit:
    def __init__(self, name):
        self.name = name
        self.ops = []

    def __repr__(self):
        s = ""
        for op in self.ops:
            s += op.__repr__() + "\n"
        return s

    def prepend_op(self, op):
        self.ops = [op] + self.ops
    
    def push_op(self, op):
        self.ops.append(op)
