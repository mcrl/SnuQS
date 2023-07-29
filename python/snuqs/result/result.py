class Result:
    def __init__(self, qreg, creg):
        self.qreg = qreg
        self.creg = creg

    def __repr__(self):
        return "Result"

    def get_statevector(self):
        return self.qreg.numpy()
