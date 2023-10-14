from snuqs import QuantumCircuit 
class Result:
    circ: QuantumCircuit

    def __repr__(self):
        return "Result"

    def get_statevector(self):
        return self.circ.qreg.numpy()
