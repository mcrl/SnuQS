from snuqs.job import Job
from snuqs.result import Result
from snuqs.utils import logger

class BaseSimulator:
    def __init__(self):
        pass

    def _run(self, qasm, **kwargs):
        pass

    def qc_to_qasm(self, qc):
        import qiskit
        import cirq
        if type(qc) is qiskit.circuit.quantumcircuit.QuantumCircuit:
            return qc.qasm()
        elif type(qc) is cirq.circuits.circuit.Circuit:
            return cirq.qasm(qc)
        else:
            raise Exception(f"Not supported quantum circuit type {type(qc)}")

    def cast_result(self, qc, job):
        import qiskit
        import cirq
        if type(qc) is qiskit.circuit.quantumcircuit.QuantumCircuit:
            return job
        elif type(qc) is cirq.circuits.circuit.Circuit:
            return job.result()
        else:
            raise Exception(f"Not supported quantum circuit type {type(qc)}")

    def run(self, qc, **kwargs):
        job = self._run(self.qc_to_qasm(qc), **kwargs)
        return self.cast_result(qc, job)
