from abc import *
from snuqs.job import Job, QiskitJob
from snuqs.compilers import QASMCompiler

class BaseSimulator(metaclass=ABCMeta):
    def __init__(self):
        #self.qasm_compiler = QASMCompiler()
        pass

    @abstractmethod
    def _run(self, qasm, **kwargs):
        pass

    @staticmethod
    def cast(typ, job):
        import qiskit
        if isinstance(typ, str):
            return job
        elif isinstance(typ, qiskit.circuit.quantumcircuit.QuantumCircuit):
            return QiskitJob(job)
        else: 
            raise Exception(f"Not supported quantum circuit type {typ}")

    def run(self, qc, circ, qreg, creg, **kwargs):
        job = self._run(circ, qreg, creg, **kwargs)
        job.submit()
        return BaseSimulator.cast(qc, job)
