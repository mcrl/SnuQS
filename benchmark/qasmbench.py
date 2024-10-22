import re
from collections.abc import Iterator
from __util import hellinger_fidelity

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit

from qiskit import QuantumCircuit
from qiskit_braket_provider import BraketLocalBackend

from QASMBench.interface.qiskit import QASMBenchmark

import warnings
warnings.filterwarnings("ignore", message="The Qiskit circuit contains barrier instructions that are ignored.")

class QASMBench:
    def __init__(self, circ_name: str, category: str = "small", backend: str = None) -> None:
        transpile_args = {}
        self.bm = QASMBenchmark("QASMBench", category, remove_final_measurements=False, do_transpile=False, **transpile_args)
        self.backend = backend
        self.circ_name = circ_name

        match = re.search(r'(\d+)$', circ_name)
        self.num_qubits = int(match.group(1))
    
    def circuit(self) -> QuantumCircuit:
        return self.bm.get(self.circ_name)

    def score(self, counts):
        ideal_dist = {b * self.num_qubits: 0.5 for b in ["0", "1"]}
        total = sum(counts.values())
        device_dist = {bitstr: count / total for bitstr, count in counts.items()}

        return hellinger_fidelity(ideal_dist, device_dist)
    
    def run(self, shots):
        if not self.backend:
            sim = BraketLocalBackend(name="default")
        else:
            sim = BraketLocalBackend(name=self.backend)

        circ = self.circuit()
        result = sim.run(circ, shots=shots).result()

        counts = result.get_counts()
        fidelity = self.score(counts)
        
        print(f"Result: {counts}")
        print(f"Hellinger fidelity: {fidelity}")
        return fidelity