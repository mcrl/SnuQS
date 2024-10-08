from collections.abc import Iterator
from __util import hellinger_fidelity

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit

from qiskit import QuantumCircuit
from qiskit_braket_provider import BraketLocalBackend

from mqt.bench import get_benchmark

import warnings
warnings.filterwarnings("ignore", message="The Qiskit circuit contains barrier instructions that are ignored.")

class MQTBench:
    def __init__(self, benchmark_name: str, level: str, circuit_size: int = None,
                 benchmark_instance_name: str = None, provider_name: str = 'ibm',
                 backend: str = None) -> None:
        self.num_qubits = circuit_size
        self.backend = backend

        self.circuit = get_benchmark(benchmark_name, level, circuit_size, benchmark_instance_name,
                                     provider_name)
 
    def circuit(self) -> QuantumCircuit:
        return self.circuit

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

        circ = self.circuit
        result = sim.run(circ, shots=shots).result()

        counts = result.get_counts()
        fidelity = self.score(counts)
        
        print(f"Result: {counts}")
        print(f"Hellinger fidelity: {fidelity}")
        return fidelity