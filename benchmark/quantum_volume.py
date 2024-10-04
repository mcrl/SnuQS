from __util import get_ideal_counts

import numpy as np
import scipy.stats

from collections.abc import Mapping
from typing import Optional, Union

from braket.circuits import Circuit, Gate, Instruction
from braket.aws import AwsDevice
from braket.devices import LocalSimulator

# https://github.com/Qiskit/qiskit/blob/stable/1.2/qiskit/circuit/library/quantum_volume.py


class QuantumVolume:
    def __init__(self, num_qubits: int, depth: int = None, seed: Optional[Union[int, np.random.Generator]] = None, classical_permutation: bool = True):
        self.num_qubits = num_qubits
        self.depth = depth or num_qubits
        self.width = num_qubits // 2

        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        if not seed:
            seed = getattr(getattr(self.rng.bit_generator, "seed_seq", None), "entropy", None)

        self.classical_permutation = classical_permutation
        self.seed = seed or np.random.randint(0, 1000)
        self.rng = np.random.default_rng(self.seed)

    def circuit(self) -> Circuit:
        circuit = Circuit()

        unitaries = scipy.stats.unitary_group.rvs(
            4, self.depth * self.width, self.rng).reshape(self.depth, self.width, 4, 4)

        for row in unitaries:
            perm = self.rng.permutation(self.num_qubits)
            if self.classical_permutation:
                for w, unitary in enumerate(row):
                    circuit.unitary(matrix=unitary, targets=[perm[2 * w], perm[2 * w + 1]])
            else:
                # Braket doesn't supports permutation gate, implement with swap gate?
                pass

        circuit.measure(range(self.num_qubits))
        return circuit

    def score(self, heavy_output_count, total_shots, num_circuits) -> float:
        heavy_output_freq = heavy_output_count / total_shots

        conf_lower = heavy_output_freq - 2 * \
            np.sqrt((heavy_output_freq * (1 - heavy_output_freq)) / num_circuits)
        return conf_lower, conf_lower > 2 / 3

    def run(self, shots: int, num_circuits: int = 100) -> float:
        if not self.backend:
            sim = LocalSimulator()
        else:
            sim = LocalSimulator(backend=self.backend)

        total_shots = 0
        heavy_output_count = 0

        for _ in range(num_circuits):
            circ = self.circuit()
            task = sim.run(circ, shots=shots)

            ideal_counts = get_ideal_counts(circ, self.backend)
            counts = task.result().measurement_counts

            sorted_probs = sorted(ideal_counts.values())
            median_prob = np.median(sorted_probs)

            heavy_outputs = {bitstring for bitstring,
                             prob in ideal_counts.items() if prob > median_prob}

            heavy_output_count += sum(counts.get(bitstring, 0) for bitstring in heavy_outputs)
            total_shots += sum(counts.values())

        # Run circuit
        conf_lower, result = self.score(heavy_output_count, total_shots, num_circuits)

        print(f"Result: {result}")
        print(f"Conf_lower: {conf_lower}")
        return conf_lower
