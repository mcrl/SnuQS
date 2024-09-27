import numpy as np

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit

# https://github.com/Infleqtion/client-superstaq/blob/main/supermarq-benchmarks/supermarq/benchmarks/ghz.py

def _fanout(*qubit_indices: int):
    if len(qubit_indices) >= 2:
        cutoff = len(qubit_indices) // 2
        yield qubit_indices[0], qubit_indices[cutoff]
        yield from _fanout(*qubit_indices[:cutoff])
        yield from _fanout(*qubit_indices[cutoff:])

class GHZ:
    """
    Represents the GHZ state preparation benchmark parameterized by the number of qubits n.

    Device performance is based on the Hellinger fidelity between the experimental and ideal
    probability distributions.
    """

    def __init__(self, num_qubits: int, method: str = "ladder") -> None:
        """
        Initialize a `GHZ` object.

        Args:
            num_qubits: Number of qubits in GHZ circuit.
            method: Circuit construction method to use. Must be "ladder", "star", or "logdepth". The
                "ladder" method uses a linear-depth CNOT ladder, appropriate for nearest-neighbor
                architectures. The "star" method is also linear depth, but with all CNOTs sharing
                the same control qubit. The "logdepth" method uses a log-depth CNOT fanout circuit.
        """

        if method not in ("ladder", "star", "logdepth"):
            raise ValueError(
                f"'{method}' is not a valid GHZ circuit construction method. Valid options are "
                "'ladder', 'star', and 'logdepth'."
            )
        self.num_qubits = num_qubits
        self.method = method

    def ghz_circuit(self) -> Circuit:
        """
        Generate an n-qubit GHZ circuit.

        Returns:
            A `Circuit`.
        """
        circuit = Circuit()
        circuit.h(0)

        if self.method == "ladder":
            for i in range(1, self.num_qubits):
                circuit.cnot(i - 1, i)

        elif self.method == "star":
            for i in range(1, self.num_qubits):
                circuit.cnot(0, i)

        else:
            for i, j in _fanout(*range(self.num_qubits)):
                circuit.cnot(i, j)

        for i in range(self.num_qubits):
            circuit.measure(i)
        return circuit

    def score(self, counts):
        r"""
        Compute the Hellinger fidelity between the experimental and ideal results.

        The ideal results are 50% probabilty of measuring the all-zero state and 50% probability
        of measuring the all-one state.

        The formula for the Hellinger fidelity between two distributions p and q is given by
        $(\sum_i{p_i q_i})^2$.

        Args:
            counts: A dictionary containing the measurement counts from circuit execution.

        Returns:
            Hellinger fidelity as a float.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.num_qubits: 0.5 for b in ["0", "1"]}
        total = sum(counts.values())
        device_dist = {bitstr: count / total for bitstr, count in counts.items()}

        fidelity = 0.0
        for bitstr in ideal_dist:
            p_i = ideal_dist.get(bitstr, 0)
            q_i = device_dist.get(bitstr, 0)
            fidelity += (np.sqrt(p_i) * np.sqrt(q_i))

        return fidelity**2
    
    def run(self, shots, backend=None):
        if not backend:
            sim = LocalSimulator()
        else:
            sim = LocalSimulator(backend=backend)

        circ = self.ghz_circuit()
        task = sim.run(circ, shots=shots)

        result = task.result().measurement_counts
        fidelity = self.score(result)
        
        print(f"Result: {result}")
        print(f"Hellinger fidelity: {fidelity}")
        return fidelity
