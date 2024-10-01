from collections.abc import Iterator
from __util import hellinger_fidelity

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit

class PhaseCode:
    """
    Creates a circuit for syndrome measurement in a phase-flip error correcting code.

    Args:
        num_data: The number of data qubits.
        num_rounds: The number of measurement rounds.
        phase_state: A list of zeros and ones denoting the state to initialize each data
            qubit to. Currently just + or - states. 0 -> +, 1 -> -.

    Returns:
        A `Circuit` for the phase-flip error correcting code.
    """

    def __init__(self, num_data_qubits: int, num_rounds: int, phase_state: list[int]) -> None:
        if len(phase_state) != num_data_qubits:
            raise ValueError("The length of `phase_state` must match the number of data qubits.")
        if not isinstance(phase_state, list):
            raise ValueError("`phase_state` must be a list[int].")
        else:
            if not set(phase_state).issubset({0, 1}):
                raise ValueError("Entries of `phase_state` must be 0, 1 integers.")
        self.num_data_qubits = num_data_qubits
        self.num_rounds = num_rounds
        self.phase_state = phase_state

    def _measurement_round(
        self, num_qubits: int, round_idx: int
    ) -> Iterator[Circuit]:
        """
        Generates operations for a single measurement round in Braket using yield, including reset.

        Args:
            num_qubits: The total number of qubits used.
            round_idx: The round index for naming the measurement key.

        Yields:
            A Braket `Circuit` with operations for a measurement round.
        """
        qubits = list(range(num_qubits))
        ancilla_qubits = qubits[1::2]

        yield Circuit().h(qubits)
        for qq in range(1, num_qubits, 2):
            yield Circuit().cz(qq - 1, qq)
        for qq in range(1, num_qubits - 1, 2):
            yield Circuit().cz(qq + 1, qq)
        yield Circuit().h(qubits)
        yield Circuit().measure(ancilla_qubits)
        # WARN: Braket doesn't support reset
        # yield Circuit().reset(ancilla_qubits)

    def pc_circuit(self) -> Circuit:
        """
        Generates phase code circuit.

        Returns:
            A `Circuit`.
        """
        num_qubits = 2 * self.num_data_qubits - 1
        circuit = Circuit()

        # Initialize the data qubits
        for i in range(self.num_data_qubits):
            if self.phase_state[i] == 1:
                circuit.x(2 * i)
            circuit.h(2 * i)

        # Apply measurement rounds
        for i in range(self.num_rounds):
            circuit += self._measurement_round(num_qubits, i)

        # Measure final outcomes in X basis to produce single product state
        for i in range(self.num_data_qubits):
            circuit += Circuit().h(2 * i)

        circuit.measure(range(num_qubits))

        return circuit

    def _get_ideal_dist(self) -> dict[str, float]:
        """
        Return the ideal probability distribution of `self.circuit()`.

        Since the initial states of the data qubits are either |+> or |->, and we measure the final
        state in the X-basis, the final state is a single product state in the noiseless case.

        Returns:
            Dictionary with measurement results as keys and probabilites as values.
        """
        ancilla_state, final_state = "", ""
        for i in range(self.num_data_qubits - 1):
            # parity checks
            ancilla_state += str((self.phase_state[i] + self.phase_state[i + 1]) % 2)
            final_state += str(self.phase_state[i]) + "0"
        else:
            final_state += str(self.phase_state[-1])

        ideal_bitstring = [ancilla_state] * self.num_rounds + [final_state]
        return {"".join(ideal_bitstring): 1.0}

    def score(self, counts: dict[str, float]) -> float:
        """
        Compute benchmark score.

        Device performance is given by the Hellinger fidelity between the experimental results and
        the ideal distribution. The ideal is known based on the `phase_state` parameter.

        Args:
            counts: Dictionary containing the measurement counts from running `self.circuit()`.

        Returns:
            A float with the computed score.
        """
        ideal_dist = self._get_ideal_dist()
        total_shots = sum(counts.values())
        device_dist = {bitstr: shots / total_shots for bitstr, shots in counts.items()}
        return hellinger_fidelity(ideal_dist, device_dist)

    def run(self, shots: int, backend=None):
        if not backend:
            sim = LocalSimulator()
        else:
            sim = LocalSimulator(backend=backend)

        circ = self.pc_circuit()
        task = sim.run(circ, shots=shots)

        result = task.result().measurement_counts
        fidelity = self.score(result)
        
        print(f"Result: {result}")
        print(f"Phase Code fidelity: {fidelity}")
        return fidelity