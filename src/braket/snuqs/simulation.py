import numpy as np
from braket.snuqs.operation import GateOperation


class Simulation:
    """
    This class tracks the evolution of a quantum system with `qubit_count` qubits.
    The state of system the evolves by application of `GateOperation`s using the `evolve()` method.
    """

    def __init__(self, qubit_count: int, shots: int):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as density matrix
                or expectation, are generated.
        """
        self._qubit_count = qubit_count
        self._shots = shots

    @property
    def qubit_count(self) -> int:
        """int: The number of qubits being simulated by the simulation."""
        return self._qubit_count

    @property
    def shots(self) -> int:
        """
        int: The number of samples to take from the simulation.

        0 means no samples are taken, and results that require sampling
        to calculate cannot be returned.
        """
        return self._shots

    def evolve(self, operations: list[GateOperation]) -> None:
        """Evolves the state of the simulation under the action of
        the specified gate operations.

        Args:
            operations (list[GateOperation]): Gate operations to apply for
                evolving the state of the simulation.

        Note:
            This method mutates the state of the simulation.
        """
        raise NotImplementedError("evolve has not been implemented.")


class StateVectorSimulation(Simulation):
    def __init__(self, qubit_count: int, shots: int):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as state vector
                or expectation, are generated.
        """

        super().__init__(qubit_count=qubit_count, shots=shots)
        initial_state = np.zeros(2**qubit_count, dtype=complex)
        initial_state[0] = 1
        self._state_vector = initial_state

    def _multiply_matrix(
            self,
            state: np.ndarray,
            matrix: np.ndarray,
            targets: tuple[int, ...],
    ) -> np.ndarray:
        gate_matrix = np.reshape(matrix, [2] * len(targets) * 2)
        axes = (
            np.arange(len(targets), 2 * len(targets)),
            targets,
        )
        product = np.tensordot(gate_matrix, state, axes=axes)

        # Axes given in `operation.targets` are in the first positions.
        unused_idxs = [idx for idx in range(len(state.shape)) if idx not in targets]
        permutation = list(targets) + unused_idxs
        # Invert the permutation to put the indices in the correct place
        inverse_permutation = np.argsort(permutation)
        return np.transpose(product, inverse_permutation)

    def evolve(self, operations: list[GateOperation]) -> None:
        self._state_vector = np.reshape(self._state_vector, [2] * self._qubit_count)
        for op in operations:
            self._state_vector = self._multiply_matrix(self._state_vector, op.matrix, op.targets)
        self._state_vector = np.reshape(self._state_vector, 2**self._qubit_count)