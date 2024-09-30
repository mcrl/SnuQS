import numpy as np
from braket.snuqs.operation import GateOperation
import braket.snuqs.quantumpy as qp


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

    def retrieve_samples(self) -> list[int]:
        """Retrieves samples of states from the state of the simulation,
        based on the probabilities.

        Returns:
            list[int]: List of states sampled according to their probabilities
            in the state. Each integer represents the decimal encoding of the
            corresponding computational basis state.
        """
        raise NotImplementedError("retrieve_samples has not been implemented.")

    @property
    def probabilities(self) -> np.ndarray:
        """np.ndarray: The probabilities of each computational basis state."""
        raise NotImplementedError("probabilities has not been implemented.")

    def evolve(self, operations: list[GateOperation], use_cuda: bool = False) -> None:
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
        initial_state = qp.state_vector(qubit_count)
        self._state_vector = initial_state

    def retrieve_samples(self) -> list[int]:
        return np.random.choice(len(self._state_vector), p=self.probabilities, size=self._shots)

    @property
    def state_vector(self) -> qp.ndarray:
        """
        qp.ndarray: The state vector specifying the current state of the simulation.

        Note:
            Mutating this array will mutate the state of the simulation.
        """
        return self._state_vector

    def evolve(self, operations: list[GateOperation], use_cuda: bool = False) -> None:
        self._state_vector = qp.evolve(
            self._state_vector, self._qubit_count, operations, use_cuda=use_cuda)

    @property
    def probabilities(self) -> qp.ndarray:
        """
        qp.ndarray: The probabilities of each computational basis state of the current state
            vector of the simulation.
        """
        return np.abs(self.state_vector) ** 2
