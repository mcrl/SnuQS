import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import fractional_matrix_power


class Operation(ABC):
    """
    Encapsulates an operation acting on a set of target qubits.
    """

    @property
    @abstractmethod
    def targets(self) -> tuple[int, ...]:
        """tuple[int, ...]: The indices of the qubits the operation applies to.

        Note: For an index to be a target of an observable, the observable must have a nontrivial
        (i.e. non-identity) action on that index. For example, a tensor product observable with a
        Z factor on qubit j acts trivially on j, so j would not be a target. This does not apply to
        gate operations.
        """


class GateOperation(Operation, ABC):
    """
    Encapsulates a unitary quantum gate operation acting on
    a set of target qubits.
    """

    def __init__(self, targets, *params, ctrl_modifiers=(), power=1):
        self._targets = tuple(targets)
        self._ctrl_modifiers = ctrl_modifiers
        self._power = power

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    @abstractmethod
    def _base_matrix(self) -> np.ndarray:
        """np.ndarray: The matrix representation of the operation."""

    @property
    def matrix(self) -> np.ndarray:
        unitary = self._base_matrix
        if int(self._power) == self._power:
            unitary = np.linalg.matrix_power(unitary, int(self._power))
        else:
            unitary = fractional_matrix_power(unitary, self._power)
        return unitary

    def __eq__(self, other):
        possible_parameters = "_angle", "_angle_1", "_angle_2"
        return self.targets == other.targets and all(
            getattr(self, param, None) == getattr(other, param, None)
            for param in possible_parameters
        )
