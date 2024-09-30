from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Any, Union, Optional
from braket.ir import jaqcd
from braket.circuits import ResultType
import braket.snuqs.quantumpy as qp
from braket.snuqs.simulation import Simulation, StateVectorSimulation
from braket.snuqs.observables import Observable


def from_braket_result_type(result_type) -> ResultType:
    """Creates a `ResultType` corresponding to the given Braket instruction.

    Args:
        result_type: Result type for a circuit specified using the `braket.ir.jacqd` format.

    Returns:
        ResultType: Instance of specific `ResultType` corresponding to the type of result_type

    Raises:
        NotImplementedError: If no concrete `ResultType` class has been registered
            for the Braket instruction type
    """
    return _from_braket_result_type(result_type)


@singledispatch
def _from_braket_result_type(result_type):
    raise NotImplementedError(f"Result type {result_type} not recognized")


class ResultType(ABC):
    """
    An abstract class that when implemented defines a calculation on a
    quantum state simulation.

    Note:
        All result types are calculated exactly, instead of approximated from samples.
        Sampled results are returned from `Simulation.retrieve_samples`, which can be processed by,
        for example, the Amazon Braket SDK.
    """

    @abstractmethod
    def calculate(self, simulation: Simulation) -> Any:
        # Return type of Any due to lack of sum type support in Python
        """Calculate a result from the given quantum state vector simulation.

        Args:
            simulation (Simulation): The simulation to use in the calculation.

        Returns:
            Any: The result of the calculation.
        """


class TargetedResultType(ResultType, ABC):
    """
    Holds an observable that may target qubits.
    """

    def __init__(self, targets: Optional[list[int]] = None):
        """
        Args:
            targets (list[int], optional): The target qubits of the result type.
                If None, no specific qubits are targeted.
        """
        self._targets = targets

    @property
    def targets(self) -> Optional[tuple[int, ...]]:
        """tuple[int], optional: The target qubits of the result type, if any."""
        return self._targets


class ObservableResultType(TargetedResultType, ABC):
    """
    Holds an observable to perform a calculation in conjunction with a state.
    """

    def __init__(self, observable: Observable):
        """
        Args:
            observable (Observable): The observable for which the desired result is calculated
        """
        super().__init__(observable.measured_qubits)
        self._observable = observable

    @property
    def observable(self):
        """Observable: The observable for which the desired result is calculated."""
        return self._observable

    @property
    def targets(self) -> Optional[tuple[int, ...]]:
        return self._observable.measured_qubits

    def calculate(self, simulation: Simulation) -> Union[float, list[float]]:
        """Calculates the result type using the underlying observable.

        Returns a real number if the observable has defined targets,
        or a list of real numbers, one for the result type on each target,
        if the observable has no target.

        Args:
            simulation (Simulation): The simulation to use in the calculation.

        Returns:
            Union[float, list[float]]: The value of the result type;
            will be a real due to self-adjointness of observable.
        """
        if self._observable.measured_qubits:
            return self._calculate_single_quantity(simulation, self._observable)
        return [
            self._calculate_single_quantity(simulation, self._observable.fix_qubit(qubit))
            for qubit in range(simulation.qubit_count)
        ]

    @staticmethod
    @abstractmethod
    def _calculate_single_quantity(simulation: Simulation, observable: Observable) -> float:
        """Calculates a single real value of the result type.

        Args:
            simulation (Simulation): The simulation to use in the calculation.
            observable (Observable): The observable used to calculate the result type.

        Returns:
            float: The value of the result type.
        """


class StateVector(ResultType):
    """
    Simply returns the given state vector.
    """

    def calculate(self, simulation: StateVectorSimulation) -> qp.ndarray:
        """Return the given state vector of the simulation.

        Args:
            simulation (StateVectorSimulation): The simulation whose state vector will be returned

        Returns:
            qp.ndarray: The state vector (before observables) of the simulation
        """
        return simulation.state_vector


@_from_braket_result_type.register
def _(_: jaqcd.StateVector):
    return StateVector()


def _from_braket_observable(
    ir_observable: list[Union[str, list[list[list[float]]]]], ir_targets: Optional[list[int]] = None
) -> Observable:
    targets = list(ir_targets) if ir_targets else None
    if len(ir_observable) == 1:
        return _from_single_observable(ir_observable[0], targets)
    else:
        observable = TensorProduct(
            [_from_single_observable(factor, targets, is_factor=True) for factor in ir_observable]
        )
        if targets:
            raise ValueError(
                f"Found {len(targets)} more target qubits than the tensor product acts on"
            )
        return observable
