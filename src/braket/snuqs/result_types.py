from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Any
from braket.ir import jaqcd
from braket.circuits import ResultType
from braket.snuqs.quantumpy import qp


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
