import numpy as np
from abc import ABC, abstractmethod

from braket.snuqs._C.result_types import StateVector
from braket.snuqs.subcircuit import Subcircuit
from braket.snuqs._C.functionals import apply
from braket.snuqs._C.functionals import initialize_basis_z, initialize_zero
from braket.snuqs._C import DeviceType
from braket.snuqs.types import AcceleratorType, OffloadType


class Simulation(ABC):
    """
    This class tracks the evolution of a quantum system with `qubit_count` qubits.
    The state of system the evolves by application of `Subcircuit`s using the `evolve()` method.
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

    @abstractmethod
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
    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """np.ndarray: The probabilities of each computational basis state."""
        raise NotImplementedError("probabilities has not been implemented.")

    @abstractmethod
    def evolve(self, subcircuit: Subcircuit) -> None:
        """Evolves the state of the simulation under the action of
        the specified gate operations.

        Args:
            subcircuit (Subcircuit): Gate operations to apply for
                evolving the state of the simulation.

        Note:
            This method mutates the state of the simulation.
        """
        raise NotImplementedError("evolve has not been implemented.")


class StateVectorSimulation(Simulation):
    def __init__(self, qubit_count: int, shots: int, subcircuit: Subcircuit):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as state vector
                or expectation, are generated.
        """

        super().__init__(qubit_count=qubit_count, shots=shots)
        self._initialize_memory_objects(subcircuit)

    def _initialize_memory_objects(self, subcircuit: Subcircuit):
        match subcircuit.offload:
            case OffloadType.NONE:
                self._initialize_memory_objects_no_offload(subcircuit)
            case OffloadType.CPU:
                self._initialize_memory_objects_cpu_offload(subcircuit)
            case OffloadType.STORAGE:
                self._initialize_memory_objects_storage_offload(subcircuit)
            case _:
                raise TypeError(f"Unknown type {subcircuit.offload}")

    def _initialize_memory_objects_no_offload(self, subcircuit: Subcircuit):
        match subcircuit.accelerator:
            case AcceleratorType.CPU:
                self._state_vector = StateVector(subcircuit.qubit_count)
                initialize_basis_z(self._state_vector)

            case AcceleratorType.CUDA:
                self._state_vector = StateVector(
                    DeviceType.CUDA, subcircuit.qubit_count)
                initialize_basis_z(self._state_vector)

            case AcceleratorType.HYBRID:
                self._state_vector = StateVector(subcircuit.qubit_count)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, subcircuit.max_qubit_count_cuda)
                state_vector_slice = self._state_vector.slice(
                    subcircuit.max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case _:
                raise TypeError(f"Unknown type {subcircuit.accelerator}")

    def _initialize_memory_objects_cpu_offload(self, subcircuit: Subcircuit):
        match subcircuit.accelerator:
            case AcceleratorType.CPU:
                self._state_vector = StateVector(subcircuit.qubit_count)
                initialize_basis_z(self._state_vector)

            case AcceleratorType.CUDA:
                self._state_vector = StateVector(subcircuit.qubit_count)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, subcircuit.max_qubit_count_cuda)

                state_vector_slice = self._state_vector.slice(
                    subcircuit.max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case AcceleratorType.HYBRID:
                self._state_vector = StateVector(subcircuit.qubit_count)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, subcircuit.max_qubit_count_cuda)

                state_vector_slice = self._state_vector.slice(
                    self._max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case _:
                raise TypeError(f"Unknown type {subcircuit.accelerator}")

    def _initialize_memory_objects_storage_offload(self, subcircuit: Subcircuit):
        match subcircuit.accelerator:
            case AcceleratorType.CPU:
                self._state_vector = StateVector(
                    DeviceType.STORAGE, subcircuit.qubit_count)
                self._state_vector_cpu = StateVector(
                    subcircuit.max_qubit_count)
                initialize_basis_z(self._state_vector_cpu)

            case AcceleratorType.CUDA:
                self._state_vector = StateVector(
                    DeviceType.STORAGE, subcircuit.qubit_count)
                self._state_vector_cpu = StateVector(
                    subcircuit.max_qubit_count_cuda)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, subcircuit.max_qubit_count_cuda)
                state_vector_slice = self._state_vector_cpu.slice(
                    subcircuit.max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case AcceleratorType.HYBRID:
                self._state_vector = StateVector(
                    DeviceType.STORAGE, subcircuit.qubit_count)
                self._state_vector_cpu = StateVector(
                    subcircuit.max_qubit_count_cuda)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, subcircuit.max_qubit_count_cuda)
                state_vector_slice = self._state_vector_cpu.slice(
                    subcircuit.max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case _:
                raise TypeError(f"Unknown type {subcircuit.accelerator}")

    @property
    def state_vector(self) -> np.ndarray:
        """
        np.ndarray: The state vector specifying the current state of the simulation.

        Note:
            Mutating this array will mutate the state of the simulation.
        """
        return np.array(self._state_vector.cpu(), copy=False)

    @property
    def probabilities(self) -> np.ndarray:
        """
        np.ndarray: The probabilities of each computational basis state of the current state
            vector of the simulation.
        """
        return np.abs(self.state_vector) ** 2

    def retrieve_samples(self) -> list[int]:
        return np.random.choice(len(self.state_vector), p=self.probabilities, size=self._shots)

    def evolve(self, subcircuit: Subcircuit) -> None:
        offload = subcircuit.offload

        match offload:
            case OffloadType.NONE:
                self._evolve_no_offload(subcircuit)
            case OffloadType.CPU:
                self._evolve_cpu_offload(subcircuit)
            case OffloadType.STORAGE:
                self._evolve_storage_offload(subcircuit)
            case _:
                raise TypeError(f"Unknown type {offload}")

    def _evolve_no_offload(self, subcircuit: Subcircuit) -> None:
        accelerator = subcircuit.accelerator
        match accelerator:
            case AcceleratorType.CPU:
                self._evolve_no_offload_cpu(subcircuit)
            case AcceleratorType.CUDA:
                self._evolve_no_offload_cuda(subcircuit)
            case AcceleratorType.HYBRID:
                self._evolve_no_offload_hybrid(subcircuit)
            case _:
                raise TypeError(f"Unknown type {accelerator}")

    def _evolve_cpu_offload(self, subcircuit: Subcircuit) -> None:
        accelerator = subcircuit.accelerator
        match accelerator:
            case AcceleratorType.CPU:
                self._evolve_cpu_offload_cpu(subcircuit)
            case AcceleratorType.CUDA:
                self._evolve_cpu_offload_cuda(subcircuit)
            case AcceleratorType.HYBRID:
                self._evolve_cpu_offload_hybrid(subcircuit)
            case _:
                raise TypeError(f"Unknown type {accelerator}")

    def _evolve_storage_offload(self, subcircuit: Subcircuit) -> None:
        accelerator = subcircuit.accelerator
        match accelerator:
            case AcceleratorType.CPU:
                self._evolve_storage_offload_cpu(subcircuit)
            case AcceleratorType.CUDA:
                self._evolve_storage_offload_cuda(subcircuit)
            case AcceleratorType.HYBRID:
                self._evolve_storage_offload_hybrid(subcircuit)
            case _:
                raise TypeError(f"Unknown type {accelerator}")

    #
    # No offload
    #
    def _evolve_no_offload_cpu(self, subcircuit: Subcircuit) -> None:
        state_vector = self._state_vector
        operations = subcircuit.operations
        for op in operations:
            apply(state_vector, op, subcircuit.qubit_count, op.targets)

    def _evolve_no_offload_cuda(self, subcircuit: Subcircuit) -> None:
        state_vector = self._state_vector
        operations = subcircuit.operations
        for op in operations:
            apply(state_vector, op, subcircuit.qubit_count, op.targets)

    def _evolve_no_offload_hybrid(self, subcircuit: Subcircuit) -> None:
        state_vector = self._state_vector
        state_vector_cuda = self._state_vector_cuda

        for s, partitioned_subcircuit in enumerate(subcircuit.operations):
            if isinstance(partitioned_subcircuit[0], list):
                for i, sliced_subcircuit in enumerate(partitioned_subcircuit):
                    state_vector_slice = state_vector.slice(
                        subcircuit.max_qubit_count_cuda, i)
                    if s != 0 or i == 0:
                        state_vector_cuda.copy(state_vector_slice)
                        for op in sliced_subcircuit:
                            apply(state_vector_cuda, op,
                                  subcircuit.qubit_count, op.targets)
                        state_vector_slice.copy(state_vector_cuda)
                    else:
                        initialize_zero(state_vector_slice)
            else:
                for op in partitioned_subcircuit:
                    apply(state_vector, op, subcircuit.qubit_count, op.targets)

    #
    # CPU offload
    #
    def _evolve_cpu_offload_cpu(self, subcircuit: Subcircuit) -> None:
        """CPU offload with CPU simulation is equivalent to CPU simulation"""
        return self._evolve_cpu(subcircuit)

    def _evolve_cpu_offload_cuda(self, subcircuit: Subcircuit) -> None:
        state_vector = self._state_vector
        state_vector_cuda = self._state_vector_cuda

        for s, partitioned_subcircuit in enumerate(subcircuit.operations):
            if isinstance(partitioned_subcircuit[0], list):
                for i, sliced_subcircuit in enumerate(partitioned_subcircuit):
                    state_vector_slice = state_vector.slice(
                        subcircuit.max_qubit_count_cuda, i)
                    if s != 0 or i == 0:
                        state_vector_cuda.copy(state_vector_slice)
                        for op in sliced_subcircuit:
                            apply(state_vector_cuda, op,
                                  subcircuit.qubit_count, op.targets)
                        state_vector_slice.copy(state_vector_cuda)
                    else:
                        initialize_zero(state_vector_slice)
            else:
                for op in partitioned_subcircuit:
                    apply(state_vector, op, subcircuit.qubit_count, op.targets)

    def _evolve_cpu_offload_hybrid(self, subcircuit: Subcircuit) -> None:
        """CPU offload with Hybrid simulation is equivalent to Hybrid simulation"""
        return self._evolve_no_offload_hybrid(subcircuit)

    #
    # Storage offload
    #
    def _evolve_storage_offload_cpu(self, subcircuit: Subcircuit) -> None:
        state_vector = self._state_vector
        state_vector_cpu = self._state_vector_cpu

        for s, partitioned_subcircuit in enumerate(subcircuit.operations):
            if isinstance(partitioned_subcircuit[0], list):
                for i, sliced_subcircuit in enumerate(partitioned_subcircuit):
                    state_vector_slice = state_vector.slice(
                        subcircuit.max_qubit_count, i)
                    if s != 0 or i == 0:
                        state_vector_cpu.copy(state_vector_slice)
                        for op in sliced_subcircuit:
                            apply(state_vector_cpu, op,
                                  subcircuit.qubit_count, op.targets)
                        state_vector_slice.copy(state_vector_cpu)
                    else:
                        initialize_zero(state_vector_slice)
            else:
                for op in partitioned_subcircuit:
                    apply(state_vector, op, subcircuit.qubit_count, op.targets)

    def _evolve_storage_offload_cuda(self, subcircuit: Subcircuit) -> None:
        state_vector = self._state_vector
        state_vector_cpu = self._state_vector_cpu
        state_vector_cuda = self._state_vector_cuda

        for s, partitioned_subcircuit in enumerate(subcircuit.operations):
            if isinstance(partitioned_subcircuit[0], list):
                for i, sliced_subcircuit in enumerate(partitioned_subcircuit):
                    state_vector_slice = state_vector.slice(
                        subcircuit.max_qubit_count, i)
                    if s != 0 or i == 0:
                        state_vector_cpu.copy(state_vector_slice)
                        state_vector_cuda.copy(state_vector_cpu)
                        for op in sliced_subcircuit:
                            apply(state_vector_cuda, op,
                                  subcircuit.qubit_count, op.targets)
                        state_vector_cpu.copy(state_vector_cuda)
                    else:
                        initialize_zero(state_vector_cpu)

                    state_vector_slice.copy(state_vector_cpu)
            else:
                for op in partitioned_subcircuit:
                    apply(state_vector, op, subcircuit.qubit_count, op.targets)

    def _evolve_storage_offload_hybrid(self, subcircuit: Subcircuit) -> None:
        state_vector = self._state_vector
        state_vector_cpu = self._state_vector_cpu
        state_vector_cuda = self._state_vector_cuda

        for s, partitioned_subcircuit in enumerate(subcircuit.operations):
            if isinstance(partitioned_subcircuit[0], list):
                for i, sliced_subcircuit in enumerate(partitioned_subcircuit):
                    state_vector_slice = state_vector.slice(
                        subcircuit.max_qubit_count, i)
                    if s != 0 or i == 0:
                        state_vector_cpu.copy(state_vector_slice)
                        if isinstance(sliced_subcircuit[0], list):
                            for j, re_sliced_subcircuit in enumerate(sliced_subcircuit):
                                state_vector_cpu_slice = state_vector_cpu.slice(
                                    subcircuit.max_qubit_count_cuda, j)
                                state_vector_cuda.copy(state_vector_cpu_slice)
                                for op in re_sliced_subcircuit:
                                    apply(state_vector_cuda, op,
                                          subcircuit.qubit_count, op.targets)
                                state_vector_cpu_slice.copy(state_vector_cuda)
                        else:
                            for op in sliced_subcircuit:
                                apply(state_vector_cpu, op,
                                      subcircuit.qubit_count, op.targets)
                    else:
                        initialize_zero(state_vector_cpu)
            else:
                for op in partitioned_subcircuit:
                    apply(state_vector, op, subcircuit.qubit_count, op.targets)
