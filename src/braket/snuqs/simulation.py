import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional, List

from braket.snuqs._C.result_types import StateVector
from braket.snuqs._C.operation import GateOperation
from braket.snuqs._C.functionals import apply, initialize_basis_z, initialize_zero
from braket.snuqs._C.core.cuda import mem_info as mem_info_cuda
from braket.snuqs._C.core import mem_info, attach_fs
from braket.snuqs._C import DeviceType
from braket.snuqs.types import AcceleratorType, OffloadType
from braket.snuqs.transpile import transpile


class Simulation(ABC):
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
    def __init__(self, qubit_count: int, shots: int,
                 accelerator: AcceleratorType,
                 offload: OffloadType,
                 path: Optional[List[str]] = None,
                 count: Optional[int] = None,
                 block_count: Optional[int] = None,
                 ):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as state vector
                or expectation, are generated.
        """

        super().__init__(qubit_count=qubit_count, shots=shots)
        free, _ = mem_info()
        self._max_qubit_count = self._compute_max_qubit_count()
        self._max_qubit_count_cuda = self._compute_max_qubit_count_cuda()

        if offload == OffloadType.STORAGE:
            attach_fs(count, block_count, path)

        # print(f"max_qubit_count: {self._max_qubit_count}, max_qubit_count_cuda: {self._max_qubit_count_cuda}")
        self._initialize_memory_objects(
            qubit_count, shots, accelerator, offload)

    def _compute_max_qubit_count(self):
        free, _ = mem_info()
        return min(
            self.qubit_count, int(math.log(free / 16, 2)))

    def _compute_max_qubit_count_cuda(self):
        free, _ = mem_info_cuda()
        return min(
            self.qubit_count, int(math.log(free / 16, 2)))

    def _initialize_memory_objects(self, qubit_count: int, shots: int,
                                   accelerator: AcceleratorType,
                                   offload: OffloadType):
        match offload:
            case OffloadType.NONE:
                self._initialize_memory_objects_no_offload(
                    qubit_count, shots, accelerator)
            case OffloadType.CPU:
                self._initialize_memory_objects_cpu_offload(
                    qubit_count, shots, accelerator)
            case OffloadType.STORAGE:
                self._initialize_memory_objects_storage_offload(
                    qubit_count, shots, accelerator)
            case _:
                raise NotImplementedError("Not Implemented")

    def _initialize_memory_objects_no_offload(self, qubit_count: int, shots: int,
                                              accelerator: AcceleratorType):
        match accelerator:
            case AcceleratorType.CPU:
                self._state_vector = StateVector(qubit_count)
                initialize_basis_z(self._state_vector)

            case AcceleratorType.CUDA:
                self._state_vector = StateVector(DeviceType.CUDA, qubit_count)
                initialize_basis_z(self._state_vector)

            case AcceleratorType.HYBRID:
                self._state_vector = StateVector(qubit_count)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, self._max_qubit_count_cuda)
                state_vector_slice = self._state_vector.slice(
                    self._max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case _:
                raise NotImplementedError("Not Implemented")

    def _initialize_memory_objects_cpu_offload(self, qubit_count: int, shots: int,
                                               accelerator: AcceleratorType):
        match accelerator:
            case AcceleratorType.CPU:
                self._state_vector = StateVector(qubit_count)
                initialize_basis_z(self._state_vector)

            case AcceleratorType.CUDA:
                self._state_vector = StateVector(qubit_count)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, self._max_qubit_count_cuda)

                state_vector_slice = self._state_vector.slice(
                    self._max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case AcceleratorType.HYBRID:
                self._state_vector = StateVector(qubit_count)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, self._max_qubit_count_cuda)

                state_vector_slice = self._state_vector.slice(
                    self._max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case _:
                raise NotImplementedError("Not Implemented")

    def _initialize_memory_objects_storage_offload(self, qubit_count: int, shots: int,
                                                   accelerator: AcceleratorType):
        match accelerator:
            case AcceleratorType.CPU:
                assert (qubit_count >= self._max_qubit_count)
                self._state_vector = StateVector(
                    DeviceType.STORAGE, qubit_count)
                self._state_vector_cpu = StateVector(self._max_qubit_count)
                initialize_basis_z(self._state_vector_cpu)

            case AcceleratorType.CUDA:
                self._state_vector = StateVector(
                    DeviceType.STORAGE, qubit_count)
                self._state_vector_cpu = StateVector(
                    self._max_qubit_count_cuda)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, self._max_qubit_count_cuda)
                state_vector_slice = self._state_vector_cpu.slice(
                    self._max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case AcceleratorType.HYBRID:
                self._state_vector = StateVector(
                    DeviceType.STORAGE, qubit_count)
                self._state_vector_cpu = StateVector(
                    self._max_qubit_count_cuda)
                self._state_vector_cuda = StateVector(
                    DeviceType.CUDA, self._max_qubit_count_cuda)
                state_vector_slice = self._state_vector_cpu.slice(
                    self._max_qubit_count_cuda, 0)
                initialize_basis_z(state_vector_slice)

            case _:
                raise NotImplementedError("Not Implemented")

    @property
    def state_vector(self) -> np.ndarray:
        """
        np.ndarray: The state vector specifying the current state of the simulation.

        Note:
            Mutating this array will mutate the state of the simulation.
        """
        return np.array(self._state_vector.cpu(), copy=False)

    @ property
    def probabilities(self) -> np.ndarray:
        """
        np.ndarray: The probabilities of each computational basis state of the current state
            vector of the simulation.
        """
        return np.abs(self.state_vector) ** 2

    def retrieve_samples(self) -> list[int]:
        return np.random.choice(len(self.state_vector), p=self.probabilities, size=self._shots)

    def evolve(self,
               operations: list[GateOperation],
               *,
               accelerator: Optional[AcceleratorType] = None,
               offload: Optional[OffloadType] = None) -> None:
        if accelerator is None:
            accelerator = AcceleratorType.CPU
        if offload is None:
            offload = OffloadType.NONE

        match offload:
            case OffloadType.NONE:
                self._evolve_no_offload(operations, accelerator)
            case OffloadType.CPU:
                self._evolve_cpu_offload(operations, accelerator)
            case OffloadType.STORAGE:
                self._evolve_storage_offload(operations, accelerator)
            case _:
                raise NotImplementedError("Not Implemented")

    def _evolve_no_offload(self,
                           operations: list[GateOperation],
                           accelerator: AcceleratorType):
        match accelerator:
            case AcceleratorType.CPU:
                self._evolve_no_offload_cpu(operations)
            case AcceleratorType.CUDA:
                self._evolve_no_offload_cuda(operations)
            case AcceleratorType.HYBRID:
                self._evolve_no_offload_hybrid(operations)
            case _:
                raise NotImplementedError("Not Implemented")

    def _evolve_cpu_offload(self,
                            operations: list[GateOperation],
                            accelerator: AcceleratorType):
        match accelerator:
            case AcceleratorType.CPU:
                self._evolve_cpu_offload_cpu(operations)
            case AcceleratorType.CUDA:
                self._evolve_cpu_offload_cuda(operations)
            case AcceleratorType.HYBRID:
                self._evolve_cpu_offload_hybrid(operations)
            case _:
                raise NotImplementedError("Not Implemented")

    def _evolve_storage_offload(self,
                                operations: list[GateOperation],
                                accelerator: AcceleratorType) -> None:
        match accelerator:
            case AcceleratorType.CPU:
                self._evolve_storage_offload_cpu(operations)
            case AcceleratorType.CUDA:
                self._evolve_storage_offload_cuda(operations)
            case AcceleratorType.HYBRID:
                self._evolve_storage_offload_hybrid(operations)
            case _:
                raise NotImplementedError("Not Implemented")

    #
    # No offload
    #
    def _evolve_no_offload_cpu(self, operations: list[GateOperation]) -> None:
        state_vector = self._state_vector

        operations = transpile(operations,
                               self._qubit_count,
                               self._max_qubit_count,
                               self._max_qubit_count_cuda,
                               AcceleratorType.CPU,
                               OffloadType.NONE
                               )
        for operation in operations:
            targets = operation.targets
            apply(state_vector, operation, self._qubit_count, targets)

    def _evolve_no_offload_cuda(self, operations: list[GateOperation]) -> None:
        state_vector = self._state_vector

        operations = transpile(operations,
                               self._qubit_count,
                               self._max_qubit_count,
                               self._max_qubit_count_cuda,
                               AcceleratorType.CUDA,
                               OffloadType.NONE
                               )
        for operation in operations:
            targets = operation.targets
            apply(state_vector, operation, self._qubit_count, targets)

    def _evolve_no_offload_hybrid(self, operations: list[GateOperation]) -> None:
        state_vector = self._state_vector
        state_vector_cuda = self._state_vector_cuda
        slice_qubit_count = self._qubit_count - self._max_qubit_count_cuda

        list_of_subcircuits = transpile(operations,
                                        self._qubit_count,
                                        self._max_qubit_count,
                                        self._max_qubit_count_cuda,
                                        AcceleratorType.HYBRID,
                                        OffloadType.NONE
                                        )

        for s, subcircuit_slices in enumerate(list_of_subcircuits):
            targets = subcircuit_slices[0][0].targets
            applying_local = len(targets) == 0 or min(
                targets) >= slice_qubit_count
            if applying_local:
                for i, subcircuit in enumerate(subcircuit_slices):
                    state_vector_slice = state_vector.slice(
                        self._max_qubit_count_cuda, i)
                    if s != 0 or i == 0:
                        state_vector_cuda.copy(state_vector_slice)
                    else:
                        initialize_zero(state_vector_cuda)
                    for operation in subcircuit:
                        targets = operation.targets
                        apply(state_vector_cuda, operation,
                              self._qubit_count, targets)
                    state_vector_slice.copy(state_vector_cuda)
            else:
                subcircuit = subcircuit_slices[0]
                for operation in subcircuit:
                    targets = operation.targets
                    apply(state_vector, operation, self._qubit_count, targets)

    #
    # CPU offload
    #
    def _evolve_cpu_offload_cpu(self, operations: list[GateOperation]) -> None:
        """CPU offload with CPU simulation is equivalent to CPU simulation"""
        return self._evolve_cpu(operations)

    def _evolve_cpu_offload_cuda(self, operations: list[GateOperation]) -> None:
        state_vector = self._state_vector
        state_vector_cuda = self._state_vector_cuda
        slice_qubit_count = self._qubit_count - self._max_qubit_count_cuda

        list_of_subcircuits = transpile(operations,
                                        self._qubit_count,
                                        self._max_qubit_count,
                                        self._max_qubit_count_cuda,
                                        AcceleratorType.CUDA,
                                        OffloadType.CPU
                                        )

        for s, subcircuit_slices in enumerate(list_of_subcircuits):
            targets = subcircuit_slices[0][0].targets
            applying_local = len(targets) == 0 or min(
                targets) >= slice_qubit_count
            if applying_local:
                for i, subcircuit in enumerate(subcircuit_slices):
                    state_vector_slice = state_vector.slice(
                        self._max_qubit_count_cuda, i)
                    if s != 0 or i == 0:
                        state_vector_cuda.copy(state_vector_slice)
                    else:
                        initialize_zero(state_vector_cuda)
                    for operation in subcircuit:
                        targets = operation.targets
                        apply(state_vector_cuda, operation,
                              self._qubit_count, targets)
                    state_vector_slice.copy(state_vector_cuda)
            else:
                subcircuit = subcircuit_slices[0]
                for operation in subcircuit:
                    targets = operation.targets
                    apply(state_vector, operation, self._qubit_count, targets)

    def _evolve_cpu_offload_hybrid(self, operations: list[GateOperation]) -> None:
        """CPU offload with Hybrid simulation is equivalent to Hybrid simulation"""
        return self._evolve_no_offload_hybrid(operations)

    #
    # Storage offload
    #
    def _evolve_storage_offload_cpu(self, operations: list[GateOperation]) -> None:
        state_vector = self._state_vector
        state_vector_cpu = self._state_vector_cpu
        slice_qubit_count = self._qubit_count - self._max_qubit_count
        list_of_subcircuits = transpile(operations,
                                        self._qubit_count,
                                        self._max_qubit_count,
                                        self._max_qubit_count_cuda,
                                        AcceleratorType.CPU,
                                        OffloadType.STORAGE
                                        )

        for s, subcircuit_slices in enumerate(list_of_subcircuits):
            targets = subcircuit_slices[0][0].targets
            applying_local = len(targets) == 0 or min(
                targets) >= slice_qubit_count
            if applying_local:
                for i, subcircuit in enumerate(subcircuit_slices):
                    state_vector_slice = state_vector.slice(
                        self._max_qubit_cuda, i)
                    if s != 0 or i == 0:
                        state_vector_cpu.copy(state_vector_slice)
                    else:
                        initialize_zero(state_vector_cpu)
                    for operation in subcircuit:
                        targets = operation.targets
                        print(operation)
                        assert len(targets) == 0 or (
                            min(targets) >= slice_qubit_count)
                        apply(state_vector_cpu, operation,
                              self._qubit_count, targets)
                    state_vector_slice.copy(state_vector_cpu)
            else:
                subcircuit = subcircuit_slices[0]
                for operation in subcircuit:
                    targets = operation.targets
                    print("Trying to apply", operation)
                    # apply(state_vector, operation, self._qubit_count, targets)

        raise NotImplementedError("Not Implemented")

    def _evolve_storage_offload_cuda(self, operations: list[GateOperation]) -> None:
        raise NotImplementedError("Not Implemented")

    def _evolve_storage_offload_hybrid(self, operations: list[GateOperation]) -> None:
        raise NotImplementedError("Not Implemented")
