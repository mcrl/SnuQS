from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)

import math
import numpy as np
import uuid
from abc import ABC, abstractmethod
from braket.task_result import (
    AdditionalMetadata,
    GateModelTaskResult,
    ResultTypeValue,
    TaskMetadata,
)
from typing import Union, List, Optional, Any
from braket.snuqs.operation_helpers import (
    from_braket_instruction,
)
from braket.snuqs._C.core import initialize_fs, is_initialized_fs
from braket.snuqs._C.core import mem_info
from braket.snuqs._C.core.cuda import mem_info as mem_info_cuda
from braket.snuqs.operation import Operation
from braket.snuqs.simulation import Simulation, StateVectorSimulation
from braket.snuqs.subcircuit import Subcircuit
from braket.snuqs.transpile import transpile
from braket.snuqs.openqasm.circuit import Circuit as OpenQASMCircuit
from braket.snuqs.openqasm.interpreter import Interpreter
from braket.snuqs.openqasm.program_context import AbstractProgramContext, ProgramContext
from braket.snuqs.types import AcceleratorType, PrefetchType, OffloadType
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.jaqcd.shared_models import MultiTarget, OptionalMultiTarget
from braket.ir.jaqcd.program_v1 import Results
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.snuqs.result_types import (
    ResultType,
    TargetedResultType,
    from_braket_result_type,
)


class BaseSimulator(ABC):
    @staticmethod
    def _validate_qubit_counts(qubit_count: int,
                               qubit_count_cpu: Optional[int],
                               qubit_count_cuda: Optional[int],
                               qubit_count_slice: Optional[int],
                               prefetch: PrefetchType):
        if qubit_count_cpu is None:
            free, _ = mem_info()
            qubit_count_cpu = int(math.log(free / 16, 2))
            if prefetch != PrefetchType.NONE:
                qubit_count_cpu -= 1
            qubit_count_cpu = min(qubit_count, qubit_count_cpu)

        if qubit_count_cuda is None:
            free_cuda, _ = mem_info_cuda()
            qubit_count_cuda = int(math.log(free_cuda / 16, 2))
            if prefetch != PrefetchType.NONE:
                qubit_count_cuda -= 1
            qubit_count_cuda = min(qubit_count, qubit_count_cuda)

        if qubit_count_slice is None:
            qubit_count_slice = qubit_count_cpu - 1

        if qubit_count_cuda < qubit_count_slice:
            raise ValueError(
                f"qubit_count_cuda {qubit_count_cuda} must be larger than or equal to qubit_count_slice {qubit_count_slice}")
        if qubit_count_cpu < qubit_count_cuda:
            raise ValueError(
                f"qubit_count_cpu {qubit_count_cpu} must be larger than or equal to qubit_count_cuda {qubit_count_cuda}")

        if qubit_count < qubit_count_cpu:
            raise ValueError(
                f"qubit_count {qubit_count} must be larger than or equal to qubit_count_cpu {qubit_count_cpu}")

        return qubit_count_cpu, qubit_count_cuda, qubit_count_slice

    @staticmethod
    def _validate_simulation_type(accelerator: Optional[str]):
        supported_accelerators = {
            'cpu': AcceleratorType.CPU,
            'cuda': AcceleratorType.CUDA,
            'hybrid': AcceleratorType.HYBRID,
        }
        if accelerator is None:
            return AcceleratorType.CPU

        if accelerator not in supported_accelerators:
            raise TypeError(f"Accelerator {accelerator} is not supported")

        return supported_accelerators[accelerator]

    @staticmethod
    def _validate_prefetch_type(prefetch: Optional[str]):
        supported_prefetches = {
            'none': PrefetchType.NONE,
            'cuda': PrefetchType.CUDA,
            'storage': PrefetchType.STORAGE,
        }
        if prefetch is None:
            return PrefetchType.NONE

        if prefetch not in supported_prefetches:
            raise TypeError(f"Prefetch {prefetch} is not supported")

        return supported_prefetches[prefetch]

    @staticmethod
    def _validate_offload_type(offload: Optional[str],
                               path: Optional[List[str]],
                               block_count: Optional[int],
                               ):
        supported_offloads = {
            'none': OffloadType.NONE,
            'cpu': OffloadType.CPU,
            'storage': OffloadType.STORAGE,
        }
        if offload is None:
            return OffloadType.NONE

        if offload not in supported_offloads:
            raise TypeError(f"Offload {offload} is not supported")

        offload_type = supported_offloads[offload]
        if offload_type == OffloadType.STORAGE:
            if path is None:
                raise ValueError("path must be given for storage offloading")

        return offload_type

    def create_program_context(self) -> AbstractProgramContext:
        return ProgramContext()

    def parse_program(self, program: OpenQASMProgram) -> AbstractProgramContext:
        """Parses an OpenQASM program and returns a program context.

        Args:
            program (OpenQASMProgram): The program to parse.

        Returns:
            AbstractProgramContext: The program context after the program has been parsed.
        """
        is_file = program.source.endswith(".qasm")
        interpreter = Interpreter(self.create_program_context())
        return interpreter.run(
            source=program.source,
            inputs=program.inputs,
            is_file=is_file,
        )

    def run(self, ir: Union[OpenQASMProgram, JaqcdProgram], *args, **kwargs):
        if isinstance(ir, OpenQASMProgram):
            return self.run_openqasm(ir, *args, **kwargs)
        return self.run_jaqcd(ir, *args, **kwargs)

    @staticmethod
    def _map_instruction_attributes(instruction, qubit_map: dict):
        """
        Maps the qubit attributes of an instruction from JaqcdProgram to the corresponding
        qubits in the qubit_map.

        Args:
            instruction: The Jaqcd instruction whose qubit attributes need to be mapped.
            qubit_map (dict): A dictionary mapping original qubits to new qubits.
        """
        if hasattr(instruction, "control"):
            instruction.control = qubit_map.get(
                instruction.control, instruction.control)

        if hasattr(instruction, "controls") and instruction.controls:
            instruction.controls = [qubit_map.get(
                q, q) for q in instruction.controls]

        if hasattr(instruction, "target"):
            instruction.target = qubit_map.get(
                instruction.target, instruction.target)

        if hasattr(instruction, "targets") and instruction.targets:
            instruction.targets = [qubit_map.get(
                q, q) for q in instruction.targets]

    @staticmethod
    def _map_circuit_instructions(circuit: OpenQASMCircuit, qubit_map: dict):
        """
        Maps the targets of each instruction in the circuit to the corresponding qubits in the
        qubit_map.

        Args:
            circuit (OpenQASMCircuit): The circuit containing the instructions.
            qubit_map (dict): A dictionary mapping original qubits to new qubits.
        """
        for ins in circuit.instructions:
            ins.targets = tuple([qubit_map[q] for q in ins.targets])

    @staticmethod
    def _map_circuit_results(circuit: OpenQASMCircuit, qubit_map: dict):
        """
        Maps the targets of each result in the circuit to the corresponding qubits in the qubit_map.

        Args:
            circuit (OpenQASMCircuit): The circuit containing the results.
            qubit_map (dict): A dictionary mapping original qubits to new qubits.
        """
        for result in circuit.results:
            if isinstance(result, (MultiTarget, OptionalMultiTarget)) and result.targets:
                result.targets = [qubit_map[q] for q in result.targets]

    @staticmethod
    def _map_jaqcd_instructions(circuit: JaqcdProgram, qubit_map: dict):
        """
        Maps the attributes of each instruction in the JaqcdProgram to the corresponding qubits in
        the qubit_map.

        Args:
            circuit (JaqcdProgram): The JaqcdProgram containing the instructions.
            qubit_map (dict): A dictionary mapping original qubits to new qubits.
        """
        for ins in circuit.instructions:
            BaseSimulator._map_instruction_attributes(ins, qubit_map)

        if hasattr(circuit, "results") and circuit.results:
            for ins in circuit.results:
                BaseSimulator._map_instruction_attributes(ins, qubit_map)

        if circuit.basis_rotation_instructions:
            for ins in circuit.basis_rotation_instructions:
                ins.target = qubit_map[ins.target]

    @staticmethod
    def _map_circuit_qubits(circuit: Union[OpenQASMCircuit, JaqcdProgram], qubit_map: dict[int, int]):
        """
        Maps the qubits in operations and result types to contiguous qubits.

        Args:
            circuit (OpenQASMCircuit): The circuit containing the operations and result types.
            qubit_map (dict[int, int]): The mapping from qubits to their contiguous indices.

        Returns:
            Circuit: The circuit with qubits in operations and result types mapped
            to contiguous qubits.
        """
        if isinstance(circuit, OpenQASMCircuit):
            BaseSimulator._map_circuit_instructions(circuit, qubit_map)
            BaseSimulator._map_circuit_results(circuit, qubit_map)
        else:
            BaseSimulator._map_jaqcd_instructions(circuit, qubit_map)
        return circuit

    @staticmethod
    def _get_qubits_referenced(operations: list[Operation]) -> set[int]:
        return {target for operation in operations for target in operation.targets}

    @staticmethod
    def _get_circuit_qubit_set(circuit: Union[OpenQASMCircuit, JaqcdProgram]) -> set[int]:
        """
        Returns the set of qubits used in the given circuit.

        Args:
            circuit (Union[OpenQASMCircuit, JaqcdProgram]): The circuit from which to extract the qubit set.

        Returns:
            set[int]: The set of qubits used in the circuit.
        """
        if isinstance(circuit, OpenQASMCircuit):
            return circuit.qubit_set
        else:
            operations = [
                from_braket_instruction(instruction) for instruction in circuit.instructions
            ]
            if circuit.basis_rotation_instructions:
                operations.extend(
                    from_braket_instruction(instruction)
                    for instruction in circuit.basis_rotation_instructions
                )
            return BaseSimulator._get_qubits_referenced(operations)

    @staticmethod
    def _contiguous_qubit_mapping(qubit_set: set[int]) -> dict[int, int]:
        """
        Maping of qubits to contiguous integers. The qubit mapping may be discontiguous or
        contiguous.

        Args:
            qubit_set (set[int]): List of qubits to be mapped.

        Returns:
            dict[int, int]: Dictionary where keys are qubits and values are contiguous integers.
        """
        return {q: i for i, q in enumerate(sorted(qubit_set))}

    @staticmethod
    def _map_circuit_to_contiguous_qubits(circuit: Union[OpenQASMCircuit, JaqcdProgram]) -> dict[int, int]:
        """
        Maps the qubits in operations and result types to contiguous qubits.

        Args:
            circuit (Union[OpenQASMCircuit, JaqcdProgram]): The circuit containing the operations and
            result types.

        Returns:
            dict[int, int]: Map of qubit index to corresponding contiguous index
        """
        circuit_qubit_set = BaseSimulator._get_circuit_qubit_set(circuit)
        qubit_map = BaseSimulator._contiguous_qubit_mapping(circuit_qubit_set)
        BaseSimulator._map_circuit_qubits(circuit, qubit_map)
        return qubit_map

    @staticmethod
    def _formatted_measurements(
        simulation: Simulation, measured_qubits: Union[list[int], None] = None
    ) -> list[list[str]]:
        """Retrieves formatted measurements obtained from the specified simulation.

        Args:
            simulation (Simulation): Simulation to use for obtaining the measurements.
            measured_qubits (list[int] | None): The qubits that were measured.

        Returns:
            list[list[str]]: List containing the measurements, where each measurement consists
            of a list of measured values of qubits.
        """
        # Get the full measurements
        measurements = [
            list("{number:0{width}b}".format(
                number=sample, width=simulation.qubit_count))
            for sample in simulation.retrieve_samples()
        ]
        #  Gets the subset of measurements from the full measurements
        if measured_qubits is not None and measured_qubits != []:
            measured_qubits = np.array(measured_qubits)
            in_circuit_mask = measured_qubits < simulation.qubit_count
            measured_qubits_in_circuit = measured_qubits[in_circuit_mask]
            measured_qubits_not_in_circuit = measured_qubits[~in_circuit_mask]

            measurements_array = np.array(measurements)
            selected_measurements = measurements_array[:,
                                                       measured_qubits_in_circuit]
            measurements = np.pad(
                selected_measurements, ((0, 0), (0, len(
                    measured_qubits_not_in_circuit)))
            ).tolist()
        return measurements

    def _create_results_obj(
        self,
        results: list[dict[str, Any]],
        openqasm_ir: OpenQASMProgram,
        simulation: Simulation,
        measured_qubits: list[int] = None,
        mapped_measured_qubits: list[int] = None,
    ) -> GateModelTaskResult:
        return GateModelTaskResult.construct(
            taskMetadata=TaskMetadata(
                id=str(uuid.uuid4()),
                shots=simulation.shots,
                deviceId=self.DEVICE_ID,
            ),
            additionalMetadata=AdditionalMetadata(
                action=openqasm_ir,
            ),
            resultTypes=results,
            measurements=self._formatted_measurements(
                simulation, mapped_measured_qubits),
            measuredQubits=(measured_qubits or list(
                range(simulation.qubit_count))),
        )

    @staticmethod
    def _generate_results(
        results: list[Results],
        result_types: list[ResultType],
        simulation: Simulation,
    ) -> list[ResultTypeValue]:
        return [
            ResultTypeValue.construct(
                type=results[index],
                value=result_types[index].calculate(simulation),
            )
            for index in range(len(results))
        ]

    @staticmethod
    def _translate_result_types(results: list[Results]) -> list[ResultType]:
        return [from_braket_result_type(result) for result in results]

    def run_openqasm(self, ir: OpenQASMProgram,
                     *,
                     shots: int = 0,
                     qubit_count: Optional[int] = None,
                     qubit_count_cpu: Optional[int] = None,
                     qubit_count_cuda: Optional[int] = None,
                     qubit_count_slice: Optional[int] = None,
                     accelerator: Optional[str] = None,
                     prefetch: Optional[str] = None,
                     offload: Optional[str] = None,
                     path: Optional[List[str]] = None,
                     block_count: Optional[int] = None,
                     ) -> GateModelTaskResult:
        circuit = self.parse_program(ir).circuit
        qubit_map = BaseSimulator._map_circuit_to_contiguous_qubits(circuit)
        qubit_count = circuit.num_qubits
        measured_qubits = circuit.measured_qubits
        mapped_measured_qubits = (
            [qubit_map[q] for q in measured_qubits] if measured_qubits else None
        )

        qubit_count_cpu, qubit_count_cuda, qubit_count_slice = BaseSimulator._validate_qubit_counts(
            qubit_count, qubit_count_cpu, qubit_count_cuda, qubit_count_slice, prefetch)
        accelerator = BaseSimulator._validate_simulation_type(accelerator)
        prefetch = BaseSimulator._validate_prefetch_type(prefetch)
        offload = BaseSimulator._validate_offload_type(
            offload, path, block_count)

        operations = circuit.instructions
        if shots:
            for instruction in circuit.basis_rotation_instructions:
                operations.append(instruction)

        if offload == OffloadType.STORAGE and not is_initialized_fs():
            initialize_fs((2**qubit_count) * 16, block_count, path)

        subcircuit = transpile(
            qubit_count,
            qubit_count_cpu,
            qubit_count_cuda,
            qubit_count_slice,
            accelerator,
            prefetch,
            offload,
            operations,
        )
        simulation = self.initialize_simulation(
            qubit_count=qubit_count, shots=shots, subcircuit=subcircuit)
        simulation.evolve(subcircuit)

        results = circuit.results
        if not shots:
            result_types = BaseSimulator._translate_result_types(results)
            results = BaseSimulator._generate_results(
                circuit.results,
                result_types,
                simulation,
            )

        return self._create_results_obj(
            results, ir, simulation, measured_qubits, mapped_measured_qubits
        )

    def run_jaqcd(self, ir: JaqcdProgram,
                  *,
                  shots: int = 0,
                  qubit_count_cpu: Optional[int] = None,
                  qubit_count_cuda: Optional[int] = None,
                  qubit_count_slice: Optional[int] = None,
                  accelerator: Optional[str] = None,
                  prefetch: Optional[str] = None,
                  offload: Optional[str] = None,
                  path: Optional[List[str]] = None,
                  block_count: Optional[int] = None,
                  ) -> GateModelTaskResult:

        qubit_map = BaseSimulator._map_circuit_to_contiguous_qubits(ir)
        qubit_count = len(qubit_map)

        qubit_count_cpu, qubit_count_cuda, qubit_count_slice = BaseSimulator._validate_qubit_counts(
            qubit_count, qubit_count_cpu, qubit_count_cuda, qubit_count_slice, prefetch)
        accelerator = BaseSimulator._validate_simulation_type(accelerator)
        prefetch = BaseSimulator._validate_prefetch_type(prefetch)
        offload = BaseSimulator._validate_offload_type(
            offload, path, block_count)

        operations = [
            from_braket_instruction(instr) for instr in ir.instructions
        ]

        if shots > 0 and ir.basis_rotation_instructions:
            for instruction in ir.basis_rotation_instructions:
                operations.append(from_braket_instruction(instruction))

        if offload == OffloadType.STORAGE and not is_initialized_fs():
            initialize_fs((2**qubit_count) * 16, block_count, path)

        subcircuit = transpile(
            qubit_count,
            qubit_count_cpu,
            qubit_count_cuda,
            qubit_count_slice,
            accelerator,
            prefetch,
            offload,
            operations,
        )
        simulation = self.initialize_simulation(
            qubit_count=qubit_count, shots=shots, subcircuit=subcircuit)
        simulation.evolve(subcircuit)

        results = []
        if not shots and ir.results:
            result_types = BaseSimulator._translate_result_types(ir.results)
            BaseSimulator._validate_result_types_qubits_exist(
                [
                    result_type
                    for result_type in result_types
                    if isinstance(result_type, TargetedResultType)
                ],
                qubit_count,
            )
            results = self._generate_results(
                ir.results,
                result_types,
                simulation,
            )

        return self._create_results_obj(results, ir, simulation)

    @abstractmethod
    def initialize_simulation(self, **kwargs) -> Simulation:
        """Initializes simulation with keyword arguments"""


class StateVectorSimulator(BaseSimulator):
    DEVICE_ID = "snuqs"

    def __init__(self):
        self.max_qubits = 42

    def _service(self):
        return {
            "executionWindows": [
                {
                    "executionDay": "Everyday",
                    "windowStartHour": "00:00",
                    "windowEndHour": "23:59:59",
                }
            ],
            "shotsRange": [0, 0],
        }

    def _jaqcd(self):
        return {
            "actionType": "braket.ir.jaqcd.program",
            "version": ["1"],
            "supportedOperations": [
                "ccnot",
                "cnot",
                "cphaseshift",
                "cphaseshift00",
                "cphaseshift01",
                "cphaseshift10",
                "cswap",
                "cy",
                "cz",
                "h",
                "i",
                "iswap",
                "pswap",
                "phaseshift",
                "rx",
                "ry",
                "rz",
                "s",
                "si",
                "swap",
                "t",
                "ti",
                "unitary",
                "v",
                "vi",
                "x",
                "xx",
                "xy",
                "y",
                "yy",
                "z",
                "zz",
            ],
            "supportedResultTypes": [
                {"name": "StateVector"},
            ],
        }

    def _openqasm(self):
        return {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": [
                # OpenQASM primitives
                "U",
                "GPhase",
                # builtin Braket gates
                "ccnot",
                "cnot",
                "cphaseshift",
                "cphaseshift00",
                "cphaseshift01",
                "cphaseshift10",
                "cswap",
                "cy",
                "cz",
                "h",
                "i",
                "iswap",
                "phaseshift",
                "pswap",
                "rx",
                "ry",
                "rz",
                "s",
                "si",
                "swap",
                "t",
                "ti",
                "unitary",
                "v",
                "vi",
                "x",
                "xx",
                "xy",
                "y",
                "yy",
                "z",
                "zz",
            ],
            "supportedModifiers": [
                {
                    "name": "ctrl",
                },
                {
                    "name": "negctrl",
                },
                {
                    "name": "pow",
                    "exponent_types": ["int", "float"],
                },
                {
                    "name": "inv",
                },
            ],
            "forbiddenPragmas": [
                "braket_unitary_matrix",
                "braket_result_type_state_vector",
                "braket_result_type_density_matrix",
                "braket_result_type_sample",
                "braket_result_type_expectation",
                "braket_result_type_variance",
                "braket_result_type_probability",
                "braket_result_type_amplitude",
                "braket_result_type_adjoint_gradient",
                "braket_noise_amplitude_damping",
                "braket_noise_bit_flip",
                "braket_noise_depolarizing",
                "braket_noise_kraus",
                "braket_noise_pauli_channel",
                "braket_noise_generalized_amplitude_damping",
                "braket_noise_phase_flip",
                "braket_noise_phase_damping",
                "braket_noise_two_qubit_dephasing",
                "braket_noise_two_qubit_depolarizing",
            ],
            "supportPhysicalQubits": False,
            "supportsPartialVerbatimBox": False,
            "requiresContiguousQubitIndices": False,
            "requiresAllQubitsMeasurement": False,
            "supportsUnassignedMeasurements": False,
            "disabledQubitRewiringSupported": False,
            "supportedResultTypes": [
                {"name": "StateVector"},
            ],
        }

    def _action(self):
        return {
            "braket.ir.jaqcd.program": self._jaqcd(),
            "braket.ir.openqasm.program": self._openqasm(),
        }

    def _paradigm(self):
        return {"qubitCount": self.max_qubits}

    def _deviceParameters(self):
        return GateModelSimulatorDeviceParameters.schema()

    @property
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        return GateModelSimulatorDeviceCapabilities.parse_obj(
            {
                "service": self._service(),
                "action": self._action(),
                "paradigm": self._paradigm(),
                "deviceParameters": self._deviceParameters(),
            }
        )

    def initialize_simulation(self,
                              *,
                              qubit_count: int,
                              shots: int,
                              subcircuit: Subcircuit,
                              ) -> StateVectorSimulation:
        """Initializes simulation with keyword arguments"""
        return StateVectorSimulation(qubit_count, shots, subcircuit)
