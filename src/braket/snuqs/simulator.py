from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)

import uuid
import braket
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
from braket.snuqs.operation import Operation
import braket.snuqs.gate_operations  # This line must follow
from braket.snuqs.simulation import Simulation, StateVectorSimulation
from braket.default_simulator.openqasm.circuit import Circuit as OpenQASMCircuit
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.program_context import AbstractProgramContext, ProgramContext
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.jaqcd.shared_models import MultiTarget, OptionalMultiTarget
from braket.ir.openqasm import Program as OpenQASMProgram


class BaseSimulator(ABC):
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
            instruction.control = qubit_map.get(instruction.control, instruction.control)

        if hasattr(instruction, "controls") and instruction.controls:
            instruction.controls = [qubit_map.get(q, q) for q in instruction.controls]

        if hasattr(instruction, "target"):
            instruction.target = qubit_map.get(instruction.target, instruction.target)

        if hasattr(instruction, "targets") and instruction.targets:
            instruction.targets = [qubit_map.get(q, q) for q in instruction.targets]

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
            ins._targets = tuple([qubit_map[q] for q in ins.targets])

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
            print(type(circuit))
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

    def run_openqasm(self, ir: OpenQASMProgram,
                     qubit_count: Any = None,
                     shots: int = 0,
                     *, path: Optional[List[str]] = None) -> GateModelTaskResult:
        circuit = self.parse_program(ir).circuit
        qubit_map = BaseSimulator._map_circuit_to_contiguous_qubits(circuit)
        qubit_count = circuit.num_qubits
        measured_qubits = circuit.measured_qubits
        mapped_measured_qubits = (
            [qubit_map[q] for q in measured_qubits] if measured_qubits else None
        )

        operations = circuit.instructions
        print(operations)

        simulation = self.initialize_simulation(qubit_count=qubit_count, shots=shots)
        simulation.evolve(operations)

        return GateModelTaskResult.construct(
            taskMetadata=TaskMetadata(
                id=str(uuid.uuid4()),
                shots=simulation.shots,
                deviceId=self.DEVICE_ID,
            ),
            additionalMetadata=AdditionalMetadata(
                action=ir,
            ),
            resultTypes=[
                ResultTypeValue.construct(
                    type=braket.ir.openqasm.StateVector(),
                    value=simulation._state_vector,
                ),
            ]
        )

    def run_jaqcd(self, ir: JaqcdProgram,
                  qubit_count: Any = None,
                  shots: int = 0,
                  *, path: Optional[List[str]] = None) -> GateModelTaskResult:
        qubit_map = BaseSimulator._map_circuit_to_contiguous_qubits(ir)
        qubit_count = len(qubit_map)

        operations = [
            from_braket_instruction(instr) for instr in ir.instructions
        ]

        simulation = self.initialize_simulation(qubit_count=qubit_count, shots=shots)
        simulation.evolve(operations)

        return GateModelTaskResult.construct(
            taskMetadata=TaskMetadata(
                id=str(uuid.uuid4()),
                shots=simulation.shots,
                deviceId=self.DEVICE_ID,
            ),
            additionalMetadata=AdditionalMetadata(
                action=ir,
            ),
            resultTypes=[
                ResultTypeValue.construct(
                    type=braket.ir.jaqcd.StateVector(),
                    value=simulation._state_vector,
                ),
            ]
        )

    @abstractmethod
    def _qubit_map(self, ir):
        """Return the qubit mapping"""

    @abstractmethod
    def initialize_simulation(self, **kwargs) -> Simulation:
        """Initializes simulation with keyword arguments"""


class StateVectorSimulator(BaseSimulator):
    DEVICE_ID = "snuqs"

    def __init__(self, *args, **kwargs):
        self.max_qubits = 1

    def _qubit_map(self, ir):
        qubit_set = set()
        for i in ir.instructions:
            if hasattr(i, 'target'):
                qubit_set.add(i.target)

            if hasattr(i, 'control'):
                qubit_set.add(i.control)

            if hasattr(i, 'targets'):
                for t in i.targets:
                    qubit_set.add(t)

            if hasattr(i, 'controls'):
                for t in i.controls:
                    qubit_set.add(t)

        qubit_map = {}
        for i, q in enumerate(sorted(qubit_set)):
            qubit_map[q] = i

        return qubit_map

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

    def initialize_simulation(self, **kwargs) -> StateVectorSimulation:
        """Initializes simulation with keyword arguments"""
        qubit_count = kwargs.get("qubit_count")
        shots = kwargs.get("shots")
        return StateVectorSimulation(qubit_count, shots)
