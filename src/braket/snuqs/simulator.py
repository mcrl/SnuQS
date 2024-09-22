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
from typing import Union
from braket.snuqs.operation_helpers import (
    from_braket_instruction,
)
import braket.snuqs.gate_operations  # This line must follow
from braket.snuqs.simulation import Simulation, StateVectorSimulation

IRTYPE = Union[braket.ir.openqasm.Program, braket.ir.jaqcd.Program],


class BaseSimulator(ABC):
    def run(self, ir: IRTYPE, *args, **kwargs):
        if isinstance(ir, braket.ir.openqasm.program_v1.Program):
            return self.run_openqasm(ir, *args, **kwargs)
        return self.run_jaqcd(ir, *args, **kwargs)

    def run_openqasm(self, ir: braket.ir.openqasm.Program, *args, **kwargs) -> GateModelTaskResult:
        raise "Not Implemented"

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
    def _map_jaqcd_instructions(circuit: braket.ir.jaqcd.Program, qubit_map: dict):
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
    def _map_circuit_qubits(circuit: braket.ir.jaqcd.Program, qubit_map: dict[int, int]):
        """
        Maps the qubits in operations and result types to contiguous qubits.

        Args:
            circuit (Circuit): The circuit containing the operations and result types.
            qubit_map (dict[int, int]): The mapping from qubits to their contiguous indices.

        Returns:
            Circuit: The circuit with qubits in operations and result types mapped
            to contiguous qubits.
        """
        BaseSimulator._map_jaqcd_instructions(circuit, qubit_map)
        return circuit

    def run_jaqcd(self, ir: braket.ir.jaqcd.Program, *args, **kwargs) -> GateModelTaskResult:
        qubit_map = self._qubit_map(ir)
        BaseSimulator._map_circuit_qubits(ir, qubit_map)
        qubit_count = len(qubit_map)

        operations = [
            from_braket_instruction(instr) for instr in ir.instructions
        ]

        simulation = self.initialize_simulation(qubit_count=qubit_count, shots=0)
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
                "cv",
                "cy",
                "cz",
                "ecr",
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
                "cv",
                "cy",
                "cz",
                "ecr",
                "h",
                "i",
                "iswap",
                "phaseshift",
                "prx",
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
