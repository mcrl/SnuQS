import numpy as np

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
from typing import Dict, Union
from braket.default_simulator.operation_helpers import (
    from_braket_instruction,
)

from braket.snuqs.simulation import Simulation, StateVectorSimulation

IRTYPE = Union[braket.ir.openqasm.Program, braket.ir.jaqcd.Program],


class BaseSimulator(ABC):
    def run(self, ir: IRTYPE, *args, **kwargs):
        if isinstance(ir, braket.ir.openqasm.Program):
            return self.run_openqasm(ir, *args, **kwargs)
        return self.run_jaqcd(ir, *args, **kwargs)

    def run_openqasm(self, ir: braket.ir.openqasm.Program, *args, **kwargs) -> GateModelTaskResult:
        raise "Not Implemented"

    def run_jaqcd(self, ir: braket.ir.jaqcd.Program, *args, **kwargs) -> GateModelTaskResult:
        qubit_map = self._qubit_map(ir)
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
    def _state_vector(self, qubit_count: int, qubit_map: Dict[int, int]):
        """Returns the state vector"""

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
            qubit_set.add(i.target)
            if hasattr(i, 'control'):
                qubit_set.add(i.control)

        qubit_map = {}
        for i, q in enumerate(qubit_set):
            qubit_map[i] = q

        return qubit_map

    def _state_vector(self, qubit_count: int, qubit_map: Dict[int, int]):
        sv = np.zeros(2**qubit_count, dtype=complex)
        sv[0] = 1
        return sv

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
