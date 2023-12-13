import snuqs._C
from snuqs.circuit import Circuit
from snuqs.result import Result
import threading
from typing import Dict, List, Union

import tempfile
import qiskit
from snuqs import QasmCompiler


class Initialize(qiskit.circuit.Gate):
    def __init__(self, name, num_qubits, params: List[complex]):
        self.name = name
        self.num_qubits = num_qubits
        self._params = params
        self._num_clbits = 0
        self._condition = None

    def validate_parameter(self, parameter):
        return True


class StatevectorSimulator:
    def __init__(self):
        self.sim = snuqs._C.StatevectorSimulator()

    def _run(self, circ: Circuit, ret: Dict[str, any]):
        ret['state'] = self.sim.run(circ)

    def qiskit_for_snuqs(self, circ: qiskit.QuantumCircuit):
        circ = circ.copy()
        new_data = []
        first_init = None
        if circ.data[0].operation.name == "initialize" and len(circ.data[0].qubits) == circ.num_qubits:
            first_init = snuqs._C.INITIALIZE(circ.data[0].operation.params)
            circ.data = circ.data[1:]

#        for instr in circ.data:
#            if instr.operation.name == "initialize":
#                new_data.append(
#                    qiskit.circuit.CircuitInstruction(
#                        Initialize("initialize", instr.operation.num_qubits,
#                                   instr.operation.params),
#                        instr.qubits,
#                        instr.clbits
#                    )
#                )
#            else:
#                new_data.append(instr)
#
#        circ.data = new_data

        with tempfile.NamedTemporaryFile() as f:
            circ.qasm(filename=f.name)
            compiler = QasmCompiler()
            circ = compiler.compile(f.name)
            if first_init:
                circ.prepend(first_init)
        return circ

    def run(self, circ: Union[qiskit.QuantumCircuit, Circuit]):
        if isinstance(circ, qiskit.QuantumCircuit):
            circ = self.qiskit_for_snuqs(circ)
        elif isinstance(circ, Circuit):
            pass
        else:
            raise "Illegal input to simulator"
        ret = {}
        return Result(threading.Thread(target=self._run, args=[circ, ret]), ret)
