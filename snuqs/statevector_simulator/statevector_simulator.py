import snuqs._C
from snuqs.circuit import Circuit
from snuqs.result import Result
import threading
from typing import Dict

import tempfile
import qiskit
from snuqs import QasmCompiler


class StatevectorSimulator:
    def __init__(self):
        self.sim = snuqs._C.StatevectorSimulator()

    def _run(self, circ: Circuit, ret: Dict[str, any]):
        ret['state'] = self.sim.run(circ)

    def run(self, circ: Circuit):
        if isinstance(circ, qiskit.QuantumCircuit):
            with tempfile.NamedTemporaryFile() as f:
                circ.qasm(filename=f.name)
                compiler = QasmCompiler()
                circ = compiler.compile(f.name)

        ret = {}
        return Result(threading.Thread(target=self._run, args=[circ, ret]), ret)
