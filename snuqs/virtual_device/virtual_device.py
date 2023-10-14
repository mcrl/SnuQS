from snuqs.request import Request
from snuqs.quantum_circuit import QuantumCircuit
from abc import *
from dataclasses import dataclass, field
import threading

@dataclass
class VirtualDevice(metaclass=ABCMeta):
    """VirtualDevice"""


    req_queue: list[Request] = field(default_factory=list)


    def run(self, circ: QuantumCircuit, **kwargs):
        self.submit(Request(circ, kwargs))


    def submit(self, req: Request):
        req_queue.append(req)
        req.queued()
        req.attach(threading.Thread(target=self._run, args=[req]))


    @abstractmethod
    def _run(self, req: Request):
        raise "Cannot be here"
