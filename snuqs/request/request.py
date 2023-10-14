from snuqs import QuantumCircuit
from enum import Enum
from typing import Optional
import threading

class Request:
    class Status(Enum):
        CREATED = 1
        QUEUED = 2
        RUNNING = 3
        CANCELLED = 4
        DONE = 5
        ERROR = 6


    def __init__(self,
                 circ: QuantumCircuit,
                 status: Status = Status.CREATED,
                 **kwargs):
        pass


    def set_queued(self):
        self.status = Status.QUEUED


    def attach(self, th: threading.Thread, start: bool = True):
        self.th = th
        if start:
            self.start()


    def start(self):
        self.th.start()


    def wait(self):
        self.th.join()


    def result(self):
        self.wait()
        return Result(self.circ)
