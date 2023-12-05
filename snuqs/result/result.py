from enum import Enum
from snuqs.buffer import Buffer


class Result:
    class Status(Enum):
        CREATED = 1
        RUNNING = 2
        CANCELLED = 3
        DONE = 4
        ERROR = 5

    def __init__(self, buffer: Buffer):
        self.buffer = Buffer
        self.status = Result.Status.CREATED

    def start(self):
        self.status = Result.Status.RUNNING
        pass

    def wait(self):
        self.status = Result.Status.DONE
        pass

    def get_statevector(self):
        self.wait()
        return self.circ.qreg.numpy()
