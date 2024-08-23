from enum import Enum
from snuqs.buffer import Buffer

from typing import Dict
import threading


class Result:
    class Status(Enum):
        CREATED = 1
        RUNNING = 2
        DONE = 3

    def __init__(self, thread: threading.Thread, ret: Dict[str, any]):
        self.ret = ret
        self.thread = thread
        self.status = Result.Status.CREATED
        self.start()

    def start(self):
        if self.status == Result.Status.CREATED:
            self.status = Result.Status.RUNNING
            self.thread.start()

    def wait(self):
        if self.status != Result.Status.DONE:
            self.thread.join()
            self.status = Result.Status.DONE

    def get_statevector(self):
        self.wait()
        return self.ret['state']

    def __repr__(self):
        rp = f'Result <{self.status}>'
        return rp
