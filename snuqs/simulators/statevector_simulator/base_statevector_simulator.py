from snuqs.simulators.base_simulator import BaseSimulator
from snuqs.job import Job, JobStatus
from snuqs.circuit.qop import QopType, Init, Fini

from abc import *
import threading
import time

class BaseStatevectorSimulator(BaseSimulator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _exec(self, job):
        pass

    def exec(self, job):
        job.status = JobStatus.RUNNING
        try:
            self._exec(job)
            job.status = JobStatus.DONE
        except Exception as e:
            print(e)
            job.status = JobStatus.ERROR

    def _run(self, circ, qreg, creg, **kwargs):
        circ.prepend_op(Init(QopType.Init, []))
        circ.push_op(Fini(QopType.Fini, []))
        job = Job(self, circ, qreg, creg, **kwargs)
        job.status = JobStatus.QUEUED

        job.th = threading.Thread(target=self.exec, args=[job])
        job.status = JobStatus.VALIDATING
        job.th.start()

        return job
