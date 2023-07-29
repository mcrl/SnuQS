from snuqs.result import Result
from enum import Enum
import qiskit
import sys
import time


class JobStatus(Enum):
    INITIALIZING = 1
    QUEUED = 2
    VALIDATING = 3
    RUNNING = 4
    CANCELLED = 5
    DONE = 6
    ERROR = 7

class Job:
    def __init__(self, sim, circ, qreg, creg, **kwargs):
        self.sim = sim
        self.circ = circ
        self.qreg = qreg
        self.creg = creg
        self.kwargs = kwargs

        self.status = JobStatus.INITIALIZING
        self._result = Result(qreg, creg)

    def cancel(self):
        if self.status != JobStatus.QUEUED and self.status != JobStatus.VALIDATING and self.status != JobStatus.RUNNING:
            raise Exception(f"Job cannot be cancled, in state {self.status}")

        self.status = JobStatus.CANCELLED

    def cancelled(self):
        return self.status == JobStatus.CANCELLED

    def done(self):
        return self.status == JobStatus.DONE

    def in_final_state(self):
        return self.status == JobStatus.DONE or self.status == JobStatus.ERROR

    def job_id(self):
        pass

    def result(self):
        self.wait_for_final_state()
        if self.status == JobStatus.ERROR:
            sys.exit(1)
        return self._result

    def running(self):
        return self.status == JobStatus.RUNNING

    def submit(self):
        pass

    def status(self):
        return self.status

    def wait_for_final_state(self):
        while not self.in_final_state():
            time.sleep(0.01)

        self.th.join()

class QiskitJob(qiskit.providers.job.JobV1):
    def __init__(self, job):
        self.job = job

    def cancel(self):
        self.job.cancel()

    def cancelled(self):
        return self.job.cancelled()

    def done(self):
        return self.job.done()

    def in_final_state(self):
        return self.job.in_final_state()

    def job_id(self):
        return self.job.job_id()

    def result(self):
        return self.job.result()

    def running(self):
        return selfe.job.running()

    def submit(self):
        self.job.submit()

    def status(self):
        return self.job.status()

    def wait_for_final_state(self):
        self.job.wait_for_final_state()
