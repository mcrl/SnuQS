from .base_statevector_simulator import BaseStatevectorSimulator
import numpy as np
import sys
import snuqs
from snuqs.circuit import QopType
from snuqs.job import Job, JobStatus

import time
from enum import Enum

class ExecutionMethod(Enum):
    OMP = 0
    CUDA = 1
    OMP_IO = 2
    CUDA_IO = 3

class SnuQSStatevectorSimulator(BaseStatevectorSimulator):
    default_method = ExecutionMethod.OMP

    def __init__(self):
        super().__init__()
        self.executor = snuqs.cpp.Executor()

    def _exec(self, job):
        if 'method' in job.kwargs:
            if job.kwargs['method'] == "omp":
                method = ExecutionMethod.OMP
            elif job.kwargs['method'] == "cuda":
                method = ExecutionMethod.CUDA
            elif job.kwargs['method'] == "omp:io":
                method = ExecutionMethod.IO_OMP
            elif job.kwargs['method'] == "cuda:io":
                method = ExecutionMethod.IO_CUDA
            else:
                print("Unknown method")
                sys.exit(1)
        else:
            method = default_method

        paths = job.kwarg['paths'] if 'paths' in job.kwargs else None

        self.executor.run(job.circ, job.qreg, job.creg, method)
