from .base_simulator import BaseSimulator
from snuqs.result import Result
from snuqs.utils import logger

class StatevectorSimulator(BaseSimulator):
    def __init__(self):
        pass

    def _run(self, qasm, **kwargs):
        logger.info("Starting simulation...")

        print(type(qasm))
        print(qasm)

        return Result('qiskit')
