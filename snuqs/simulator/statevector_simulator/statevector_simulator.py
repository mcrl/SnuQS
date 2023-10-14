from snuqs.simulator import BaseSimulator
from snuqs.request import Request
from abc import *

class StatevectorSimulator(BaseSimulator):
    def __init__(self, method: str='cuda'):
        self.impl = StatevectorSimulatorImpl()

    @abstractmethod
    def _run(self, req: Request):
        self.impl.run(req.circ, req.kwargs)
