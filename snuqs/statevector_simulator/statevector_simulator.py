import snuqs._C
from snuqs.circuit import Circuit
from snuqs.result import Result
import threading
from typing import Dict

class StatevectorSimulator:
    def __init__(self):
        self.sim = snuqs._C.StatevectorSimulator()

    def _run(self, circ: Circuit, ret: Dict[str, any]):
        ret['state'] = self.sim.run(circ)

    def run(self, circ: Circuit):
        ret = {}
        return Result(threading.Thread(target=self._run, args=[circ, ret]), ret)

    def test(self):
        self.sim.test()
