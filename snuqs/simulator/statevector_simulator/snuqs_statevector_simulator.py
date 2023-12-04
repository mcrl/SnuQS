from snuqs.circuit import Circuit
from snuqs.simulator.statevector_simulator import StatevectorSimulatorImpl

class SnuQSStatevectorSimulator(StatevectorSimulatorImpl):
    def __init__(self):
        super().__init__()
        self.launcher = snuqs.impl.Launcher()

    def run(self, circ: Circuit, **kwargs):
        self.launcher.run(circ)
