from snuqs.quantum_circuit import QuantumCircuit
from snuqs.simulator.statevector_simulator import StatevectorSimulatorImpl

class SnuQSStatevectorSimulator(StatevectorSimulatorImpl):
    def __init__(self):
        super().__init__()
        self.launcher = snuqs.impl.Launcher()

    def run(self, circ: QuantumCircuit, **kwargs):
        self.launcher.run(circ)
