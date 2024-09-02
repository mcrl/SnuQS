from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit

bell = Circuit().h(0).cnot(0, 1)
sim = LocalSimulator(backend="snuqs")
sim.run(bell, shots=5)
