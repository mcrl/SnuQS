from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit
from braket.circuits.serialization import (
    IRType
)

bell = Circuit().h(0).cnot(0, 1)
print(bell.to_ir())
sim = LocalSimulator(backend="snuqs")
sim.run(bell, shots=5)
