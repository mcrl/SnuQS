import unittest

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit


class BraketTest(unittest.TestCase):
#    def test_braket(self):
#        bell = Circuit()
#        bell = bell.h(0)
#        bell = bell.cnot(0, 1)
#        bell = bell.cnot(1, 2)
#        bell.state_vector()
#
#        sim = LocalSimulator()
#        task = sim.run(bell)
#        print(task.result())

    def test_braket_snuqs(self):
        bell = Circuit()
        bell = bell.h(0)
        bell = bell.cnot(0, 1)
        bell = bell.cnot(1, 2)
        bell.state_vector()

        sim = LocalSimulator(backend="snuqs")
        task = sim.run(bell)
        print(task.result())


if __name__ == '__main__':
    unittest.main()
