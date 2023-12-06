import unittest
from snuqs import QasmCompiler, StatevectorSimulator


class GateTest(unittest.TestCase):
    def test_help(self):
        sim = StatevectorSimulator()
        result = sim.test()
        print(result)


if __name__ == '__main__':
    unittest.main()
