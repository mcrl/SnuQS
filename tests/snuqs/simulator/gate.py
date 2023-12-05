import unittest
from snuqs import QasmCompiler, StatevectorSimulator


class GateTest(unittest.TestCase):
    def test_h(self):
        compiler = QasmCompiler()
        circ = compiler.compile('qasm/gate/h.qasm')
        sim = StatevectorSimulator()
        result = sim.run(circ)
        print(result)


if __name__ == '__main__':
    unittest.main()
