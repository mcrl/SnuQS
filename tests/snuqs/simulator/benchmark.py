import unittest
from snuqs import QasmCompiler, StatevectorSimulator


class BenchmarkTest(unittest.TestCase):
    def test_qft(self):
        compiler = QasmCompiler()
        circ = compiler.compile('qasm/qft.qasm')
        sim = StatevectorSimulator(device=StatevectorSimulator.CUDA)
        print(circ)


if __name__ == '__main__':
    unittest.main()
