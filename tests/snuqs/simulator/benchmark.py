import unittest
from snuqs import QasmCompiler


class BenchmarkTest(unittest.TestCase):
    def test_qft(self):
        compiler = QasmCompiler()
        circ = compiler('qasm/qft.qasm')
        print(circ)


if __name__ == '__main__':
    unittest.main()
