import unittest

from snuqs.compiler.qasm2 import Qasm2Compiler


class QasmPreprocessorTest(unittest.TestCase):
    def test_invalid_file_name(self):
        file_name = "qasm2/preprocessor/invalid_file_name.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: Qasm2Compiler().compile(file_name))

    def test_invalid_default_include(self):
        file_name = "qasm2/preprocessor/invalid_default_include.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: Qasm2Compiler().compile(file_name))

    def test_invalid_version(self):
        file_name = "qasm2/preprocessor/invalid_version.qasm"
        self.assertRaises(NotImplementedError,
                          lambda: Qasm2Compiler().compile(file_name))

    def test_valid_include(self):
        file_name = "qasm2/preprocessor/valid_include.qasm"
        Qasm2Compiler().compile(file_name)


if __name__ == '__main__':
    unittest.main()
