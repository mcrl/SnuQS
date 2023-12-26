import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmPreprocessorTest(unittest.TestCase):
    def test_invalid_file_name(self):
        file_name = "qasm/preprocessor/invalid_file_name.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: QasmCompiler().compile(file_name))

    def test_invalid_default_include(self):
        file_name = "qasm/preprocessor/invalid_default_include.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: QasmCompiler().compile(file_name))

    def test_invalid_version(self):
        file_name = "qasm/preprocessor/invalid_version.qasm"
        self.assertRaises(NotImplementedError,
                          lambda: QasmCompiler().compile(file_name))

    def test_valid_include(self):
        file_name = "qasm/preprocessor/valid_include.qasm"
        QasmCompiler().compile(file_name)


if __name__ == '__main__':
    unittest.main()
