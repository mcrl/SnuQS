import unittest

from snuqs.qasm3.compiler import QasmPreprocessor


class QasmPreprocessorTest(unittest.TestCase):
    def test_invalid_file_name(self):
        file_name = "qasm/preprocessor/invalid_file_name.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: QasmPreprocessor().preprocess(file_name))

    def test_invalid_include_syntax(self):
        file_name = "qasm/preprocessor/invalid_include_syntax.qasm"
        self.assertRaises(SyntaxError,
                          lambda: QasmPreprocessor().preprocess(file_name))

    def test_invalid_default_include(self):
        file_name = "qasm/preprocessor/invalid_default_include.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: QasmPreprocessor().preprocess(file_name))

    def test_invalid_version(self):
        file_name = "qasm/preprocessor/invalid_version.qasm"
        self.assertRaises(NotImplementedError,
                          lambda: QasmPreprocessor().preprocess(file_name))

    def test_valid_include(self):
        file_name = "qasm/preprocessor/valid_include.qasm"
        QasmPreprocessor().preprocess(file_name)


if __name__ == '__main__':
    unittest.main()
