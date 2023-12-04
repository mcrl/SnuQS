import unittest

from snuqs.compiler.qasm_compiler import QasmPreprocessor


class QasmPreprocessorTest(unittest.TestCase):
    def test_invalid_file_name(self):
        with open("qasm/preprocessor/invalid_file_name.qasm") as f:
            qasm = f.read()
        pp = QasmPreprocessor()
        self.assertRaises(FileNotFoundError, lambda: pp.preprocess(qasm))

    def test_invalid_include_syntax(self):
        with open("qasm/preprocessor/invalid_include_syntax.qasm") as f:
            qasm = f.read()
        pp = QasmPreprocessor()
        self.assertRaises(SyntaxError, lambda: pp.preprocess(qasm))

    def test_invalid_default_include(self):
        with open("qasm/preprocessor/invalid_default_include.qasm") as f:
            qasm = f.read()
        pp = QasmPreprocessor()
        self.assertRaises(FileNotFoundError, lambda: pp.preprocess(qasm))

    def test_invalid_version(self):
        with open("qasm/preprocessor/invalid_version.qasm") as f:
            qasm = f.read()
        pp = QasmPreprocessor()
        self.assertRaises(NotImplementedError, lambda: pp.preprocess(qasm))

    def test_qelib1(self):
        with open("qasm/preprocessor/qelib1.qasm") as f:
            qasm = f.read()
        pp = QasmPreprocessor()
        pp.preprocess(qasm)


if __name__ == '__main__':
    unittest.main()
