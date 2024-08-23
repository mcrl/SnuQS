import unittest
from snuqs.qasm2.compiler import Parser


class ParserTest(unittest.TestCase):
    def test_invalid_include_syntax(self):
        file_name = "data/parser/invalid_include_syntax.qasm"
        self.assertRaises(SyntaxError, lambda: Parser().parse(file_name))

    def test_invalid_file_name(self):
        file_name = "data/parser/invalid_file_name.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: Parser().parse(file_name))

    def test_invalid_default_include(self):
        file_name = "data/parser/invalid_default_include.qasm"
        self.assertRaises(FileNotFoundError,
                          lambda: Parser().parse(file_name))

    def test_valid_include(self):
        file_name = "data/parser/valid_include.qasm"
        pp = Parser()
        pp.parse(file_name)


if __name__ == '__main__':
    unittest.main()
