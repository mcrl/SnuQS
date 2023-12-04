from .generated.QASMLexer import QASMLexer
from antlr4 import InputStream, CommonTokenStream
import snuqs
import os


PREFIXES = [
    f"{snuqs.__path__[0]}/compiler/qasm_compiler",
]

DEFAULT_INCLUDES = [
    '__snuqs.inc',
]


class QasmPreprocessor:
    def read_file(self, file_name: str):
        path = ''
        if file_name[0] == '/' or file_name[0:2] == './':
            path = file_name
        else:
            for prefix in PREFIXES:
                target = f"{prefix}/{file_name}"
                if os.path.exists(target):
                    path = target
                    break

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{file_name}' does not exists for inclusion.")

        with open(path) as f:
            fs = f.read()

        return fs

    def _preprocess(self, qasm: str, idx: int):
        input_stream = InputStream(qasm)
        lexer = QASMLexer(input_stream)
        stream = CommonTokenStream(lexer)

        tok = stream.LT(idx)
        while tok.type != tok.EOF:
            tok = stream.LT(idx)
            if tok.text == 'include':
                next_tok = stream.LT(idx+1)
                nnext_tok = stream.LT(idx+2)

                file_name = next_tok.text[1:-1]
                tok_start = tok.start
                tok_end = nnext_tok.stop+1

                if len(next_tok.text) <= 1 or next_tok.text[0] != '"' or next_tok.text[-1] != '"' or nnext_tok.text != ';':
                    raise SyntaxError(
                        f"illegal include format: {qasm[tok_start:tok_end]}.")

                qasm = qasm[:tok_start] + \
                    self.read_file(file_name) + qasm[tok_end:]
                return self._preprocess(qasm, idx)
            else:
                idx += 1
        return qasm

    def preprocess(self, file_name: str, idx: int = 1):
        with open(file_name) as f:
            qasm = f.read()

        input_stream = InputStream(qasm)
        lexer = QASMLexer(input_stream)
        stream = CommonTokenStream(lexer)

        if stream.LT(1).text != 'OPENQASM' or stream.LT(2).text != '2.0' or stream.LT(3).text != ';':
            raise NotImplementedError(
                "Illegal OpenQASM file format: OpenQASM must start with OPENQASM 2.0;")
        tok = stream.LT(4)
        pre = qasm[:tok.start]
        post = qasm[tok.start:]
        includes = ''
        for file_name in DEFAULT_INCLUDES:
            includes = f'include "{file_name}";\n' + includes
        qasm = pre + includes + post
        return self._preprocess(qasm, 4)
