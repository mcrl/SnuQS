from .generated.QASMLexer import QASMLexer
from .generated.QASMListener import QASMListener
from .generated.QASMParser import QASMParser
from .qasm_exception import QASMPreprocessingException
from antlr4 import FileStream, CommonTokenStream
import snuqs
import os


class Preprocessor:
    def __init__(self):
        super().__init__()
        self.includes = []
        self.text = ""

    def get_path(self, fname):
        PATHS = [
                #snuqs.__path__,
            f"{snuqs.__path__[0]}/compilers",
        ]

        if fname[0] == '/':
            return fname

        else:
            for path in PATHS:
                target = f"{path}/{fname}"
                if os.path.exists(target):
                    return target
            raise QASMPreprocessingException(f"Cannot open file {fname}")

    def readFile(self, fname):
        path = self.get_path(fname)
        # FIXME: is correct??
        return open(path).read().replace('\n', '\r\n')

    def run(self, fname, out=None):

        tmp = ".tmp.qasm.out" if out is None else out
        open(tmp, 'w').write(self.readFile(fname))

        while True:
            self.text = self.readFile(tmp)
            input_stream = FileStream(tmp)
            lexer = QASMLexer(input_stream)
            stream = CommonTokenStream(lexer)
            idx = 1
            tok = stream.LT(idx)

            while tok.type != tok.EOF:

                if tok.text == 'include':
                    next_tok = stream.LT(idx+1)
                    nnext_tok = stream.LT(idx+2)
                    if len(next_tok.text) <= 3 or next_tok.text[0] != '"' or next_tok.text[-1] != '"' or nnext_tok.text != ';':
                        raise QASMPreprocessingException("Illegal include format")
                    self.includes.append((next_tok.text[1:-1], tok.start, nnext_tok.stop+1))
                    idx += 3
                else:
                    idx += 1
                tok = stream.LT(idx)

            if len(self.includes) == 0:
                break

            while len(self.includes) != 0:
                incl = self.includes.pop()

                self.text = self.text[:incl[1]] + self.readFile(incl[0]) + self.text[incl[2]:]
            open(tmp, 'w').write(self.text)

        return tmp
