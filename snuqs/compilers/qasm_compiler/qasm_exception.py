from snuqs.exception import BaseException

import sys

def red(msg):
    return f"\033[91m[{msg}\033[0m"

class QASMException(BaseException):
    def __init__(self, msg):
        super().__init__(msg)

class QASMPreprocessingException(QASMException):
    def __init__(self, msg):
        super().__init__(f"[Preprocessing Exception] {msg}")

class QASMParsingException(QASMException):
    def __init__(self, msg):
        super().__init__(f"[Parsing Exception] {msg}")

class QASMSemanticException(QASMException):
    def __init__(self, msg):
        super().__init__(f"[Semantic Exception] {msg}")

class QASMCGException(QASMException):
    def __init__(self, msg):
        super().__init__(f"[CG Exception] {msg}")
