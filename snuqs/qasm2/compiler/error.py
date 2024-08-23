class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CompileError:
    def __init__(self, file_name: str, line: int, column: int, msg: str):
        self.file_name = file_name
        self.line = line
        self.column = column
        self.msg = msg

    def __str__(self):
        notice = f"{bcolors.FAIL}error{bcolors.ENDC}"
        return f"{self.file_name}:{self.line}:{self.column} {notice}: {self.msg}"
