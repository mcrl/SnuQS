from antlr4 import ParseTreeWalker, ParserRuleContext
from .qasm2_stage import Qasm2Stage


class Qasm2Walker:
    def __init__(self):
        self.walker = ParseTreeWalker()

    def walk(self, stage: Qasm2Stage, tree: ParserRuleContext):
        self.walker.walk(stage, tree)
