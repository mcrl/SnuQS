from antlr4 import ParseTreeWalker, ParserRuleContext
from .qasm_stage import QasmStage


class QasmWalker:
    def __init__(self):
        self.walker = ParseTreeWalker()

    def walk(self, stage: QasmStage, tree: ParserRuleContext):
        self.walker.walk(stage, tree)
