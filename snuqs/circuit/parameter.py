from snuqs._C import Parameter, Identifier, BinOp, BinOpType, NegOp
from snuqs._C import UnaryOp, UnaryOpType, Parenthesis, Constant, Pi

#import snuqs._C as _C
#Parameter = _C.Parameter
#BinOp = _C.BinOp
#Identifier = _C.Identifier
#NegOp = _C.NegOp
#UnaryOp = _C.UnaryOp
#Parenthesis = _C.Parenthesis
#Constant = _C.Constant
#Pi = _C.Pi
#
#from abc import ABCMeta, abstractmethod
#from enum import Enum
#import math
#import operator
#from .reg import Creg
#
#
#class Parameter(metaclass=ABCMeta):
#    @abstractmethod
#    def eval(self):
#        pass
#
#
#class BinOp(Parameter):
#    class Type:
#        ADD = 1
#        SUB = 2
#        MUL = 3
#        DIV = 4
#
#    def __init__(self, op: Type, expr0: Parameter, expr1: Parameter):
#        self.op = op
#        self.expr0 = expr0
#        self.expr1 = expr1
#
#    def eval(self):
#        if self.op == BinOp.Type.ADD:
#            op = operator.add
#        elif self.op == BinOp.Type.SUB:
#            op = operator.sub
#        elif self.op == BinOp.Type.MUL:
#            op = operator.mul
#        elif self.op == BinOp.Type.DIV:
#            op = operator.div
#
#        return op(self.expr0.eval(), self.expr1.eval())
#
#    def dump(self, indent=0):
#        if self.op == BinOp.Type.ADD:
#            op = '+'
#        elif self.op == BinOp.Type.SUB:
#            op = '-'
#        elif self.op == BinOp.Type.MUL:
#            op = '*'
#        elif self.op == BinOp.Type.DIV:
#            op = '/'
#
#        rp_expr0 = self.expr0.dump(indent=indent+1)
#        rp_expr1 = self.expr1.dump(indent=indent+1)
#
#        ind = ' ' * (2*indent)
#        rp = f"{ind}BinOp <'{op}'>\n"
#        rp += f"{rp_expr0}\n"
#        rp += f"{rp_expr1}\n"
#        return rp
#
#    def __repr__(self):
#        if self.op == BinOp.Type.ADD:
#            op = '+'
#        elif self.op == BinOp.Type.SUB:
#            op = '-'
#        elif self.op == BinOp.Type.MUL:
#            op = '*'
#        elif self.op == BinOp.Type.DIV:
#            op = '/'
#
#        return self.expr0.__repr__() + op + self.expr1.__repr__()
#
#
#class Identifier(Parameter):
#    def __init__(self, creg: Creg):
#        self.creg = creg
#
#    def eval(self, creg: Creg):
#        # FIXME:
#        return self.creg
#
#    def dump(self, indent=0):
#        ind = ' ' * (2*indent)
#        return f"{ind}Identifier <'{self.creg.name}'>"
#
#    def __repr__(self):
#        return self.creg.__repr__()
#
#
#class NegOp(Parameter):
#    def __init__(self, expr: Parameter):
#        self.expr = expr
#
#    def eval(self):
#        return -self.expr.eval()
#
#    def dump(self, indent=0):
#        ind = ' ' * (2*indent)
#        rp_expr = self.expr.dump(indent=indent+1)
#        rp = f"{ind}NegOp <'-'>\n"
#        rp += f"{ind}{rp_expr}"
#        return rp
#
#    def __repr__(self):
#        return '-' + self.expr.__repr__()
#
#
#class UnaryOp(Parameter):
#    class Type(Enum):
#        SIN = 1
#        COS = 2
#        TAN = 3
#        EXP = 4
#        LN = 5
#        SQRT = 6
#
#    def __init__(self, op: Type, expr: Parameter):
#        self.op = op
#        self.expr = expr
#
#    def eval(self):
#        if self.op == UnaryOp.Type.SIN:
#            op = math.sin
#        elif self.op == UnaryOp.Type.COS:
#            op = math.cos
#        elif self.op == UnaryOp.Type.TAN:
#            op = math.tan
#        elif self.op == UnaryOp.Type.EXP:
#            op = math.exp
#        elif self.op == UnaryOp.Type.LN:
#            op = math.log
#        elif self.op == UnaryOp.Type.SQRT:
#            op = math.sqrt
#
#        return op(self.expr.eval())
#
#    def dump(self, indent=0):
#        if self.op == UnaryOp.Type.SIN:
#            op = 'sin'
#        elif self.op == UnaryOp.Type.COS:
#            op = 'cos'
#        elif self.op == UnaryOp.Type.TAN:
#            op = 'tan'
#        elif self.op == UnaryOp.Type.EXP:
#            op = 'exp'
#        elif self.op == UnaryOp.Type.LN:
#            op = 'log'
#        elif self.op == UnaryOp.Type.SQRT:
#            op = 'sqrt'
#
#        ind = ' ' * (2*indent)
#        rp_expr = self.expr.dump(indent=indent+1)
#        rp = f"{ind}UnaryOp <'{op}'>\n"
#        rp += f"{ind}{rp_expr}"
#        return rp
#
#    def __repr__(self):
#        if self.op == UnaryOp.Type.SIN:
#            op = 'sin'
#        elif self.op == UnaryOp.Type.COS:
#            op = 'cos'
#        elif self.op == UnaryOp.Type.TAN:
#            op = 'tan'
#        elif self.op == UnaryOp.Type.EXP:
#            op = 'exp'
#        elif self.op == UnaryOp.Type.LN:
#            op = 'log'
#        elif self.op == UnaryOp.Type.SQRT:
#            op = 'sqrt'
#
#        return op + "(" + self.expr.__repr__() + ")"
#
#
#class Parenthesis(Parameter):
#    def __init__(self, expr: Parameter):
#        self.expr = expr
#
#    def eval(self):
#        return self.expr.eval()
#
#    def dump(self, indent=0):
#        ind = ' ' * (2*indent)
#        rp_expr = self.expr.dump(indent=indent+1)
#        rp = f"{ind}Parenthesis <'()'>"
#        rp += f"{ind}{rp_expr}"
#        return rp
#
#    def __repr__(self):
#        return "(" + self.expr.__repr__() + ")"
#
#
#class Constant(Parameter):
#    def __init__(self, value: float):
#        self.value = value
#
#    def eval(self):
#        return self.value
#
#    def dump(self, indent=0):
#        ind = ' ' * (2*indent)
#        return f"{ind}Constant <'{self.value}'>"
#
#    def __repr__(self):
#        return str(self.value)
#
#
#class Pi(Constant):
#    def __init__(self):
#        super().__init__(math.pi)
#
#    def eval(self):
#        return math.pi
#
#    def dump(self, indent=0):
#        ind = ' ' * (2*indent)
#        return f"{ind}Pi <'pi'>"
#
#    def __repr__(self):
#        return 'pi'
