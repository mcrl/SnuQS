# How to generate QASM parser

## Installation

```
pip install -r requirements.txt
```

## Generation
```
# May take a few minutes
# 1. for OpenQASM2
antlr4 -Dlanguage=Python3 qasm2.g4 -o ../snuqs/qasm2/compiler/generated/

# 2. for OpenQASM3
antlr4 -Dlanguage=Python3 qasm3Lexer.g4 -o ../snuqs/qasm3/compiler/generated/
antlr4 -Dlanguage=Python3 qasm3Parser.g4 -o ../snuqs/qasm3/compiler/generated/
```
