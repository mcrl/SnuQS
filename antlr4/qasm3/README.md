# How to generate QASM parser

## Installation

```
pip install -r requirements.txt
```

## Generation
```
# May take a few minutes
antlr4 -Dlanguage=Python3 Qasm3Lexer.g4 -o ../../snuqs/compiler/qasm3/generated/
antlr4 -Dlanguage=Python3 Qasm3Parser.g4 -o ../../snuqs/compiler/qasm3/generated/
```
