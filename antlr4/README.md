# How to generate QASM parser

## Installation

```
pip install -r requirements.txt
```

## Generation
```
# May take a few minutes
antlr4 -Dlanguage=Python3 QASM.g4 -o generated/
antlr4 -Dlanguage=Python3 QASM.g4 -o ../snuqs/compiler/qasm_compiler/generated/
antlr4 -Dlanguage=C++ QASM.g4 -o ../src/compiler/qasm_compiler/generated/
```
