# How to generate Qasm2 parser

## Installation

```
pip install -r requirements.txt
```

## Generation
```
# May take a few minutes
antlr4 -Dlanguage=Python3 Qasm2.g4 -o ../../snuqs/compiler/qasm2/generated/
```
