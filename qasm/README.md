# How to generate QASM parser

## Installation

```
pip install -r requirements.txt
```

## Generation
```
# May take a few minutes
antlr4 -DLanguage=Python3 QASM.g4 -o generated/
```
