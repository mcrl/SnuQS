#!/bin/sh

#java -jar antlr-4.12.0-complete.jar -Dlanguage=Cpp snuqasm.g4 -o ../src/compiler/snuqasm-parser
java -jar antlr-4.12.0-complete.jar -Dlanguage=Python3 QASM.g4  -o ../python/snuqs/compilers/qasm_compiler/generated
