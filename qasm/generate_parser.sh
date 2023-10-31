#!/bin/sh

java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 QASM.g4  -o ./generated
