#!/bin/sh
g++ -O3 -Wall -shared -std=c++17 -I../src -I../src/compiler -fPIC $(python3 -m pybind11 --includes) _snuqs.cpp -o _snuqs$(python3-config --extension-suffix) -L../build/lib.linux-x86_64-cpython-310/ -lsnuqs
