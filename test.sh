#!/bin/sh

LD_PATH="./dep/antlr4/runtime/Cpp/run/usr/local/lib/:./dep/boost/stage/lib:./dep/liburing/src/"
#LD_LIBRARY_PATH=${LD_PATH} build/bin/snuqs --method=statevector --snuql=examples/qft_30.soqsl
#LD_LIBRARY_PATH=${LD_PATH} build/bin/snuqs --method=statevector --device=cpu --snuql=examples/qft_30.soqsl
#LD_LIBRARY_PATH=${LD_PATH} build/bin/snuqs --method=statevector --device=gpu --snuql=examples/qft_30.soqsl
#LD_LIBRARY_PATH=${LD_PATH} build/bin/snuqs
LD_LIBRARY_PATH=${LD_PATH} build/bin/socl-io-test
