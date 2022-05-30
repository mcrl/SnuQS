# soqs_update

## Prerequisites

```bash
$ sudo apt install libboost-all-dev
$ export PATH=/usr/local/cuda/bin:$PATH
```

## How to run
0. To compile, 'make -j[n]'
1. To run, See the 'test.sh' file.
* e.g., 
'LD_LIBRARY_PATH=./antlr4/lib ./src/soqs --method=statevector --soqsl=examples/test.soqsl'

## Status
0. (Optional) To generate parser, install 'antlr4' system-wide and run gen-parser.sh.
1. Primitive gate optimization [O]
* src/gate.hpp src/gate_cpu.cpp src/gate_gpu.cu 
	- ID(q)
	- H(q)
	- X(q)
	- Y(q)
	- Z(q)
	- SX(q)
	- SY(q)
	- S(q)
	- SDG(q)
	- T(q)
	- TDG(q)
	- RX(theta)(q)
	- RY(theta)(q)
	- RZ(theta)(q)
	- U1(lambda)(q)
	- U2(phi, lambda)(q)
	- U3(theta, phi, lambda)(q)
	- SWAP(q, p)
	- CX(c, t)
	- CY(c, t)
	- CZ(q, p)
	- CH(c, t)
	- CRX(theta)(c, t)
	- CRY(theta)(c, t)
	- CRZ(theta)(c, t)
	- CU1(lambda)(q, p)
	- CU2(phi, lambda)(c, t)
	- CU3(theta, phi, lambda)(c, t)
	- For gate information, search google and check qiskit pages. e.g., search "qiskit cx" then see the qiskit cx gate page. (https://qiskit.org/documentation/stubs/qiskit.circuit.library.CXGate.html)

2. Gate optimization with compiler [@]
* See comp-opt branch.
* TODO: Optimize kernel 'gate_block'.
* Combine gate fusion & cache (shared memory) blocking -> gate fusion through shared memory (best performing).
* Determine best-performing optimization parameters.
* Compiler option: \_\_BLOCKING\_\_ (turn on/off cache blocking).

3. Runtime improvement [X]
* Design hierarchcal runtime system.
* TBD

4. Extension to cluster system [X]
* TBD
# SnuQS
