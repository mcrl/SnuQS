OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[2];
cswap q[2], q[1], q[0];
//000
//001
//010
//011
//100
//101
//110
//111
