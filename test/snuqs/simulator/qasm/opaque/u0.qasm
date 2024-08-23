OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
u0(1) q[0];
