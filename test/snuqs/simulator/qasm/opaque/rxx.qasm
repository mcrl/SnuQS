OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[1];
rxx(pi/8) q[1], q[0];
