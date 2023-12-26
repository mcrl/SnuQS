OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[1];
cu3(pi/8, pi/16, pi/32) q[1], q[0];
