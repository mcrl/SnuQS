OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
u3(pi/8, pi/16, pi/32) q[0];
