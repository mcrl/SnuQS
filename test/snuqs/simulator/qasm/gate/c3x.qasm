OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[3];
c3x q[3], q[2], q[1], q[0];
