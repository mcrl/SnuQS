OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
h q[4];
c4x q[4], q[3], q[2], q[1], q[0];
