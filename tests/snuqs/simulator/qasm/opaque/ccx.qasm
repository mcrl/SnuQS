OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[2];
ccx q[2], q[1], q[0];
