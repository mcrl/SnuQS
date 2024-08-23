OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[5];

if (q == 5) x q[1];
