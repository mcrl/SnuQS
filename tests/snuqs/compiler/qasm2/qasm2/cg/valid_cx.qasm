OPENQASM 2.0;
include "qelib1.inc";


qreg q[5];
qreg r[5];
creg c[5];

CX q[0], q[1];
CX q, r;
CX q[0], r;
