OPENQASM 2.0;
include "qelib1.inc";


qreg q[5];
qreg r[5];
creg c[5];

measure q[0] -> c[1];
measure q -> c;
measure r -> c;
