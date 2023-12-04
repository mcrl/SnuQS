OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];

opaque invalid a, b, c;

gate gg a, b, c {

}

invalid q[0], q[2], q[1];
