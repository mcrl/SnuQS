OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];

gate custom1 a, b, c {
    CX a, b;
    CX b, c;
    CX c, a;

    x a;
    y b;
    z c;
}

custom1 q[0], q[1], q[5];
