OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

gate custom(x, y, z) a, b, c {
    CX a, c;
    CX b, d;
    U(x+1, y+2) a;
}
