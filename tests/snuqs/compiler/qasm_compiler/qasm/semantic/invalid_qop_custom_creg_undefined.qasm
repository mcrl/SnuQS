OPENQASM 2.0;


qreg q[5];
qreg c[5];

gate custom1(x, y, z) a, b, c {
    CX a, b;
    CX b, c;
    CX c, a;

    x a;
    y b;
    z c;
}

custom1(d, 1, pi) q[0], q[2], q[1];
