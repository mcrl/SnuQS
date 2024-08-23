OPENQASM 2.0;


qreg q[5];

gate custom1(x, y, z) a, b, c {
    CX a, b;
    CX b, c;
    CX c, a;

    x a;
    y b;
    z c;
}


gate custom2(x, y, z) a, b, c {
    custom1 a, b, b;
}
