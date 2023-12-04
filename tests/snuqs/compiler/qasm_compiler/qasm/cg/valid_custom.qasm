OPENQASM 2.0;


qreg q[5];

gate custom(x1, y1, z1) a, b, c {
    x c;
    y b;
    z a;
    U(x1, y1, z1) a;
}

gate custom2(x, y, z) a, b, c {
    custom(x, y, z) c, b, a;
    U(z, 2*y, x*pi) c;
    u2(y, x) a;
}

x q[3];
custom2(3, 4, 2*pi) q[0], q[1], q[2];
