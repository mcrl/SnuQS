OPENQASM 2.0;


qreg q[5];
creg c[5];

gate custom_gate (x, y, z) a, b, c {
    U(x, y, xx) a;
}

