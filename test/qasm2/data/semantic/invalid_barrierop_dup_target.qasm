OPENQASM 2.0;


qreg q[5];
qreg r[5];

gate custom_gate (x, y, z) a, b, c {
    barrier a, b, b;
}

