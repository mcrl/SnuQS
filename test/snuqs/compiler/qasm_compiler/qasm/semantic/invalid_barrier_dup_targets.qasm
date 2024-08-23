OPENQASM 2.0;
qreg q[5];
qreg r[1];

barrier r, q, q[0];
