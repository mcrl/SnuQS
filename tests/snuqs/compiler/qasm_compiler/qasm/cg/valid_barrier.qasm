OPENQASM 2.0;


qreg q[5];
qreg r[5];
creg c[5];

barrier q[0];
barrier q;
barrier r, q[1], q[0];
