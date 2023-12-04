OPENQASM 2.0;


qreg q[5];
creg c[5];

U((1+pi)/(-2.3 * c), --2, 3 * c) q[4];
if (c == 5) U(1, 2, 3) q[3];
