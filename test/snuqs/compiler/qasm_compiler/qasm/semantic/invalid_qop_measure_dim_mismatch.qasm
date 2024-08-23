OPENQASM 2.0;

qreg q[2];
creg c[1];

measure q -> c;
measure q[0] -> c;


