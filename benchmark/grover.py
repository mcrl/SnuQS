import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Diagonal


def grover(num_qubits, targets, num_iter):
	qr = QuantumRegister(num_qubits, 'q')
	circ = QuantumCircuit(qr)

	for q in range(num_qubits):
		circ.h(q)

	np_oracle = np.ones(2**num_qubits)
	for target in targets:
		np_oracle[target] = -np_oracle[target]
	oracle = Diagonal(np_oracle)

	diffuser = QuantumCircuit(num_qubits, name='diffuser')

	for q in range(num_qubits):
		diffuser.h(q)
	for q in range(num_qubits):
		diffuser.x(q) 

	diffuser.h(num_qubits-1)
	diffuser.mct(list(range(num_qubits-1)), num_qubits-1)
	diffuser.h(num_qubits-1)

	for q in range(num_qubits):
		diffuser.x(q)
	for q in range(num_qubits):
		diffuser.h(q)

	for it in range(num_iter):
		circ.append(oracle, qr)
		circ.append(diffuser, qr)

	
	return circ;

if __name__ == "__main__":
	num_qubits = 4
	targets = [0, (2**num_qubits-1)]
	num_iter = 1
	print(grover(num_qubits, targets, num_iter))
	print(grover(num_qubits, targets, num_iter).qasm())
