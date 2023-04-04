import numpy as np
from qiskit import QuantumCircuit


def qpe(num_qubits):
	circ = QuantumCircuit(num_qubits+1)

	for q in range(num_qubits):
		circ.h(q)

	circ.x(num_qubits)
	rep = 1
	for q in range(num_qubits):
		for i in range(rep):
			circ.cp(np.pi/4, q, num_qubits);
		rep *= 2
	

	circ.barrier()
	for q in range(num_qubits//2):
	 	 circ.swap(q, num_qubits-q-1)

	for q in range(num_qubits):
		for p in range(q+1, num_qubits):
			circ.cp(-np.pi/2**(num_qubits-p), q, p) 
		circ.h(q)

	
	return circ;


if __name__ == "__main__":
	num_qubits = 4
	print(qpe(num_qubits))
