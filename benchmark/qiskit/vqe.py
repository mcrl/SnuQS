import numpy as np
import time
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Diagonal


def linear_vqe(num_qubits, depth):
	circ = QuantumCircuit(num_qubits)

	for d in range(depth):
		for q in range(num_qubits):
			circ.ry(np.random.rand() * 2 * np.pi, q)
			circ.rz(np.random.rand() * 2 * np.pi, q)

		for q in range(0, num_qubits-1):
			circ.cx(q, q+1)

		circ.barrier()
	return circ

def full_vqe(num_qubits, depth):
	circ = QuantumCircuit(num_qubits)

	for d in range(depth):
		for q in range(num_qubits):
			circ.ry(np.random.rand() * 2 * np.pi, q)
			circ.rz(np.random.rand() * 2 * np.pi, q)

		for q in range(0, num_qubits-1):
			for i in range(q+1, num_qubits):
				circ.cx(q, i)
		circ.barrier()

	return circ

def vqe(num_qubits, entanglement, depth):
	if entanglement == 'linear':
		return linear_vqe(num_qubits, depth)
	else:
		return full_vqe(num_qubits, depth)
	

if __name__ == "__main__":
	num_qubits = 5
	depth = 2
	np.random.seed(12345)
	print(vqe(num_qubits, 'linear', depth))
	print(vqe(num_qubits, 'full', depth))

	print(vqe(num_qubits, 'linear', depth).qasm())
	print(vqe(num_qubits, 'full', depth).qasm())
