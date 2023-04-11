import networkx as nx
import numpy as np
import time
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import Diagonal


def qaoa_maxcut(num_qubits, formula):
	circ = QuantumCircuit(num_qubits)

	for i in range(num_qubits):
		circ.h(i)

	alpha = Parameter("$\\alpha$")
	for i in range(num_qubits):
		circ.rx(2 * alpha, i)
	circ.barrier()

	beta = Parameter("$\\beta$")
	for i in range(num_qubits):
		circ.rx(-2 * beta, i)
	circ.barrier()

	gamma = Parameter("$\\gamma$")
	for a, b, c in list(formula):
		circ.rzz(2 * gamma, a, b)
		circ.rzz(2 * gamma, b, c)
		circ.rzz(2 * gamma, a, c)
	circ.barrier()

	return circ

def vqe(num_qubits, entanglement, depth):
	if entanglement == 'linear':
		return linear_vqe(num_qubits, depth)
	else:
		return full_vqe(num_qubits, depth)
	

if __name__ == "__main__":
	num_qubits = 4
	formula = [
	(0, 1, 2),
	(1, 2, 3),
	(0, 1, 3)
	]
	print(qaoa_maxcut(num_qubits, formula))
