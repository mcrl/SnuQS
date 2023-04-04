import networkx as nx
import numpy as np
import time
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import Diagonal


def qaoa_maxcut(num_qubits, graph):
	circ = QuantumCircuit(num_qubits)

	for i in range(num_qubits):
		circ.h(i)

	beta = Parameter("$\\beta$")
	for i in range(num_qubits):
		circ.rx(2 * beta, i)
	circ.barrier()

	gamma = Parameter("$\\gamma$")
	for a, b in list(graph.edges()):
		circ.rzz(2 * gamma, a, b)
	circ.barrier()

	return circ

def vqe(num_qubits, entanglement, depth):
	if entanglement == 'linear':
		return linear_vqe(num_qubits, depth)
	else:
		return full_vqe(num_qubits, depth)
	

if __name__ == "__main__":
	num_qubits = 4
	graph = nx.Graph()
	graph.add_nodes_from([0, 1, 2, 3])
	graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
	print(qaoa_maxcut(num_qubits, graph))
