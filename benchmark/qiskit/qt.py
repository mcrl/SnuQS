import numpy as np
# importing Qiskit
from qiskit import QuantumCircuit

def qt():
	circ = QuantumCircuit(3, 2)

	circ.h(1)
	circ.cz(1, 2)
	circ.barrier()
	circ.cz(0, 1)
	circ.h(0)

	circ.barrier()
	circ.measure(0, 0)
	circ.measure(1, 1)


	circ.barrier()
	circ.x(2).c_if(1, 1)
	circ.z(2).c_if(0, 1)

	return circ;


if __name__ == "__main__":
	print(qt())
