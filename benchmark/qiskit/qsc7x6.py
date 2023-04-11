from qiskit import QuantumCircuit, execute
from qiskit import Aer


from functools import reduce
import random
import math


#patterns = [
#	[ (2, 3), (6, 7), (10, 11), (14, 15), (18, 19), (22, 23), (26, 27), (30, 31), (34, 35), (38, 39) ],
#	[ (0, 1), (4, 5), (8, 9), (12, 13), (16, 17), (20, 21), (24, 25), (28, 29), (32, 33), (36, 37), (40, 41) ],
#	[ (7, 13), (9, 15), (11, 17), (18, 24), (20, 26), (22, 28), (31, 37), (33, 39), (35, 41) ],
#	[ (6, 12),  (8, 14), (10, 16), (19, 25), (21, 27), (23, 29), (30, 36), (32, 38), (34, 40) ],
#	[ (3, 4), (7, 8), (15, 16), (19, 20), (27, 28), (31, 32), (39, 40) ],
#	[ (1, 2), (9, 10), (13, 14), (21, 22), (25, 26), (33, 34), (37, 38) ],
#	[ (0, 6), (2, 8), (4, 10), (13, 19), (15, 21), (17, 23), (24, 30), (26, 32), (28, 34) ],
#	[ (1, 7), (3, 9), (5, 11), (12, 18), (14, 20), (16, 22), (25, 31), (27, 33), (29, 35) ],
#]

patterns = [
	[
		((0, 2), (0, 3)),
		((1, 0), (1, 1)), ((1, 4), (1, 5)),
		((2, 2), (2, 3)),
		((3, 0), (3, 1)), ((3, 4), (3, 5)),
		((4, 2), (4, 3)),
		((5, 0), (5, 1)), ((5, 4), (5, 5)),
		((6, 2), (6, 3)),
	],
	[
		((0, 0), (0, 1)), ((0, 4), (0, 5)),
		((1, 2), (1, 3)),
		((2, 0), (2, 1)), ((2, 4), (2, 5)),
		((3, 2), (3, 3)),
		((4, 0), (4, 1)), ((4, 4), (4, 5)),
		((5, 2), (5, 3)),
		((6, 0), (6, 1)), ((6, 4), (6, 5)),
	],
	[
		((1,1), (2,1)), ((1,3), (2,3)), ((1,5), (2,5)),
		((3,0), (4,0)), ((3,2), (4,2)), ((3,4), (4,4)), ((3,6), (4,6)),
		((5,1), (6,1)), ((5,3), (6,3)), ((5,5), (6,5)),
	],
	[
		((1,0), (2,0)), ((1,2), (2,2)), ((1,4), (2,4)), ((1,6), (2,6)),
		((3,1), (4,1)), ((3,3), (4,3)), ((3,5), (4,5)), 
		((5,0), (6,0)), ((5,2), (6,2)), ((5,4), (6,4)), ((5,6), (6,6)),
	],
	[
		((0,3), (0,4)),
		((1,1), (1,2)), ((1,5), (1,6)),
		((2,3), (2,4)),
		((3,1), (3,2)), ((3,5), (3,6)),
		((4,3), (4,4)),
		((5,1), (5,2)), ((5,5), (5,6)),
		((6,3), (6,4)),
	],
	[
		((0,1), (0,2)), ((0,5), (0,6)),
		((1,3), (1,4)),
		((2,1), (2,2)), ((2,5), (2,6)),
		((3,3), (3,4)),
		((4,1), (4,2)), ((4,5), (5,6)),
		((5,3), (5,4)),
		((6,1), (6,2)), ((6,5), (6,6)),
	],
	[
		((0,0), (1,0)), ((0,2), (1,2)), ((0,4), (1,4)), ((0,6), (1,6)),
		((2,1), (3,1)), ((2,3), (3,3)), ((2,5), (3,5)),
		((4,0), (5,0)), ((4,2), (5,2)), ((4,4), (5,4)), ((4,6), (5,6)),
	],
	[
		((0,1), (1,1)), ((0,3), (1,3)), ((0,5), (1,5)),
		((2,0), (3,0)), ((2,2), (3,2)), ((2,4), (3,4)), ((2,6), (3,6)),
		((4,1), (5,1)), ((4,3), (5,3)), ((4,5), (5,5)),
	],
]

class QSC:
	def __init__(self, row, col, cycle):
		self.row = row
		self.col = col
		self.num_qubits = row * col
		self.cycle = cycle
		self.circ = QuantumCircuit(self.num_qubits, self.num_qubits)
		self.num_gate = 0

		self.GATES = ["T", "SQRT_X", "SY"]

		for i in range(self.num_qubits):
			self.circ.h(i)
		self.cur = ["H"] * self.num_qubits

		for c in range(cycle):
			print()
			print("# Cycle", c+1)
			self.applyCZGates(c)
			self.applyOneQubitGates(c)

		for i in range(self.num_qubits):
			self.circ.h(i)

	def getIndex(self, tup):
		return tup[0] * self.col + tup[1]
		

	def applyCZGates(self, c):
		for a, b in patterns[(c % 8)]:
			i = self.getIndex(a)
			j = self.getIndex(b)
			print("CZ", i, j)
			self.circ.cz(i, j)

	def applyOneQubitGates(self, c):
		prev_cz_qubits = [] if c == 0 else reduce(lambda x, y: list(x) + list(y), patterns[(c-1)%8])
		cz_qubits = reduce(lambda x, y: list(x) + list(y), patterns[c%8])

		for row in range(self.row):
			for col in range(self.col):
				if (row, col) not in patterns[c]:
					i = self.getIndex((row, col))
					if self.cur[i] == "H":
						self.circ.t(i)
						print("T", i)
						self.cur[i] = "T"
					else:
						gate = self.cur[i]
						while gate == self.cur[i]:
							gate = self.GATES[random.randint(0, 2)]
						if gate == "T":
							print("T", i)
							self.circ.t(i)
						elif gate == "SX":
							print("SX", i)
							self.circ.rx(math.pi/2, i)
						else:
							print("SY", i)
							self.circ.ry(math.pi/2, i)
						self.cur[i] = gate

circ = QSC(7, 7, 8)
print(circ.circ)
print(circ.circ.qasm())
#print(lat.qasm())
