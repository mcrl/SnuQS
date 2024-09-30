from braket.devices import LocalSimulator
from braket.circuits.circuit import Circuit

with open("ghz.qasm", "r") as ghz:
    ghz_qasm_string = ghz.read()

from braket.ir.openqasm import Program

program = Program(source=ghz_qasm_string)
circ = Circuit.from_ir(program)
print(circ)

sim = LocalSimulator(backend='snuqs')
task = sim.run(circ)
