import cirq

from snuqs.simulators import StatevectorSimulator
# Pick a qubit.
qubit = cirq.GridQubit(0, 0)

# Create a circuit that applies a square root of NOT gate, then measures the qubit.
circuit = cirq.Circuit(cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m'))
print("Circuit:")
print(circuit)

# Simulate the circuit several times.
#simulator = cirq.Simulator()
simulator = StatevectorSimulator()
result = simulator.run(circuit, repetitions=20)
print("Results:")
print(type(result))
print(result)
