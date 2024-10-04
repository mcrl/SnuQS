import numpy as np

from braket.devices import LocalSimulator
from braket.circuits import Circuit, ResultType

def hellinger_fidelity(ideal_dist, device_dist):
    fidelity = 0.0
    for bitstr in ideal_dist:
        p_i = ideal_dist.get(bitstr, 0)
        q_i = device_dist.get(bitstr, 0)
        fidelity += (np.sqrt(p_i) * np.sqrt(q_i))

    return fidelity**2

def remove_measurements(circuit):
    # Create a new circuit without measurement gates
    removed_circuit = Circuit()
    for instruction in circuit.instructions:
        if instruction.operator.name != "Measure":
            removed_circuit.add_instruction(instruction)
    return removed_circuit

def get_ideal_counts(circuit: Circuit, backend) -> dict[str, float]:
    """
    Noiseless statevector simulation.

    Note that the qubits in the returned bitstrings are in big-endian order.
    For example, for a circuit defined on qubits
        q0 ------
        q1 ------
        q2 ------
    the bitstrings are written as `q0q1q2`.

    Args:
        circuit: Input `cirq.Circuit` to be simulated.

    Returns:
        A dictionary with bitstring and probability as the key, value pairs.
    """
    ideal_counts = {}
    removed_circuit = remove_measurements(circuit)
    removed_circuit.state_vector()

    if backend:
        sim = LocalSimulator(backend=backend)
    else:
        sim = LocalSimulator()
        
    task = sim.run(removed_circuit, shots=0)

    state_vector = task.result().get_value_by_result_type(ResultType.StateVector())

    for i, amplitude in enumerate(state_vector):
        bitstring = f"{i:>0{circuit.qubit_count}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return ideal_counts