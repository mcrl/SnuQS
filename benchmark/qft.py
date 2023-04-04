import numpy as np
from qiskit import transpile, Aer, QuantumCircuit

import snuqs

def qft(num_qubits):
    circ = QuantumCircuit(num_qubits)

    for q in range(num_qubits):
        circ.h(q)
        for p in range(q+1, num_qubits):
            circ.cp(np.pi/2**(num_qubits-p), q, p) 

    for q in range(num_qubits//2):
          circ.swap(q, num_qubits-q-1)

    
    return circ;



if __name__ == "__main__":
    num_qubits = 4
    qc = qft(num_qubits)

    backend_sim = Aer.get_backend('qasm_simulator')
    # Execute the circuit on the qasm simulator.
    # We've set the number of repeats of the circuit
    # to be 1024, which is the default.
    qc = transpile(qc, backend_sim)
    #job_sim = backend_sim.run(qc, backend_sim), shots=1024)
    job_sim = snuqs.run(qc, shots=1024)

    print(type(job_sim))
    print(job_sim)

    # Grab the results from the job.
    result_sim = job_sim.result()
    print(result_sim)
