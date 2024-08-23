from qiskit import qasm2
from qiskit.circuit.random import random_circuit
from optparse import OptionParser

import numpy as np
import sys
import time
from io import StringIO

def run_snuqs(fname):
    from snuqs import StatevectorSimulator
    from snuqs import QasmCompiler

    qasm_compiler = QasmCompiler()
    simulator = StatevectorSimulator()
    circ = qasm_compiler.compile(fname)


    s = time.time()
    result = simulator.run(circ)
    print("[SnuQS] Execution time: ", time.time() - s)
    sv = result.get_statevector()
    sv = np.array([sv[i] for i in range(len(sv))])
    return sv

def run_aer(fname):
    from qiskit import transpile, QuantumCircuit
    from qiskit_aer import AerSimulator
    #np.random.seed(0)

    simulator = AerSimulator()
    qc = qasm2.load(fname)
    qc = transpile(qc, simulator)
    qc.save_statevector()
    s = time.time()
    job = simulator.run(qc, device="CPU")
    result = job.result()
    print("[Aer] Execution time: ", time.time() - s)
    return result.get_statevector().data

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-q", "--nqubits", dest="nqubits")
    parser.add_option("-d", "--depth", dest="depth")
    parser.add_option("-i", "--niter", dest="niter")
    options, args = parser.parse_args()

    nqubits = int(options.nqubits) if options.nqubits else 5
    depth = int(options.depth) if options.depth else 5
    niter = int(options.niter) if options.niter else 5


    print("nqubits: ", nqubits)
    print("depth: ", depth)
    print("niter: ", niter)

    i = 0
    while True:
        try:
            circ = random_circuit(nqubits, depth, measure=False, conditional=False, reset=False)
            fname = "/tmp/dypshong"
            qasm2.dump(circ, fname)

            result1 = run_aer(fname)
            result0 = run_snuqs(fname)
        except Exception as e:
            continue

        i0 = np.argmax(np.abs(result0) * np.abs(result1))
        s0 = result0[i0]
        s1 = result1[i0]
        gp = (s1 / s0)
        gp = gp / np.abs(gp)
        result0 = result0 * gp
        error = sum(np.square(np.abs(result0 - result1)))

        if not np.isclose(error, 0):
            print(i, ": Error: ", error)
            print(open("/tmp/dypshong").read())
        else:
            print(i, ": Success")

        i += 1
        if i == niter:
            break
