from qiskit.circuit.random import random_circuit
from optparse import OptionParser

from snuqs.compilers.qasm_compiler import QASMSemanticException
import numpy as np
import sys
import time

def run_snuqs(qc):
    from snuqs.simulators import StatevectorSimulator
    from snuqs.compilers import QASMCompiler

    qasm_compiler = QASMCompiler()
    simulator = StatevectorSimulator()
    circ, qreg, creg = qasm_compiler.compile(QASMCompiler.to_qasm(qc))

    s = time.time()
    job = simulator.run(qc, circ, qreg, creg, method="cuda")
    result = job.result()
    print("[SnuQS] Execution time: ", time.time() - s)
    return result.get_statevector()

def run_aer(qc):
    from qiskit import transpile, Aer, QuantumCircuit
    from qiskit.providers.aer import AerSimulator
    #np.random.seed(0)

    simulator = AerSimulator(method='statevector')
    qc = transpile(qc, simulator)
    qc.save_statevector()
    s = time.time()
    job = simulator.run(qc)
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
            result0 = run_snuqs(circ)

            result1 = run_aer(circ)
        except QASMSemanticException as e:
            continue
        except Exception as e:
            print("HERE")
            print(e)
            sys.exit(1)

        print(result1[0])
        i0 = np.argmax(np.abs(result0) * np.abs(result1))
        s0 = result0[i0]
        s1 = result1[i0]
        gp = (s1 / s0)
        gp = gp / np.abs(gp)
        result0 = result0 * gp
        error = sum(np.square(np.abs(result0 - result1)))

        if not np.isclose(error, 0):
            print(result0)
            print(result1)
            print(i, ": Error: ", error)
            print(circ.qasm())
        else:
            #print(circ.qasm())
            print(i, ": Success")

        i += 1
        if i == niter:
            break
