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
    print(circ)

    s = time.time()
    result = simulator.run(circ)
    print("[SnuQS] Execution time: ", time.time() - s)
    sv = result.get_statevector()
    sv = np.array([sv[i] for i in range(len(sv))])
    return sv

def run_aer(fname):
    from qiskit import transpile, QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = qasm2.load(fname, include_path=(".",))
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    qc.save_statevector()
    s = time.time()
    job = simulator.run(qc, device="CPU")
    result = job.result()
    print("[Aer] Execution time: ", time.time() - s)
    return result.get_statevector().data

if __name__ == '__main__':
    parser = OptionParser()

    try:
        fname = "test.qasm"
        print("Running SnuQS...")
        result0 = run_snuqs(fname)
        print("Running Aer...")
        result1 = run_aer(fname)
    except Exception as e:
        print("HERE")
        print(e)
        sys.exit(1)

    print(result0)
    print(result1)
    i0 = np.argmax(np.abs(result0) * np.abs(result1))
    s0 = result0[i0]
    s1 = result1[i0]
    gp = (s1 / s0)
    gp = gp / np.abs(gp)
    result0 = result0 * gp
    error = sum(np.square(np.abs(result0 - result1)))

    if not np.isclose(error, 0):
        print(": Error: ", error)
    else:
        print(": Success")
