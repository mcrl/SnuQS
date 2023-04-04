import sys
import qiskit
import logging

from qiskit import Aer

logger = logging.getLogger('snume')
sh = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(filename)s:%(lineno)d:%(funcName)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)


def run(qc, **kwargs):
    if type(qc) is qiskit.circuit.quantumcircuit.QuantumCircuit:
        logger.info("Running qiskit quantum circuit")
        #print(qc.qasm())
        return Aer.get_backend("qasm_simulator").run(qc, **kwargs)
    else:
        logger.error(f"Not supported quantum circuit type {type(qc)}")
        sys.exit(1)
