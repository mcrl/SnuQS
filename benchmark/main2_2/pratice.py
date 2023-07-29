from qiskit import QuantumCircuit, assemble, Aer, transpile

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)


qc2 = QuantumCircuit(2)
qc2.cx(0,1)

qc.append(qc2, [0,1])


#qc.measure(1,0)
####################################################
from snuqs.simulators import StatevectorSimulator
backend = StatevectorSimulator()
print(backend)
qc = transpile(qc)
#qc.save_statevector()
job = backend.run(qc)
result = job.result()
print(result)
sv = result.get_statevector()
print(sv)

###################################################

#from qiskit.providers.aer import *
#backend = AerSimulator(method='statevector')
#print(backend)
#
#qc.save_statevector() 
#
#qc = transpile(qc, backend)
#result = backend.run(qc).result()
#print(result)
#state = result.get_statevector()
#print(state)
