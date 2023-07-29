from scipy.linalg import expm
from qiskit import QuantumCircuit, assemble, Aer, transpile
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.quantum_info import Statevector
import snuqs


from problemQ import *
from operatorQ import *
from circuitQ import *

#################################################################
##################################################################
# Setting
qubit_n = 3
qubit_c = 1
qubit_h = 4
total_qubit_sim = qubit_n + qubit_n + qubit_c + qubit_h + 1

qubit_m = 4
#total_qubit_hhl = qubit_n + qubit_m + 1


total_qubit = total_qubit_sim + qubit_m + 1

time = 2*np.pi
seperate = 10
d_sparse = 3
C_costant = 0.01
##################################################################
# Classical
k = 20000 # N/m
m = 400 # kg
c = 2000 # Ns/m
dt = 0.01

H, H_ana, init = make_Hamiltonian(qubit_n, d_sparse, dt, k,m,c)
Of_matrix_d, Oh_matrix_d, Oc_matrix, Os_matrix = make_Operator(qubit_n, qubit_h, d_sparse, H)
ham_sim = one_sparse(qubit_n, qubit_c, qubit_h, time, seperate, d_sparse,Of_matrix_d, Oh_matrix_d, Oc_matrix, Os_matrix)


iham_sim = ham_sim.inverse()


#init = np.array([np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8),np.sqrt(1/8),np.sqrt(0.05),np.sqrt(0.1),np.sqrt(0.15),np.sqrt(0.2)])




ana_result = np.matmul( np.linalg.inv(H_ana), init)
print("Python result: ",ana_result / np.linalg.norm(ana_result))

##################################################################
##################################################################

qc = QuantumCircuit(total_qubit)

# b
qc.initialize(init, range(qubit_n))


# H
for i in range(qubit_m):
    qc.h(total_qubit_sim+i)

     

# e^iAt
for i in range(qubit_m):
    ham_qubit=[]
    for j in range(total_qubit_sim):
        ham_qubit.append(j)
    ham_qubit.append(total_qubit_sim+i)
    '''              
    for j in range(2**i):
        qc.append(ham_sim, ham_qubit)
    '''

# IFT
for i in range(qubit_m):
    targetq = qubit_m + total_qubit_sim-i-1
    
    
    for j in range(i):
        theta = -2*np.pi/2**(i+1-j)
        qc.cp(theta, qubit_m+total_qubit_sim-j-1, targetq)
    qc.h(targetq)

##################################################################
qc.barrier()

#R
for i in range(1,2**qubit_m):
    qubit_list = np.zeros(qubit_m,dtype=int)
    binary_val = [int(xbin) for xbin in bin(i)[2:]]
    qubit_list[len(qubit_list)-len(binary_val):]=binary_val
    
    
    qubit_list = np.flip(qubit_list)

    
    if i <2**(qubit_m-1):
        theta = np.arcsin(C_costant/i)
    else:
        theta = -np.arcsin(C_costant/(2**qubit_m-i))

    
    list_string = map(str, qubit_list)
    ctrlL = ''.join(list_string)
    CCRY = RYGate(2*theta).control(qubit_m, ctrl_state=ctrlL)
    


    unarr=np.append(total_qubit_sim + np.arange(qubit_m), qubit_m + total_qubit_sim)
    #qc.append(CCRY, list(unarr))

qc.barrier()

##################################################################
# FT
for i in range(qubit_m):
    targetq = total_qubit_sim + i
    
    qc.h(targetq)
    for j in range(qubit_m-i-1):
        theta = 2*np.pi/2**(2+j)
        qc.cp(theta, total_qubit_sim+i+j+1, targetq)
    

# U
for i in range(qubit_m-1, -1, -1):
    ham_qubit=[]
    for j in range(total_qubit_sim):
        ham_qubit.append(j)
    ham_qubit.append(total_qubit_sim+i)
    '''
    for j in range(2**i):
        qc.append(iham_sim, ham_qubit)
    '''


# H
for i in range(qubit_m):
    qc.h(total_qubit_sim+i)
    




# Reuslt ############################
print("quantum calculation start")
#qc.draw(output='mpl')
#ham_sim.draw(output='mpl')


job = snuqs.run(qc)
result = job.result()
sv = result.get_statevector()
print(sv[0])
'''

from qiskit.providers.aer import *
backend = AerSimulator(method='statevector')

qc.save_statevector() 

qc = transpile(qc, backend)
result = backend.run(qc).result()
#state = result.get_statevector(decimals=4)
state = result.get_statevector()
print("circuit result: ",state[2^(total_qubit-1)+4],state[2^(total_qubit-1)+5],state[2^(total_qubit-1)+6],state[2^(total_qubit-1)+7])
#print("circuit result: ",state[260100],state[260101],state[260102],state[260103])
print(np.nonzero(state))
print(np.count_nonzero(init))
'''