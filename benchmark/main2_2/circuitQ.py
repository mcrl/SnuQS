from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import MCMT, SwapGate, RXGate, HGate


def one_sparse(qubit_n, qubit_c, qubit_h, time, seperate,d_sparse, Of_matrix_d, Oh_matrix_d, Oc_matrix, Os_matrix):
    qubit_imp =qubit_n + qubit_n + qubit_c + qubit_h
    total_qubit = qubit_n + qubit_n + qubit_c + qubit_h + 1 +1
    control_qubit = qubit_n + qubit_n + qubit_c + qubit_h + 1
    
    #Of
    
    Of_qubit = []
    
    for i in range(2*qubit_n):
        Of_qubit.append(i)
    Of_qubit.append(control_qubit)    
    # OC
    Oc = UnitaryGate(Oc_matrix)
    OcT = UnitaryGate(Oc_matrix.transpose())
    Oc_qubit = []
    
    for i in range(2*qubit_n+1):
        Oc_qubit.append(i)
    Oc_qubit.append(control_qubit)
    
    
    
    # Oh
    
    Oh_qubit_list = []
    
    for i in range (qubit_n*2):
        Oh_qubit_list.append(i)
    for i in range( qubit_h):
        Oh_qubit_list.append(qubit_n + qubit_n + qubit_c + i)        
        
    Oh_qubit_list.append(control_qubit)    
    
    # Os
    Os = UnitaryGate(Os_matrix)
    OsT = UnitaryGate(Os_matrix.transpose())
    Os_qubit=[]
               
    for i in range(2*qubit_n+1):
        Os_qubit.append(i)
    
    Os_qubit.append(control_qubit) 
    
    
    qc = QuantumCircuit(total_qubit)
    for time_repeat in range(seperate):
        for d_repeat in range(d_sparse):
            Of = UnitaryGate(Of_matrix_d[:,:,d_repeat])
            Oh = UnitaryGate(Oh_matrix_d[:,:,d_repeat])
            OhT = UnitaryGate(Oh_matrix_d[:,:,d_repeat].transpose())
            OfT = UnitaryGate(Of_matrix_d[:,:,d_repeat].transpose())
            
            
            
            # Of 
            qc.append(Of, Of_qubit)
            
            # Comp
            qc.append(Oc, Oc_qubit)        
            qc.ccx(control_qubit, qubit_n*2, qubit_imp )
            qc.append(OcT, Oc_qubit)     


            # CSWAP
            for i in range(qubit_n):
                qc.append(SwapGate().control(2), [qubit_imp, control_qubit, i,qubit_n+i])
        
            
            # OH 
            qc.append(Oh, Oh_qubit_list)

        
        
            # H
            qc.append(Os, Os_qubit)
            qc.append(HGate().control(2), [control_qubit, qubit_n*2,  qubit_imp])
            qc.append(OsT, Os_qubit)

            
            
            # CRX
            qc.ccz(qubit_n + qubit_n + qubit_c + qubit_h-1, control_qubit, qubit_imp)
            
            for i in range(qubit_h-1):
                qc.append(RXGate(-2 * time/seperate / 2**(qubit_h-i-1)).control(2), [qubit_n + qubit_n + qubit_c + i, control_qubit,  qubit_imp])
            
            qc.ccz(qubit_n + qubit_n + qubit_c + qubit_h-1, control_qubit, qubit_imp)
            
            
            # uncomputation
            # Un H
            qc.append(OsT, Os_qubit)
            qc.append(HGate().control(2), [control_qubit, qubit_n*2,  qubit_imp])
            qc.append(Os, Os_qubit)
            
            
            
            # Un Oh
            qc.append(OhT, Oh_qubit_list)
            
            for i in range(qubit_n):
                qc.append(SwapGate().control(2), [qubit_imp, control_qubit, i,qubit_n+i])
                
                
                
            # Un Comp
            qc.append(OcT, Oc_qubit)  
            qc.ccx(qubit_n*2, control_qubit, qubit_imp )
            qc.append(Oc, Oc_qubit)   
  
            
            
            # Un Of
            qc.append(OfT, Of_qubit)
        
    
    return qc