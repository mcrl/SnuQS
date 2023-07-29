import numpy as np
import math

def make_Operator(qubit_n, qubit_h, d_sparse, H):
    
    Of_matrix_d = np.zeros(( 2**(2*qubit_n+1),2**(2*qubit_n+1),d_sparse))
    Oh_matrix_d = np.zeros(( 2**(2*qubit_n+qubit_h+1),2**(2*qubit_n+qubit_h+1),d_sparse))
    
    
    for d in range(d_sparse):
        Of_matrix = np.eye(2**(2*qubit_n))
        Oh_matrix = np.eye(2**(2*qubit_n+qubit_h))
        nonzero_loc = []
        nonzero_list = []
    
        #Of
        for i in range(2**qubit_n):
            bef = i
            aft = i * 2**qubit_n + i
            only_zero = True
            for j in range(2**qubit_n):
                if H[i,j,d]!=0:
                    aft = 2**qubit_n * j + i
                    only_zero = False
            
            Of_matrix[bef,bef] = 0
            Of_matrix[aft,aft] = 0
            Of_matrix[bef,aft] = 1
            Of_matrix[aft,bef] = 1
                
            nonzero_loc.append(aft)
            nonzero_list.append(only_zero)
            
        #Oh
        
        for i in range(2**qubit_n):
            if (nonzero_list[i] == False):
                bef = nonzero_loc[i]
                
        
                for j in range(2**qubit_n):
                    if H[i,j,d]!=0:
                        #aft = 2**(2*qubit_n) *  H[i,j,d] + bef
                        orgH_value = H[i,j,d]
                        newH_value = np.zeros(qubit_h)
                        
                        if orgH_value>0:
                            newH_value[0] = 0
                        else:
                            newH_value[0] = 1
                        
                        orgH_value = abs(orgH_value)
                        for k in range(1,qubit_h):
                            orgH_value = orgH_value * 2
                            newH_value[k] = math.floor(orgH_value)
                            orgH_value = orgH_value - math.floor(orgH_value)
                        
                        newH_value_merge = 0
                        for k in range(qubit_h):
                            newH_value_merge = newH_value[k] * 2**(qubit_h-k-1) + newH_value_merge
                            
                        aft = 2**(2*qubit_n) *  int(newH_value_merge) + bef
    
                        Oh_matrix[bef,bef] = 0
                        Oh_matrix[aft,aft] = 0
                        Oh_matrix[bef,aft] = 1
                        Oh_matrix[aft,bef] = 1   
    
        
    
    
        Of_matrix_d[:,:,d] =np.eye(2**(2*qubit_n+1))
        Oh_matrix_d[:,:,d] =np.eye(2**(2*qubit_n+qubit_h+1))
        Of_matrix_d[2**(2*qubit_n):,2**(2*qubit_n):,d] = Of_matrix
        Oh_matrix_d[2**(2*qubit_n+qubit_h):,2**(2*qubit_n+qubit_h):,d] = Oh_matrix
    
    
    # Oc
    Oc_matrix = np.eye(2**(2*qubit_n+1 +1))
    for i in range(2**qubit_n):
        for j in range(2**qubit_n):
            if i>j:
                old_loc = 2**(2*qubit_n+1) + i + 2**qubit_n * j
                new_loc = 2**(2*qubit_n+1) + 2**(2*qubit_n) + i + 2**qubit_n * j
                
                Oc_matrix[old_loc,old_loc] = 0
                Oc_matrix[new_loc,new_loc] = 0
                Oc_matrix[new_loc,old_loc] = 1  
                Oc_matrix[old_loc,new_loc] = 1  
    
    # Os
    Os_matrix = np.eye(2**(2*qubit_n+1 +1))
    for i in range(2**qubit_n):
        old_loc = 2**(2*qubit_n+1) + i + 2**qubit_n * i
        new_loc = 2**(2*qubit_n+1) + 2**(2*qubit_n) + i + 2**qubit_n * i
        
        Os_matrix[old_loc,old_loc] = 0
        Os_matrix[new_loc,new_loc] = 0
        Os_matrix[new_loc,old_loc] = 1  
        Os_matrix[old_loc,new_loc] = 1  
        
    return Of_matrix_d, Oh_matrix_d, Oc_matrix, Os_matrix
    