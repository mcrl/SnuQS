import numpy as np

def make_Hamiltonian(qubit_n, d_sparse, dt, k,m,c):
    
    a0 = m/dt**2 - c/2/dt
    b0 = k - 2 * m / dt / dt
    c0 = m/dt**2 + c/2/dt
    norm_abc = abs(a0) + abs(b0) + abs(c0)
    
    '''
    a=0.25
    b=-0.5
    c=0.75
    '''
    a= 0.125
    b= -0.25 
    c= 0.125
    
    
    H1 = np.zeros((2**qubit_n,2**qubit_n))
    for i in range(2**(qubit_n-1)):
        H1[i, i+2**(qubit_n-1)] = c
    for i in range(2**(qubit_n-1)):
        H1[i+2**(qubit_n-1), i] = c
    
        
    
    H2 = np.zeros((2**qubit_n,2**qubit_n))
    for i in range(1, 2**(qubit_n-1)):
        H2[i, i+2**(qubit_n-1)-1] = b
    for i in range(2**(qubit_n-1) -1):
        H2[i+2**(qubit_n-1), i+1] = b
    H2[1, 2**(qubit_n-1)] = -c
    H2[2**(qubit_n-1), 1] = -c
    
    
    
    H3 = np.zeros((2**qubit_n,2**qubit_n))
    for i in range(2, 2**(qubit_n-1)):
        H3[i, i+2**(qubit_n-1)-2] = a
    for i in range(2**(qubit_n-1) -2):
        H3[i+2**(qubit_n-1), i+2] = a
        
        
    
    
    H = np.zeros((2**qubit_n,2**qubit_n,d_sparse))
    H[:,:,0] = H1
    H[:,:,1] = H2
    H[:,:,2] = H3
    H_ana = H1 + H2 + H3
    
    
    
    total_step = 2**(qubit_n)
    bm = np.zeros(total_step)

    bm[2]=c/dt+k
    bm[3]=-c/dt+k

    b = bm / np.linalg.norm(bm)
    
    
    return H, H_ana, b