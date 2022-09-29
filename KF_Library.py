import numpy as np


######################################### Filtering Init #########################################
def filteringInitializer(F, Q, H, R, y, m0, P0, RunTime,l):
    n = int(2**np.ceil(np.log2(RunTime)))
    a = [[]] * n

    for k in range(l,RunTime,l):
        if k == l:
            m1 = F @ m0
            P1 = F @ P0 @ F.T + Q
            S = H @ P1 @ H.T + R
            K = P1 @ H.T @ np.linalg.inv(S)
            A = np.zeros(F.shape)
            b = m1 + K @ (y[0] - (H @ m1))
            b = b[:, None]
            C = P1 - (K @ S @ K.T)

            eta = np.zeros((F.shape[0],1))
            J = np.zeros(F.shape)

        else:
            S = H @ Q @ H.T + R
            K = Q @ H.T @ np.linalg.inv(S)
            A = F - K @ H @ F
            b = K @ y[k//l,None].T
            C = Q - K @ H @ Q

            eta = F.T @ H.T @ np.linalg.inv(S) @ y[k//l,None].T
            J = F.T @ H.T @ np.linalg.inv(S) @ H @ F 

        a[k] = {'A': A, 'b': b, 'C': C, 'eta': eta, 'J': J}
    return a

######################################### Filtering PerSum #########################################

def filtering(a,b):
    c = {}
    
    c['A'] = b['A'] @ np.linalg.inv(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ a['A']
    c['b'] = b['A'] @ np.linalg.inv(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ (a['b'] + a['C']@b['eta']) + b['b']
    c['C'] = b['A'] @ np.linalg.inv(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ a['C']@b['A'].T + b['C']
    c['eta'] = a['A'].T @ np.linalg.inv(np.eye(a['C'].shape[0]) + b['J']@a['C']) @ (b['eta'] - b['J']@a['b']) + a['eta']
    c['J'] =   a['A'].T @ np.linalg.inv(np.eye(a['C'].shape[0]) + b['J']@a['C']) @ b['J']@a['A'] + a['J']
           
    return c

#################################### parallel Scan Algorithm  #####################################

#@jit
def parallelScanAlgorithm(a,RunTime, op):
    
    n = int(2**np.ceil(np.log2(RunTime)))
    a = a.copy()
    a0 = a.copy()
    
    ## Up pass    
    for d in range(0, int(np.log2(n)), 1):
        for k in range(0, n, 2**(d+1)):
            i = k + 2**d - 1 
            j = k + 2**(d+1) - 1
            
            if len(a[j]) == 0:
                a[j] = a[i]
            elif len(a[i]) == 0:
                pass
            else:
                a[j] = op(a[i],a[j])
    a[-1] = []
    
    ## Down pass 
    
    for d in range(int(np.log2(n)-1), -1, -1):
        for k in range(0, n, 2**(d+1)): 
            i = k + 2**d - 1 
            j = k + 2**(d+1) - 1
            
            temp = a[i]
            a[i] =  a[j]
            
            if len(a[j]) == 0:
                a[j] = temp
            elif len(temp) == 0:
                pass
            else:
                a[j] = op(a[j],temp)
                
    ### Extra pass

    for k in range(1, n+1): 
        i = k-1
        
        if len(a[i]) == 0:
            a[i] = a0[i]
        elif len(a0[i]) == 0:
            pass
        else:
            a[i] = op(a[i],a0[i])
            
    a = a[:RunTime]
    
    return a



       