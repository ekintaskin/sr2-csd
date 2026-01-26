import numpy as np
from qpsolvers import solve_qp

def csdeconv_qp(dwsignal, voxel_sh, rho, X, B_reg):
    """ Constrained Spherical Deconvolution with quadprog solver from qpsolvers including modified version"""

    B_reg = np.array(-B_reg)
    h_mat = np.zeros(B_reg.shape[0])
    max_X = np.amax(X)
    new_rho = rho*max_X 

    Q = np.dot(X.T, dwsignal) + np.dot((new_rho **2), np.transpose(voxel_sh))
    Q = np.array(-Q)  
    P = np.dot(X.T, X) + np.dot((new_rho**2), np.identity(X.shape[1])) 
        
    x = solve_qp(P=P, q=Q, G=B_reg, h=h_mat, solver='quadprog')
    return x