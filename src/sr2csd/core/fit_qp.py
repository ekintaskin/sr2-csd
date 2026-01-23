import numpy as np
from .qp_solver import csdeconv_qp

def fit_qp(dwi_data, voxel_sh, rho, X, B_reg, tau):
    """ Alternative fit function for QP solver """    

    if np.sum(dwi_data) != 0:
        shm_coeff = csdeconv_qp(dwi_data, voxel_sh, rho, X, B_reg, tau)
    
    else:
        shm_coeff = np.zeros_like(voxel_sh)

    return shm_coeff