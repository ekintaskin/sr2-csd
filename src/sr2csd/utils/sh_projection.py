import numpy as np


def project_sh_coeffs_nonneg(sh_coeff, B_reg, *, threshold=0.0, pinv=None):
    """
    Project SH coefficients to non-negative FOD amplitudes and back to SH.
    """
    sh_coeff = np.asarray(sh_coeff)
    B_reg = np.asarray(B_reg)

    if pinv is None:
        BtB = B_reg.T @ B_reg
        pinv = np.linalg.solve(BtB, B_reg.T)

    fodf_amp = np.tensordot(sh_coeff, B_reg.T, axes=([sh_coeff.ndim - 1], [0]))
    fodf_amp = np.maximum(fodf_amp, threshold)
    thr_sh = np.tensordot(fodf_amp, pinv.T, axes=([fodf_amp.ndim - 1], [0]))

    return thr_sh
