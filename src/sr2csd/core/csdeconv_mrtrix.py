import numpy as np
import scipy.linalg.lapack as ll
import scipy.linalg as la
import warnings

potrf, potrs = ll.get_lapack_funcs(('potrf', 'potrs'))

def _solve_cholesky(Q, z):
    L, info = potrf(Q, lower=False, overwrite_a=False, clean=False)
    if info > 0:
        msg = "%d-th leading minor not positive definite" % info
        raise la.LinAlgError(msg)
    if info < 0:
        msg = 'illegal value in %d-th argument of internal potrf' % -info
        raise ValueError(msg)
    f, info = potrs(L, z, lower=False, overwrite_b=False)
    if info != 0:
        msg = 'illegal value in %d-th argument of internal potrs' % -info
        raise ValueError(msg)
    return f

def csdeconv_mrtrix(dwsignal, X, B_reg, tau=0.1, convergence=50, eta=1, P=None):
    r""" 
    Modified from dipy.reconst.csdeconv.py. 
    The updated version employs the default parameters in MRtrix3.

    Constrained-regularized spherical deconvolution (CSD) [1]_

    Deconvolves the axially symmetric single fiber response function `r_rh` in
    rotational harmonics coefficients from the diffusion weighted signal in
    `dwsignal`.

    Parameters
    ----------
    dwsignal : array
        Diffusion weighted signals to be deconvolved.
    X : array
        Prediction matrix which estimates diffusion weighted signals from FOD
        coefficients.
    B_reg : array (N, B)
        SH basis matrix which maps FOD coefficients to FOD values on the
        surface of the sphere. B_reg should be scaled to account for lambda.
    tau : float
        Threshold controlling the amplitude below which the corresponding fODF
        is assumed to be zero.  
    convergence : int
        Maximum number of iterations to allow the deconvolution to converge.
    P : ndarray
        This is an optimization to avoid computing ``dot(X.T, X)`` many times.
        If the same ``X`` is used many times, ``P`` can be precomputed and
        passed to this function.

    Returns
    -------
    fodf_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
         Spherical harmonics coefficients of the constrained-regularized fiber
         ODF.
    num_it : int
         Number of iterations in the constrained-regularization used for
         convergence.


    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the
           fibre orientation distribution in diffusion MRI: Non-negativity
           constrained super-resolved spherical deconvolution.
    
    """

    if P is None:
        P = np.dot(X.T, X)
    z = np.dot(X.T, dwsignal)


    Pt = P + 0.0002 * P[0,0] * np.eye(P.shape[0])
    fodf_sh = _solve_cholesky(Pt, z)
    
    # For the first iteration we use a smooth FOD that only uses SH orders up
    # to 4 (the first 15 coefficients).
    fodf = np.dot(B_reg[:, :15], fodf_sh[:15])
    # The mean of an fodf can be computed by taking $Y_{0,0} * coeff_{0,0}$
    
    threshold = B_reg[0, 0] * fodf_sh[0] * 0
    where_fodf_small = (fodf < threshold).nonzero()[0]

    # If the low-order fodf does not have any values less than threshold, the
    # full-order fodf is used.
    if len(where_fodf_small) == 0:
        fodf = np.dot(B_reg, fodf_sh)
        where_fodf_small = (fodf < threshold).nonzero()[0]
        # If the fodf still has no values less than threshold, return the fodf.
        if len(where_fodf_small) == 0:
            return fodf_sh, 0

    for num_it in range(1, convergence + 1):
        # This is the super-resolved trick.  Wherever there is a negative
        # amplitude value on the fODF, it concatenates a value to the S vector
        # so that the estimation can focus on trying to eliminate it. In a
        # sense, this "adds" a measurement, which can help to better estimate
        # the fodf_sh, even if you have more SH coefficients to estimate than
        # actual S measurements.
        
        H = B_reg.take(where_fodf_small, axis=0)

        # We use the Cholesky decomposition to solve for the SH coefficients.
        # Modified employing MRtrix values 
        Q = P + eta * np.dot(H.T, H)
        Q = Q + np.eye(Q.shape[0]) * 0.0002 * Q[0,0]
        fodf_sh = _solve_cholesky(Q, z)

        # Sample the FOD using the regularization sphere and compute k.
        fodf = np.dot(B_reg, fodf_sh)
        where_fodf_small_last = where_fodf_small
        where_fodf_small = (fodf < threshold).nonzero()[0]

        if (len(where_fodf_small) == len(where_fodf_small_last) and
                (where_fodf_small == where_fodf_small_last).all()):
            break
    else:
        msg = 'maximum number of iterations exceeded - failed to converge'
        warnings.warn(msg)

    return fodf_sh, num_it
    
