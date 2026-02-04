import numpy as np

def calculate_max_order(n_coeffs, full_basis=False):
    r"""Calculate the maximal harmonic order, given that you know the
    number of parameters that were estimated.

    Parameters
    ----------
    n_coeffs : int
        The number of SH coefficients
    full_basis: bool, optional
        True if the used SH basis contains even and odd order SH functions.
        False if the SH basis consists only of even order SH functions.

    Returns
    -------
    L : int
        The maximal SH order, given the number of coefficients

    Notes
    -----
    The calculation in this function for the symmetric SH basis
    proceeds according to the following logic:
    .. math::
        n = \frac{1}{2} (L+1) (L+2)
        \rarrow 2n = L^2 + 3L + 2
        \rarrow L^2 + 3L + 2 - 2n = 0
        \rarrow L^2 + 3L + 2(1-n) = 0
        \rarrow L_{1,2} = \frac{-3 \pm \sqrt{9 - 8 (1-n)}}{2}
        \rarrow L{1,2} = \frac{-3 \pm \sqrt{1 + 8n}}{2}

    Finally, the positive value is chosen between the two options.

    For a full SH basis, the calculation consists in solving the equation
    $n = (L + 1)^2$ for $L$, which gives $L = sqrt(n) - 1$.
    """

    # L2 is negative for all positive values of n_coeffs, so we don't
    # bother even computing it:
    # L2 = (-3 - np.sqrt(1 + 8 * n_coeffs)) / 2
    # L1 is always the larger value, so we go with that:
    if full_basis:
        L1 = np.sqrt(n_coeffs) - 1
        if L1.is_integer():
            return int(L1)
    else:
        L1 = (-3 + np.sqrt(1 + 8 * n_coeffs)) / 2.0
        # Check that it is a whole even number:
        if L1.is_integer() and not np.mod(L1, 2):
            return int(L1)

    # Otherwise, the input didn't make sense:
    raise ValueError("The input to ``calculate_max_order`` was ",
                     "%s, but that is not a valid number" % n_coeffs,
                     "of coefficients for a spherical harmonics ",
                     "basis set.")


def sph_harm_ind_list(sh_order, full_basis=False):
    """
    Returns the degree (``m``) and order (``n``) of all the symmetric spherical
    harmonics of degree less then or equal to ``sh_order``. The results,
    ``m_list`` and ``n_list`` are kx1 arrays, where k depends on ``sh_order``.
    They can be passed to :func:`real_sh_descoteaux_from_index` and
    :func:``real_sh_tournier_from_index``.

    Parameters
    ----------
    sh_order : int
        even int > 0, max order to return
    full_basis: bool, optional
        True for SH basis with even and odd order terms

    Returns
    -------
    m_list : array
        degrees of even spherical harmonics
    n_list : array
        orders of even spherical harmonics

    See Also
    --------
    shm.real_sh_descoteaux_from_index, shm.real_sh_tournier_from_index

    """
    if full_basis:
        n_range = np.arange(0, sh_order + 1, dtype=int)
        ncoef = int((sh_order + 1) * (sh_order + 1))
    else:
        if sh_order % 2 != 0:
            raise ValueError('sh_order must be an even integer >= 0')
        n_range = np.arange(0, sh_order + 1, 2, dtype=int)
        ncoef = int((sh_order + 2) * (sh_order + 1) // 2)

    n_list = np.repeat(n_range, n_range * 2 + 1)
    offset = 0
    m_list = np.empty(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return m_list, n_list

def convert_sh_descoteaux_tournier(sh_coeffs):
    """Convert SH coefficients between legacy-descoteaux07 and tournier07.

    Convert SH coefficients between the legacy ``descoteaux07`` SH basis and
    the non-legacy ``tournier07`` SH basis. Because this conversion is equal to
    its own inverse, it can be used to convert in either direction:
    legacy-descoteaux to non-legacy-tournier or non-legacy-tournier to
    legacy-descoteaux.

    This can be used to convert SH representations between DIPY and MRtrix3.

    See [descoteaux07]_ and [tournier19]_ for the origin of these SH bases.
    See [mrtrixbasis]_ for a description of the basis used in MRtrix3.
    See [mrtrixdipybases]_ for more details on the conversion.

    Parameters
    ----------
    sh_coeffs: ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel.

    Returns
    -------
    out_sh_coeffs: ndarray
        The array of coefficients expressed in the "other" SH basis. If the
        input was in the legacy-descoteaux basis then the output will be in the
        non-legacy-tournier basis, and vice versa.

    References
    ----------
    .. [descoteaux07] Descoteaux, M., Angelino, E., Fitzgibbons, S. and
           Deriche, R. Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [tournier19] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    .. [mrtrixbasis] https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html
    .. [mrtrixdipybases] https://github.com/dipy/dipy/discussions/2959#discussioncomment-7481675
    """  # noqa: E501

    sh_order = calculate_max_order(sh_coeffs.shape[-1])
    m, n = sph_harm_ind_list(sh_order)
    basis_indices = list(zip(n, m))  # dipy basis ordering
    basis_indices_permuted = list(zip(n, -m))  # mrtrix basis ordering
    permutation = [
        basis_indices.index(basis_indices_permuted[i])
        for i in range(len(basis_indices))
    ]
    return sh_coeffs[..., permutation]