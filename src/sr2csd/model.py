import numpy as np
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.shm import SphHarmFit

from .core.fit_qp import fit_qp
from .core.csdeconv_mrtrix import csdeconv_mrtrix
from .utils.multi_voxel import multi_voxel_fit_qp


class SR2CSDModel(ConstrainedSphericalDeconvModel):
    """
    SR²-CSD model implemented on top of DIPY's CSD model.

      1) compute an initial SH estimate (mrtrix-like / projected CSD) per voxel
      2) run the QP-based SR²-CSD step using that initial estimate as prior
      3) return a DIPY SphHarmFit
    """

    def __init__(self, gtab, response, sh_order, *, rho, tau=0.1, convergence=50, eta=1.0, **kwargs):
        super().__init__(
            gtab,
            response,
            sh_order=sh_order,
            tau=tau,
            convergence=convergence,
            **kwargs,
        )
        self.rho = float(rho)
        self.tau = float(tau)
        self.convergence = int(convergence)
        self.eta = float(eta)


    @multi_voxel_fit_qp
    def fit_qp(self, data, idx=None, verbose=False):
        """
        QP-based fit (SR²-CSD) for a single voxel, wrapped by multi_voxel_fit_qp for volumes.
        """
        # keep exactly the same DWI selection behavior
        dwi_data = data[self._where_dwi]

        # match original behavior for empty voxels
        if np.sum(dwi_data) == 0:
            # coefficient length should match model basis
            ncoef = self._X.shape[1]
            shm_coeff = np.zeros((ncoef,), dtype=float)
            return SphHarmFit(self, shm_coeff, None)

        # (1) initial SH estimate (this replaces self.sh_coeff[idx])
        # Expecting csdeconv_mrtrix returns (voxel_sh, n_iter). If it returns only voxel_sh, adjust below.
        voxel_sh, _ = csdeconv_mrtrix(
            dwi_data,
            self._X,
            self.B_reg,
            self.tau,
            convergence=self.convergence,
            eta=self.eta,
            P=self._P,
        )

        # (2) SR²-CSD QP step
        shm_coeff = fit_qp(
            dwi_data,
            voxel_sh,
            rho=self.rho,
            X=self._X,
            B_reg=self.B_reg,
            tau=self.tau,
        )

        return SphHarmFit(self, shm_coeff, None)

    def fit(self, data, mask=None, verbose=False):
        """
        Public fit API. Uses the QP-based SR²-CSD implementation.
        """
        return self.fit_qp(data, mask=mask, verbose=verbose)
