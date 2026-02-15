from pathlib import Path

import numpy as np
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.shm import SphHarmFit

from .core.fit_qp import fit_qp
from .core.csdeconv_mrtrix import csdeconv_mrtrix
from .utils.denoise import calibrate_and_denoise_tv_chambolle_4d
from .utils.multi_voxel import multi_voxel_fit_qp
from .utils.sh_projection import project_sh_coeffs_nonneg


class MRtrixCSDModel(ConstrainedSphericalDeconvModel):
    """
    CSD model that uses the MRtrix-hyperparameters CSD fitter (csdeconv_mrtrix).
    """

    def __init__(self, gtab, response, sh_order, *, tau=0.1, convergence=50, eta=1.0, **kwargs):
        super().__init__(
            gtab,
            response,
            sh_order=sh_order,
            tau=tau,
            convergence=convergence,
            **kwargs,
        )
        self.tau = float(tau)
        self.convergence = int(convergence)
        self.eta = float(eta)

    @multi_voxel_fit_qp
    def fit_csd(self, data, idx=None, verbose=False):
        """
        MRtrix-hyperparameters CSD fit for a single voxel, wrapped for volumes.
        """
        dwi_data = data[self._where_dwi]

        if np.sum(dwi_data) == 0:
            ncoef = self._X.shape[1]
            shm_coeff = np.zeros((ncoef,), dtype=float)
            return SphHarmFit(self, shm_coeff, None)

        voxel_sh, _ = csdeconv_mrtrix(
            dwi_data,
            self._X,
            self.B_reg,
            self.tau,
            convergence=self.convergence,
            eta=self.eta,
            P=self._P,
        )

        return SphHarmFit(self, voxel_sh, None)

    def fit(self, data, mask=None, verbose=False):
        """
        Public fit API. Uses the MRtrix-style CSD implementation.
        """
        return self.fit_csd(data, mask=mask, verbose=verbose)


class SR2CSDModel(ConstrainedSphericalDeconvModel):
    """
    SR²-CSD model implemented on top of DIPY's CSD model.

      1) compute an initial SH estimate (mrtrix-like / projected CSD) per voxel
      2) run the QP-based SR²-CSD step using that initial estimate as prior
      3) return a DIPY SphHarmFit
    """

    def __init__(
        self,
        gtab,
        response,
        sh_order,
        *,
        rho,
        tau=0.1,
        convergence=50,
        eta=1.0,
        use_prior_pipeline=True,
        denoise=True,
        denoise_k_range=None,
        denoise_eps=2e-4,
        denoise_max_num_iter=200,
        denoise_verbose=True,
        denoise_mask=None,
        csd_use_mask=True,
        projection_threshold=0.0,
        **kwargs,
    ):
        super().__init__(
            gtab,
            response,
            sh_order=sh_order,
            tau=tau,
            convergence=convergence,
            **kwargs,
        )
        self.gtab = gtab
        self.response = response
        self.sh_order = sh_order
        self.rho = float(rho)
        self.tau = float(tau)
        self.convergence = int(convergence)
        self.eta = float(eta)
        self.use_prior_pipeline = bool(use_prior_pipeline)
        self.denoise = bool(denoise)
        if denoise_k_range is None:
            denoise_k_range = np.arange(0.25, 5, 0.1)
        self.denoise_k_range = np.asarray(denoise_k_range)
        self.denoise_eps = float(denoise_eps)
        self.denoise_max_num_iter = int(denoise_max_num_iter)
        self.denoise_verbose = bool(denoise_verbose)
        self.denoise_mask = denoise_mask
        self.csd_use_mask = bool(csd_use_mask)
        self.projection_threshold = float(projection_threshold)


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
        # If a prior was provided, use it; otherwise run MRtrix-style CSD.
        prior_sh = None
        if getattr(self, "_prior_sh", None) is not None:
            prior_sh = self._prior_sh if idx is None else self._prior_sh[idx]

        if prior_sh is None:
            # Expecting csdeconv_mrtrix returns (voxel_sh, n_iter).
            voxel_sh, _ = csdeconv_mrtrix(
                dwi_data,
                self._X,
                self.B_reg,
                self.tau,
                convergence=self.convergence,
                eta=self.eta,
                P=self._P,
            )
        else:
            voxel_sh = prior_sh

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

    def fit_with_prior(self, data, prior_sh, mask=None, verbose=False):
        """
        Fit SR²-CSD using a precomputed SH prior (e.g., denoised/projected CSD).
        """
        prior_sh = np.asarray(prior_sh)
        ncoef = self._X.shape[1]
        if data.ndim == 1:
            if prior_sh.shape != (ncoef,):
                raise ValueError(
                    f"prior_sh shape must be ({ncoef},) for 1D data, got {prior_sh.shape}"
                )
        else:
            expected = data.shape[:-1] + (ncoef,)
            if prior_sh.shape != expected:
                raise ValueError(
                    f"prior_sh shape must be {expected} for 4D data, got {prior_sh.shape}"
                )

        self._prior_sh = prior_sh
        try:
            return self.fit_qp(data, mask=mask, verbose=verbose)
        finally:
            self._prior_sh = None

    def fit(self, data, mask=None, verbose=False):
        """
        Public fit API. Uses the QP-based SR²-CSD implementation.
        If use_prior_pipeline=True, runs CSD -> denoise -> project -> QP.
        """
        if not self.use_prior_pipeline:
            return self.fit_qp(data, mask=mask, verbose=verbose)

        # (1) CSD prior
        csd_model = MRtrixCSDModel(
            self.gtab,
            self.response,
            sh_order=self.sh_order,
            tau=self.tau,
            convergence=self.convergence,
            eta=self.eta,
        )
        csd_mask = mask if self.csd_use_mask else None
        csd_fit = csd_model.fit(data, mask=csd_mask, verbose=verbose)
        sh_coeff = csd_fit.shm_coeff

        # (2) Denoiser calibration + denoise (optional)
        if self.denoise:
            denoise_mask = self.denoise_mask if self.denoise_mask is not None else mask
            if isinstance(denoise_mask, (str, Path)):
                from dipy.io.image import load_nifti
                denoise_mask, _ = load_nifti(str(denoise_mask))
            if denoise_mask is not None and denoise_mask.ndim != 3:
                raise ValueError("denoise_mask must be a 3D array or a path to a 3D NIfTI mask.")
            sh_coeff, best_params, _, _ = calibrate_and_denoise_tv_chambolle_4d(
                sh_coeff,
                K_range=self.denoise_k_range,
                mask=denoise_mask,
                eps=self.denoise_eps,
                max_num_iter=self.denoise_max_num_iter,
                verbose=self.denoise_verbose,
            )
            if self.denoise_verbose:
                try:
                    print(f"Optimal K: {best_params['K']}")
                except Exception:
                    print(f"Optimal K: {best_params}")

        # (3) Project + threshold
        thr_sh_coeff = project_sh_coeffs_nonneg(
            sh_coeff,
            csd_model.B_reg,
            threshold=self.projection_threshold,
        )

        # (4) SR²-CSD QP using prior
        return self.fit_with_prior(data, thr_sh_coeff, mask=None, verbose=verbose)
