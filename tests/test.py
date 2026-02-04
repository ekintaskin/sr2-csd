"""Run standard SSST-CSD and SR2-CSD on HARDI dataset."""

import sys
from pathlib import Path

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.segment.mask import median_otsu
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sr2csd.utils.sh_basis import convert_sh_descoteaux_tournier
from sr2csd import SR2CSDModel

def main() -> None:
    data, data_affine = load_nifti(r"/home/ekin/Desktop/paper_analysis3/hardi/snr15/set10/denoised_hardi_snr15_set10.nii.gz")
    bvals = np.loadtxt(r"/home/ekin/Documents/data/hardi/hardi-scheme.bval", dtype=float)
    bvecs = np.loadtxt(r"/home/ekin/Documents/data/hardi/hardi-scheme.bvec", dtype=float)
    gtab = gradient_table(bvals, bvecs)

    mask, _ = load_nifti(r"/home/ekin/Documents/data/hardi/mask_compute_local_metrics.nii.gz")

    b0_idx = np.flatnonzero(bvals < 50)

    response, ratio = auto_response_ssst(
        gtab,
        data,
        roi_radii=4,
        fa_thr=0.3,
    )

    csd_model = ConstrainedSphericalDeconvModel(
        gtab,
        response,
        sh_order=8,
    )
    csd_fit = csd_model.fit(data, mask=mask)

    sr2csd = SR2CSDModel(
        gtab,
        response,
        sh_order=12,
        rho=0.1,
        tau=0.1,
        convergence=30,
        eta=1.0,
    )
    sr2csd_fit = sr2csd.fit(data, mask=mask)

    csd_coeff = csd_fit.shm_coeff
    sr2csd_coeff = sr2csd_fit.shm_coeff

    tournier_base_csd = convert_sh_descoteaux_tournier(csd_coeff)
    fods_img_csd = nib.Nifti1Image(tournier_base_csd, data_affine)
    nib.save(fods_img_csd,  f"/home/ekin/Desktop/sr2-csd/tests/csd8_test1.nii.gz")

    tournier_base_sr2csd = convert_sh_descoteaux_tournier(sr2csd_coeff)
    fods_img_sr2csd = nib.Nifti1Image(tournier_base_sr2csd, data_affine)
    nib.save(fods_img_sr2csd,  f"/home/ekin/Desktop/sr2-csd/tests/sr2csd_test1.nii.gz")

    print("CSD and SR2-CSD ran successfully.")
    print(f"Response ratio (lambda2/lambda1): {ratio:.6f}")
    print(f"CSD coeff shape: {csd_fit.shm_coeff.shape}")
    print(f"SR2 coeff shape: {sr2csd_fit.shm_coeff.shape}")


if __name__ == "__main__":
    main()
