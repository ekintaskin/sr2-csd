"""Run standard SSST-CSD and SR2-CSD on HARDI dataset."""

import sys
from pathlib import Path

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.csdeconv import response_from_mask_ssst
import nibabel as nib
from dipy.denoise.localpca import mppca
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sr2csd.utils.sh_basis import convert_sh_descoteaux_tournier
from sr2csd.utils.response import select_voxel
from sr2csd import MRtrixCSDModel, SR2CSDModel

def main() -> None:
    data, data_affine = load_nifti(f"{ROOT}/data/hardi/denoised_hardi_SNR-20.nii.gz")
    bvals = np.loadtxt(f"{ROOT}/data/hardi/hardi-scheme.bval", dtype=float)
    bvecs = np.loadtxt(f"{ROOT}/data/hardi/hardi-scheme.bvec", dtype=float)
    gtab = gradient_table(bvals, bvecs)

    mask, _ = load_nifti(f"{ROOT}/data/hardi/mask.nii")

    # denoise data
    #data = mppca(data_noisy, patch_radius=2)
    #save_nifti(f"{ROOT}/data/hardi/denoised_hardi_SNR-20.nii.gz", data, data_affine)
    response_mask = select_voxel(gtab, data, roi_radii=10, voxel_nb=200)
    response, ratio = response_from_mask_ssst(gtab, data, response_mask)

    lmax = 12

    #csd_model = MRtrixCSDModel(gtab, response, sh_order=lmax, tau=0.1, convergence=30, eta=1.0)
    #csd_fit = csd_model.fit(data) # can load mask
    #csd_coeff = csd_fit.shm_coeff
    #tournier_base_csd = convert_sh_descoteaux_tournier(csd_coeff)
    #fods_img_csd = nib.Nifti1Image(tournier_base_csd, data_affine)
    #nib.save(fods_img_csd,  f"/home/ekin/Desktop/comparison/new/csd{lmax}_v2.nii.gz")


    sr2csd = SR2CSDModel(
        gtab,
        response,
        sh_order=lmax,
        rho=1.0,
        tau=0.1,
        convergence=30,
        eta=1.0,
        denoise_mask=f"{ROOT}/data/hardi/mask.nii",
        csd_use_mask=False,  # set True only if your CSD prior used a mask
    )

    sr2csd_fit = sr2csd.fit(data)  # uses prior pipeline by default

    #csd_coeff = csd_fit.shm_coeff
    sr2csd_coeff = sr2csd_fit.shm_coeff

    #tournier_base_csd = convert_sh_descoteaux_tournier(csd_coeff)
    #fods_img_csd = nib.Nifti1Image(tournier_base_csd, data_affine)
    #nib.save(fods_img_csd,  f"/home/ekin/Desktop/sr2-csd/tests/csd8_test1.nii.gz")

    tournier_base_sr2csd = convert_sh_descoteaux_tournier(sr2csd_coeff)
    fods_img_sr2csd = nib.Nifti1Image(tournier_base_sr2csd, data_affine)
    nib.save(fods_img_sr2csd,  f"/home/ekin/Desktop/comparison/new/sr2csd_v5.nii.gz")

    print("CSD and SR2-CSD ran successfully.")
    print(f"Response ratio (lambda2/lambda1): {ratio:.6f}")
    #print(f"CSD coeff shape: {csd_fit.shm_coeff.shape}")
    print(f"SR2 coeff shape: {sr2csd_fit.shm_coeff.shape}")


if __name__ == "__main__":
    main()
