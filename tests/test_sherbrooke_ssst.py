"""Run standard SSST-CSD and SR2-CSD on Sherbrooke dataset."""

import ssl
import sys
from pathlib import Path
from urllib.error import URLError

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.denoise.localpca import mppca
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.csdeconv import response_from_mask_ssst
from dipy.segment.mask import median_otsu
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sr2csd import MRtrixCSDModel, SR2CSDModel
from sr2csd.utils.response import select_voxel
from sr2csd.utils.sh_basis import convert_sh_descoteaux_tournier


def _get_sherbrooke_fnames() -> tuple[str, str, str]:
    try:
        return get_fnames("sherbrooke_3shell")
    except URLError:
        # Some local Python installs do not expose a trusted system CA bundle.
        try:
            import certifi
        except ImportError as exc:
            raise RuntimeError(
                "Failed to download DIPY sherbrooke_3shell dataset due SSL certificate "
                "verification. Install certifi (`pip install certifi`) and rerun."
            ) from exc

        ssl._create_default_https_context = lambda: ssl.create_default_context(
            cafile=certifi.where()
        )
        try:
            return get_fnames("sherbrooke_3shell")
        except URLError as exc:
            raise RuntimeError(
                "Failed to download DIPY sherbrooke_3shell dataset due SSL certificate "
                "verification. Retry after trusting your local certificates (for macOS "
                "Python.org builds, run Install Certificates.command)."
            ) from exc


def main() -> None:
    out_dir = ROOT / "data" / "sherbrooke"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data from DiPy's Sherbrooke dataset
    fraw, fbval, fbvec = _get_sherbrooke_fnames()
    data_noisy, data_affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[:, 0] *= -1

    # extract brain mask
    b_sel = np.where(bvals == 0)
    b0_data = data_noisy[..., b_sel]
    print(np.shape(b0_data))
    b0_mask, mask = median_otsu(b0_data, median_radius=2, numpass=1)
    mask = mask[:, :, :, 0, 0]
    save_nifti(str(out_dir / "brain_mask.nii.gz"), np.uint8(mask), data_affine)

    # denoise data
    data = mppca(data_noisy, patch_radius=2)
    #save_nifti(str(out_dir / "denoised_sherbrooke_data.nii.gz"), data, data_affine)

    # single-shell selection
    sel_b = np.logical_or(bvals == 0, bvals == 3500)
    data = data[..., sel_b]
    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])
    response_mask = select_voxel(gtab, data, roi_radii=10, voxel_nb=200)
    response, ratio = response_from_mask_ssst(gtab, data, response_mask)

    lmax = 12

    csd_model = MRtrixCSDModel(gtab, response, sh_order=lmax, tau=0.1, convergence=30, eta=1.0)
    csd_fit = csd_model.fit(data)  # can load mask
    csd_coeff = csd_fit.shm_coeff
    tournier_base_csd = convert_sh_descoteaux_tournier(csd_coeff)
    fods_img_csd = nib.Nifti1Image(tournier_base_csd, data_affine)
    nib.save(fods_img_csd, str(out_dir / f"csd{lmax}.nii.gz"))

    sr2csd = SR2CSDModel(
        gtab,
        response,
        sh_order=lmax,
        rho=1.0,
        tau=0.1,
        convergence=30,
        eta=1.0,
        denoise_mask=mask,
        csd_use_mask=False,  # set True only if the CSD prior used a mask
    )

    sr2csd_fit = sr2csd.fit(data)  # uses prior pipeline by default

    sr2csd_coeff = sr2csd_fit.shm_coeff

    tournier_base_sr2csd = convert_sh_descoteaux_tournier(sr2csd_coeff)
    fods_img_sr2csd = nib.Nifti1Image(tournier_base_sr2csd, data_affine)
    nib.save(fods_img_sr2csd, str(out_dir / "sr2csd.nii.gz"))

    print("CSD and SR2-CSD ran successfully.")
    #print(f"CSD coeff shape: {csd_fit.shm_coeff.shape}")
    print(f"SR2 coeff shape: {sr2csd_fit.shm_coeff.shape}")


if __name__ == "__main__":
    main()
