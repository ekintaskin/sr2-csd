import numbers
import warnings

import numpy as np
from dipy.reconst.csdeconv import _mask_from_roi, _roi_in_volume
from dipy.reconst.dti import TensorModel, fractional_anisotropy


def select_voxel(gtab, data, roi_center=None, roi_radii=10,
                           voxel_nb = 200):
    """
    Select high-FA voxels within an ROI for response estimation.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table.
    data : ndarray
        4D diffusion data (X, Y, Z, N).
    roi_center : array-like, optional
        Center of the ROI. Defaults to volume center.
    roi_radii : int or tuple, optional
        Radius (or radii) of the ROI in voxels.
    voxel_nb : int, optional
        Number of voxels to select with highest FA.
    """
    if len(data.shape) < 4:
        msg = """Data must be 4D (3D image + directions). To use a 2D image,
        please reshape it into a (N, N, 1, ndirs) array."""
        raise ValueError(msg)

    if isinstance(roi_radii, numbers.Number):
        roi_radii = (roi_radii, roi_radii, roi_radii)

    if roi_center is None:
        roi_center = np.array(data.shape[:3]) // 2

    roi_radii = _roi_in_volume(data.shape, np.asarray(roi_center),
                               np.asarray(roi_radii))

    roi_mask = _mask_from_roi(data.shape[:3], roi_center, roi_radii)

    ten = TensorModel(gtab)
    tenfit = ten.fit(data, mask=roi_mask)
    fa = fractional_anisotropy(tenfit.evals)
    fa[np.isnan(fa)] = 0

    mask = np.zeros(fa.shape, dtype=np.int64)
    
    flat = fa.flatten()
    indices = np.argpartition(flat, -voxel_nb)[-voxel_nb:]
    indices = indices[np.argsort(-flat[indices])]
    final_idx = np.unravel_index(indices, fa.shape)
    mask[final_idx]=1

    if np.sum(mask) == 0:
        msg = """No voxel with a FA higher than were found.
        Try a larger roi or a lower threshold."""
        warnings.warn(msg, UserWarning)

    return mask
