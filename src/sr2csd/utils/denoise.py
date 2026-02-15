import time

import numpy as np


def denoise_tv_chambolle_K_4d(
    data_4d,
    K,
    *,
    mask=None,
    eps=2e-4,
    max_num_iter=200,
):
    """
    Denoise a 4D volume using TV-Chambolle, scaling sigma by K per volume.
    """
    data_4d = np.asarray(data_4d)
    if data_4d.ndim != 4:
        raise ValueError("data_4d must be 4D (X, Y, Z, N)")

    try:
        from skimage.restoration import denoise_tv_chambolle, estimate_sigma
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "scikit-image is required for denoising. Install with `pip install scikit-image`."
        ) from exc

    denoised = np.zeros_like(data_4d, dtype=float)
    for v in range(data_4d.shape[3]):
        data_vol = data_4d[..., v]
        if mask is not None:
            data_vol = data_vol * mask
            values = data_vol[mask != 0]
        else:
            values = data_vol.reshape(-1)

        sigma_est = np.mean(estimate_sigma(values, channel_axis=None))
        scaled_sigma = K * sigma_est
        denoised[..., v] = denoise_tv_chambolle(
            data_vol,
            weight=scaled_sigma,
            eps=eps,
            max_num_iter=max_num_iter,
            channel_axis=None,
        )

    return denoised


def calibrate_tv_chambolle_4d(
    data_4d,
    K_range,
    *,
    mask=None,
    eps=2e-4,
    max_num_iter=200,
):
    """
    Calibrate TV-Chambolle denoising over a range of K values.
    """
    try:
        from skimage.restoration import calibrate_denoiser
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "scikit-image is required for denoising. Install with `pip install scikit-image`."
        ) from exc

    K_range = np.asarray(K_range)

    def _denoise_fn(data, K):
        return denoise_tv_chambolle_K_4d(
            data,
            K,
            mask=mask,
            eps=eps,
            max_num_iter=max_num_iter,
        )

    _, (params_tested, losses) = calibrate_denoiser(
        data_4d,
        _denoise_fn,
        denoise_parameters={"K": K_range},
        extra_output=True,
    )
    best = params_tested[int(np.argmin(losses))]
    return best, params_tested, losses


def calibrate_and_denoise_tv_chambolle_4d(
    sh_coeff,
    *,
    K_range=None,
    mask=None,
    eps=2e-4,
    max_num_iter=200,
    verbose=True,
):
    """
    Exact calibration + denoise sequence with timing prints.
    """
    if verbose:
        print("J-invariance start time: ", time.asctime())

    if K_range is None:
        K_range = np.arange(0.25, 5, 0.1)

    K_range = {"K": np.asarray(K_range)}

    best_params, parameters_tested, losses = calibrate_tv_chambolle_4d(
        sh_coeff,
        K_range["K"],
        mask=mask,
        eps=eps,
        max_num_iter=max_num_iter,
    )

    if verbose:
        print(best_params)
        print("denoising starting...")

    sh_coeff_new = denoise_tv_chambolle_K_4d(
        sh_coeff,
        best_params["K"],
        mask=mask,
        eps=eps,
        max_num_iter=max_num_iter,
    )

    if verbose:
        print("J-invariance + denoising stop time: ", time.asctime())

    return sh_coeff_new, best_params, parameters_tested, losses


def denoise_tv_chambolle_4d(*args, **kwargs):
    """Backward-compatible alias."""
    return denoise_tv_chambolle_K_4d(*args, **kwargs)
