# SR²-CSD: Spatially Regularized Super-Resolved Constrained Spherical Deconvolution

This repository provides a reference implementation of **SR²-CSD**, a spatially
regularized extension of super-resolved constrained spherical deconvolution
(CSD) of diffusion MRI data.

SR²-CSD incorporates a spatial prior into the classical CSD framework to improve
the stability, spatial coherence, and reproducibility of fiber orientation
distribution (FOD) estimation, while preserving angular super-resolution.

---

## Method overview

Standard CSD estimates FODs independently in each voxel, relying solely on
non-negativity constraints for regularization. While Super-CSD enables higher
angular resolution by increasing the spherical harmonic order, it is more
sensitive to noise and often produces spatially incoherent FODs.

SR²-CSD addresses this limitation by:
- Constructing a **spatial prior** from a Total Variation denoised Super-CSD solution
- Automatically calibrating the denoising strength using a **self-supervised J-invariant denoising approach**
- Solving a **quadratic programming problem** that balances data fidelity,
  spatial regularity, and non-negativity constraints

This formulation stabilizes super-resolved FOD estimation and improves angular
accuracy, spatial coherence, and test–retest reproducibility.

---

## Compatibility

- Compatible with **single-shell / multi-shell** and **single-tissue / multi-tissue** variants
- Implemented on top of **DIPY**, enabling integration with existing pipelines

---

## Status

This repository is under active development.
The **method is finalized and published**, but the software API may evolve.

---

## Citation

> Taskin, E., Girard, G., Haro, J. L. V., Rafael-Patiño, J.,
> Garyfallidis, E., Thiran, J. P., Canales-Rodríguez, E. J.,
> *Spatially regularized super-resolved constrained spherical deconvolution (SR²-CSD) of diffusion MRI data*,
> NeuroImage, vol. 325, 2026, 121656.
> https://doi.org/10.1016/j.neuroimage.2025.121656

---

## License
This project is released under the BSD 3-Clause License.
