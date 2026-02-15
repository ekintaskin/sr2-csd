# sr2-csd
Spatially Regularized Super-Resolved Constrained Spherical Deconvolution (SR2-CSD) of diffusion MRI data.

## Install
```bash
pip install -e .
```

Dependencies include `dipy`, `scikit-image`, and `PyWavelets`. The default
`SR2CSDModel.fit(...)` pipeline uses TV-Chambolle denoising via scikit-image,
which depends on PyWavelets for sigma estimation.

If you want to skip the denoising pipeline, set:
```python
SR2CSDModel(..., use_prior_pipeline=False)
```

## Minimal usage
```python
from sr2csd import SR2CSDModel

model = SR2CSDModel(
    gtab,
    response,
    sh_order=12,
    rho=0.1,
    tau=0.1,
    convergence=30,
    eta=1.0,
)
fit = model.fit(data, mask=mask)
```
