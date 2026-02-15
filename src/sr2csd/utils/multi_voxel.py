"""Tools to easily make multi voxel models (SR²-CSD local copy).

This is kept intentionally close to DIPY's multi_voxel utilities to ease
future upstreaming. The only substantive change is that tqdm is optional.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

try:
    from tqdm import tqdm 
except Exception: 
    tqdm = None

from dipy.core.ndindex import ndindex
from dipy.reconst.quick_squash import quick_squash as _squash
from dipy.reconst.base import ReconstFit


def _progress_bar(total, verbose: bool):
    """Return a tqdm progress bar if available and requested."""
    if (not verbose) or (tqdm is None):
        return None
    return tqdm(total=int(total), position=0)


def multi_voxel_fit_qp(single_voxel_fit_qp):
    """
    Decorator to turn a single-voxel QP fit into a multi-voxel fit.

    Expected single_voxel_fit_qp signature:
        single_voxel_fit_qp(self, data_1d, ijk, verbose=True) -> fit_object
    """
    def new_fit_qp(self, data, mask=None, verbose=False):
        # If only one voxel just return a normal fit
        if data.ndim == 1:
            # For single voxel, pass a dummy index
            return single_voxel_fit_qp(self, data, None, verbose=verbose)

        # Make a mask if mask is None
        if mask is None:
            shape = data.shape[:-1]
            strides = (0,) * len(shape)
            mask = as_strided(np.array(True), shape=shape, strides=strides)
        # Check the shape of the mask if mask is not None
        elif mask.shape != data.shape[:-1]:
            raise ValueError("mask and data shape do not match")

        # Fit data where mask is True
        fit_array = np.empty(data.shape[:-1], dtype=object)
        bar = _progress_bar(np.sum(mask), verbose=verbose)

        for ijk in ndindex(data.shape[:-1]):
            if mask[ijk]:
                fit_array[ijk] = single_voxel_fit_qp(self, data[ijk], ijk, verbose=verbose)
                if bar is not None:
                    bar.update()

        if bar is not None:
            bar.close()

        return MultiVoxelFit(self, fit_array, mask)

    return new_fit_qp


def multi_voxel_fit(single_voxel_fit):
    """Method decorator to turn a single voxel model fit definition into
    a multi voxel model fit definition.
    """
    def new_fit(self, data, mask=None, verbose=False):
        """Fit method for every voxel in data."""
        # If only one voxel just return a normal fit
        if data.ndim == 1:
            return single_voxel_fit(self, data)

        # Make a mask if mask is None
        if mask is None:
            shape = data.shape[:-1]
            strides = (0,) * len(shape)
            mask = as_strided(np.array(True), shape=shape, strides=strides)
        # Check the shape of the mask if mask is not None
        elif mask.shape != data.shape[:-1]:
            raise ValueError("mask and data shape do not match")

        # Fit data where mask is True
        fit_array = np.empty(data.shape[:-1], dtype=object)
        bar = _progress_bar(np.sum(mask), verbose=verbose)

        for ijk in ndindex(data.shape[:-1]):
            if mask[ijk]:
                fit_array[ijk] = single_voxel_fit(self, data[ijk])
                if bar is not None:
                    bar.update()

        if bar is not None:
            bar.close()

        return MultiVoxelFit(self, fit_array, mask)

    return new_fit


class MultiVoxelFit(ReconstFit):
    """Holds an array of fits and allows access to their attributes and methods."""
    def __init__(self, model, fit_array, mask):
        self.model = model
        self.fit_array = fit_array
        self.mask = mask

    @property
    def shape(self):
        return self.fit_array.shape

    def __getattr__(self, attr):
        result = CallableArray(self.fit_array.shape, dtype=object)
        for ijk in ndindex(result.shape):
            if self.mask[ijk]:
                result[ijk] = getattr(self.fit_array[ijk], attr)
        return _squash(result, self.mask)

    def __getitem__(self, index):
        item = self.fit_array[index]
        if isinstance(item, np.ndarray):
            return MultiVoxelFit(self.model, item, self.mask[index])
        else:
            return item

    def predict(self, *args, **kwargs):
        """
        Predict for the multi-voxel object using each single-object's prediction
        API, with S0 provided from an array.
        """
        S0 = kwargs.get("S0", np.ones(self.fit_array.shape))
        idx = ndindex(self.fit_array.shape)
        ijk = next(idx)

        def gimme_S0(S0_in, ijk_in):
            if isinstance(S0_in, np.ndarray):
                return S0_in[ijk_in]
            else:
                return S0_in

        kwargs["S0"] = gimme_S0(S0, ijk)

        # If we have a mask, we might have some Nones up front, skip those:
        while self.fit_array[ijk] is None:
            ijk = next(idx)

        if not hasattr(self.fit_array[ijk], "predict"):
            raise NotImplementedError("This model does not have prediction implemented yet")

        first_pred = self.fit_array[ijk].predict(*args, **kwargs)
        result = np.zeros(self.fit_array.shape + (first_pred.shape[-1],))
        result[ijk] = first_pred

        for ijk in idx:
            kwargs["S0"] = gimme_S0(S0, ijk)
            # If it's masked, we predict a 0:
            if self.fit_array[ijk] is None:
                result[ijk] *= 0
            else:
                result[ijk] = self.fit_array[ijk].predict(*args, **kwargs)

        return result


class CallableArray(np.ndarray):
    """An array which can be called like a function."""
    def __call__(self, *args, **kwargs):
        result = np.empty(self.shape, dtype=object)
        for ijk in ndindex(self.shape):
            item = self[ijk]
            if item is not None:
                result[ijk] = item(*args, **kwargs)
        return _squash(result)
