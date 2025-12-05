"""Hypothesis strategies for generating label-based indexers."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from hypothesis import note
from hypothesis import strategies as st
from xarray.core.indexes import Indexes
from xarray.testing.strategies import (
    basic_indexers,
    outer_array_indexers,
    vectorized_indexers,
)


def pos_to_label_indexer(idx: pd.Index, idxr: int | slice | np.ndarray, *, use_scalar: bool = True) -> Any:
    """Convert a positional indexer to a label-based indexer.

    Parameters
    ----------
    idx : pd.Index
        The pandas Index to use for label lookup.
    idxr : int | slice | np.ndarray
        The positional indexer (integer, slice, or array of integers).
    use_scalar : bool, optional
        If True, attempt to convert scalar values to Python scalars. Default is True.

    Returns
    -------
    Any
        The label-based indexer (scalar, slice, or array of labels).
    """
    if isinstance(idxr, slice):
        return slice(
            None if idxr.start is None else idx[idxr.start],
            # FIXME: This will never go past the label range
            None if idxr.stop is None else idx[min(idxr.stop, idx.size - 1)],
        )
    elif isinstance(idxr, np.ndarray):
        # Convert array of position indices to array of label values
        return idx[idxr].values
    else:
        val = idx[idxr]
        if use_scalar:
            try:
                # pass python scalars occasionally
                val = val.item()
            except Exception:
                note(f"casting {val!r} to item() failed")
                pass
        return val


@st.composite
def basic_label_indexers(draw, /, *, indexes: Indexes) -> dict[Hashable, float | slice]:
    """Generate label-based indexers by converting position indexers to labels.

    This works in label space by using the coordinate Index values.

    Parameters
    ----------
    draw : callable
        The Hypothesis draw function (automatically provided by @st.composite).
    indexes : Indexes
        Dictionary mapping dimension names to their associated indexes

    Returns
    -------
    dict[Hashable, float | slice]
        Label-based indexers as a dict with keys from sizes.keys().
        Values are either float (for scalar labels) or slice (for label ranges).
    """
    idxs = indexes.get_unique()
    assert all(isinstance(idx, xr.indexes.PandasIndex) for idx in idxs)

    # FIXME: this should be indexes.sizes!
    sizes = indexes.dims

    pos_indexer = draw(basic_indexers(sizes=sizes))
    pdindexes = indexes.to_pandas_indexes()

    label_indexer = {
        dim: pos_to_label_indexer(pdindexes[dim], idx, use_scalar=draw(st.booleans()))
        for dim, idx in pos_indexer.items()
    }
    return label_indexer


@st.composite
def outer_array_label_indexers(draw, /, *, indexes: Indexes) -> dict[Hashable, np.ndarray]:
    """Generate label-based outer array indexers by converting position indexers to labels.

    This works in label space by using the coordinate Index values.

    Parameters
    ----------
    draw : callable
        The Hypothesis draw function (automatically provided by @st.composite).
    indexes : Indexes
        Dictionary mapping dimension names to their associated indexes

    Returns
    -------
    dict[Hashable, np.ndarray]
        Label-based indexers as a dict with keys from indexes.
        Values are numpy arrays of label values for each dimension.
    """
    idxs = indexes.get_unique()
    assert all(isinstance(idx, xr.indexes.PandasIndex) for idx in idxs)

    # FIXME: this should be indexes.sizes!
    sizes = indexes.dims

    pos_indexer = draw(outer_array_indexers(sizes=sizes))
    pdindexes = indexes.to_pandas_indexes()

    label_indexer = {
        dim: pos_to_label_indexer(pdindexes[dim], idx, use_scalar=False) for dim, idx in pos_indexer.items()
    }
    return label_indexer


@st.composite
def vectorized_label_indexers(draw, /, *, indexes: Indexes, **kwargs) -> dict[Hashable, xr.DataArray]:
    """Generate label-based vectorized indexers by converting position indexers to labels.

    This works in label space by using the coordinate Index values.

    Parameters
    ----------
    draw : callable
        The Hypothesis draw function (automatically provided by @st.composite).
    indexes : Indexes
        Dictionary mapping dimension names to their associated indexes
    **kwargs : dict
        Additional keyword arguments to pass to vectorized_indexers

    Returns
    -------
    dict[Hashable, xr.DataArray]
        Label-based indexers as a dict with keys from indexes.
        Values are DataArrays of label values for each dimension.
    """
    idxs = indexes.get_unique()
    assert all(isinstance(idx, xr.indexes.PandasIndex) for idx in idxs)

    # FIXME: this should be indexes.sizes!
    sizes = indexes.dims

    pos_indexer = draw(vectorized_indexers(sizes=sizes, **kwargs))
    pdindexes = indexes.to_pandas_indexes()

    label_indexer = {}
    for dim, idx_array in pos_indexer.items():
        # Convert each position in the array to its corresponding label
        # Flatten, index, then reshape back to original shape
        flat_indices = idx_array.values.ravel()
        flat_labels = pdindexes[dim][flat_indices].values
        label_values = flat_labels.reshape(idx_array.shape)
        label_indexer[dim] = xr.DataArray(label_values, dims=idx_array.dims)

    return label_indexer
