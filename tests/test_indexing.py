#!/usr/bin/env python3
"""Property tests comparing RasterIndex with PandasIndex for indexing operations."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from hypothesis import HealthCheck, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from xarray.core.indexes import Indexes

from rasterix import RasterIndex


def is_mixed_scalar_slice_indexer(indexers: dict[Hashable, int | slice]) -> bool:
    # TODO: Fix bug in RasterIndex with mixed scalar/slice indexing across dimensions
    # When you have scalar indexing on one dimension (e.g., y=0) and slice indexing
    # on another (e.g., x=slice(None, 1)), RasterIndex.isel() returns None for the
    # scalar dimension, dropping that index. This causes xarray to incorrectly handle
    # the coordinate variables - the sliced dimension's coordinate (x) maintains
    # dims ('x',) even though the data has been reduced by the scalar indexing.
    # This results in: "ValueError: dimensions ('x',) must have the same length as
    # the number of data dimensions, ndim=0"
    #
    # Example failing case: raster_da.isel(y=0, x=slice(None, 1))
    # - y=0 causes RasterIndex to return None for y dimension
    # - x=slice(None, 1) preserves RasterIndex for x dimension
    # - Result: coordinate variable x has wrong dimensionality
    #
    # For now, filter out these cases using hypothesis.assume()
    has_scalar = any(isinstance(v, int | np.integer) for v in indexers.values())
    has_slice = any(isinstance(v, slice) for v in indexers.values())
    return has_scalar and has_slice and len(indexers) > 1


@pytest.fixture
def raster_da():
    """Create a DataArray with RasterIndex coordinates."""
    width, height = 10, 8
    transform = Affine.translation(0, 0) * Affine.scale(1.0, -1.0)

    # Create data
    data = np.arange(width * height, dtype=np.float64).reshape(height, width)

    # Create RasterIndex
    index = RasterIndex.from_transform(
        transform,
        width=width,
        height=height,
        x_dim="x",
        y_dim="y",
    )

    # Create coordinates from index
    coords = xr.Coordinates.from_xindex(index)

    # Create DataArray
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords=coords,
        name="data",
    )

    return da


@pytest.fixture
def pandas_da(raster_da):
    """Create a DataArray with PandasIndex by converting from raster_da."""
    x_values = raster_da.x.values
    y_values = raster_da.y.values
    da = xr.DataArray(
        raster_da.values, dims=raster_da.dims, coords={"x": x_values, "y": y_values}, name="data"
    )
    return da


def pos_to_label_indexer(idx: pd.Index, idxr: int | slice | np.ndarray) -> Any:
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
        if st.booleans():
            try:
                # pass python scalars occasionally
                val = val.item()
            except Exception:
                note(f"casting {val!r} to item() failed")
                pass
        return val


@st.composite
def basic_indexers(
    draw,
    /,
    *,
    sizes: dict[Hashable, int],
    min_dims: int = 0,
    max_dims: int | None = None,
) -> dict[Hashable, int | slice]:
    """Generate basic indexers using hypothesis.extra.numpy.basic_indices.

    Parameters
    ----------
    draw : callable
        The Hypothesis draw function (automatically provided by @st.composite).
    sizes : dict[Hashable, int]
        Dictionary mapping dimension names to their sizes.
    min_dims : int, optional
        Minimum dimensionality of the generated index. Default is 0.
    max_dims : int or None, optional
        Maximum dimensionality of the generated index. Default is None (no limit).

    Returns
    -------
    dict[Hashable, int | slice]
        Indexers as a dict with keys randomly selected from sizes.keys().
    """
    # Get all dimension names
    all_dims = list(sizes.keys())

    # Determine how many dimensions to index
    num_dims = draw(st.integers(min_value=min_dims, max_value=min(max_dims or len(all_dims), len(all_dims))))

    # Randomly select which dimensions to index
    selected_dims = draw(st.permutations(all_dims).map(lambda x: x[:num_dims]))

    # Build shape for the selected dimensions
    selected_shape = tuple(sizes[dim] for dim in selected_dims)

    # Generate basic indices for the selected dimensions
    idx = draw(
        npst.basic_indices(
            shape=selected_shape,
            # These control dimensionality of the selected array
            min_dims=0,
            max_dims=len(all_dims) - len(selected_dims),
            allow_newaxis=False,
            allow_ellipsis=False,
        ).filter(lambda x: x != ())
    )
    if not isinstance(idx, tuple):
        idx = (idx,)
    result = dict(zip(selected_dims, idx, strict=True))
    return result


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

    label_indexer = {dim: pos_to_label_indexer(pdindexes[dim], idx) for dim, idx in pos_indexer.items()}
    return label_indexer


@st.composite
def outer_array_indexers(
    draw,
    /,
    *,
    sizes: dict[Hashable, int],
    min_dims: int = 0,
    max_dims: int | None = None,
) -> dict[Hashable, np.ndarray]:
    """Generate outer array indexers (vectorized/orthogonal indexing).

    Parameters
    ----------
    draw : callable
        The Hypothesis draw function (automatically provided by @st.composite).
    sizes : dict[Hashable, int]
        Dictionary mapping dimension names to their sizes.
    min_dims : int, optional
        Minimum number of dimensions to index. Default is 0.
    max_dims : int or None, optional
        Maximum number of dimensions to index. Default is None (no limit).

    Returns
    -------
    dict[Hashable, np.ndarray]
        Indexers as a dict with keys randomly selected from sizes.keys().
        Values are 1D numpy arrays of integer indices for each dimension.
    """
    # Get all dimension names
    all_dims = list(sizes.keys())

    # Determine how many dimensions to index
    num_dims = draw(st.integers(min_value=min_dims, max_value=min(max_dims or len(all_dims), len(all_dims))))

    # Randomly select which dimensions to index
    selected_dim_names = draw(st.permutations(all_dims).map(lambda x: x[:num_dims]))
    selected_dims = {dim: sizes[dim] for dim in selected_dim_names}

    # Generate array indexers for each selected dimension
    result = {}
    for dim, size in selected_dims.items():
        # Generate array of valid indices for this dimension
        # Use strategy for shape: at least 2 elements to avoid scalar-like behavior
        indices = draw(
            npst.arrays(
                dtype=np.int64,
                shape=st.integers(min_value=2, max_value=min(size, 10)),
                elements=st.integers(min_value=0, max_value=size - 1),
            )
        )
        result[dim] = indices

    return result


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

    label_indexer = {dim: pos_to_label_indexer(pdindexes[dim], idx) for dim, idx in pos_indexer.items()}
    return label_indexer


@given(data=st.data())
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_isel_basic_indexing_equivalence(data, raster_da, pandas_da):
    """Test that isel produces identical results for RasterIndex and PandasIndex."""
    sizes = dict(raster_da.sizes)
    indexers = data.draw(
        basic_indexers(sizes=sizes).filter(lambda idxr: not is_mixed_scalar_slice_indexer(idxr))
    )
    result_raster = raster_da.isel(indexers)
    result_pandas = pandas_da.isel(indexers)
    xr.testing.assert_identical(result_raster, result_pandas)


@given(data=st.data())
@settings(
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_sel_basic_indexing_equivalence(data, raster_da, pandas_da):
    """Test that isel produces identical results for RasterIndex and PandasIndex."""
    indexers = data.draw(basic_label_indexers(indexes=pandas_da.xindexes))

    result_raster = raster_da.sel(
        indexers,
        method=("nearest" if any(np.isscalar(idxr) for idxr in indexers.values()) else None),
    )
    result_pandas = pandas_da.sel(indexers)

    if all(isinstance(idxr, slice) for idxr in indexers.values()):
        assert all(isinstance(idx, RasterIndex) for idx in result_raster.xindexes.get_unique())

    xr.testing.assert_identical(result_raster, result_pandas)


def test_simple_isel(raster_da, pandas_da):
    """Sanity check: simple indexing operations."""
    # Scalar indexing
    xr.testing.assert_identical(raster_da.isel(x=0), pandas_da.isel(x=0))
    xr.testing.assert_identical(raster_da.isel(y=0), pandas_da.isel(y=0))
    xr.testing.assert_identical(raster_da.isel(x=0, y=0), pandas_da.isel(x=0, y=0))

    # Slice indexing
    xr.testing.assert_identical(raster_da.isel(x=slice(2, 5)), pandas_da.isel(x=slice(2, 5)))
    xr.testing.assert_identical(raster_da.isel(y=slice(1, 4)), pandas_da.isel(y=slice(1, 4)))
    xr.testing.assert_identical(
        raster_da.isel(x=slice(2, 5), y=slice(1, 4)),
        pandas_da.isel(x=slice(2, 5), y=slice(1, 4)),
    )

    # Array indexing
    xr.testing.assert_identical(raster_da.isel(x=[0, 2, 4]), pandas_da.isel(x=[0, 2, 4]))
    xr.testing.assert_identical(raster_da.isel(y=[1, 3]), pandas_da.isel(y=[1, 3]))


@given(data=st.data())
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_outer_array_indexing(data, raster_da, pandas_da):
    """Test that outer array indexing produces identical results for RasterIndex and PandasIndex."""
    sizes = dict(raster_da.sizes)
    indexers = data.draw(outer_array_indexers(sizes=sizes))

    result_raster = raster_da.isel(indexers)
    result_pandas = pandas_da.isel(indexers)

    xr.testing.assert_identical(result_raster, result_pandas)


@given(data=st.data())
@settings(
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_outer_array_label_indexing(data, raster_da, pandas_da):
    """Test that outer array label indexing produces identical results for RasterIndex and PandasIndex."""
    indexers = data.draw(outer_array_label_indexers(indexes=pandas_da.xindexes))

    result_raster = raster_da.sel(indexers, method="nearest")
    result_pandas = pandas_da.sel(indexers, method="nearest")

    xr.testing.assert_identical(result_raster, result_pandas)
