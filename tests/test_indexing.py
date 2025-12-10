#!/usr/bin/env python3
"""Property tests comparing RasterIndex with PandasIndex for indexing operations."""

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from xarray.testing.strategies import (
    basic_indexers,
    outer_array_indexers,
    vectorized_indexers,
)

from rasterix import RasterIndex
from rasterix.strategies import (
    basic_label_indexers,
    outer_array_label_indexers,
    vectorized_label_indexers,
)


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


@given(data=st.data())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_isel_basic_indexing_equivalence(data, raster_da, pandas_da):
    """Test that isel produces identical results for RasterIndex and PandasIndex."""
    sizes = dict(raster_da.sizes)
    indexers = data.draw(basic_indexers(sizes=sizes))
    result_raster = raster_da.isel(indexers)
    result_pandas = pandas_da.isel(indexers)
    xr.testing.assert_identical(result_raster, result_pandas)


@given(data=st.data())
@settings(
    deadline=None,
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
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_outer_array_label_indexing(data, raster_da, pandas_da):
    """Test that outer array label indexing produces identical results for RasterIndex and PandasIndex."""
    indexers = data.draw(outer_array_label_indexers(indexes=pandas_da.xindexes))
    result_raster = raster_da.sel(indexers, method="nearest")
    result_pandas = pandas_da.sel(indexers, method="nearest")
    xr.testing.assert_identical(result_raster, result_pandas)


@given(data=st.data())
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_vectorized_indexing(data, raster_da, pandas_da):
    """Test that vectorized indexing produces identical results for RasterIndex and PandasIndex."""
    sizes = dict(raster_da.sizes)
    indexers = data.draw(vectorized_indexers(sizes=sizes))
    result_raster = raster_da.isel(indexers)
    result_pandas = pandas_da.isel(indexers)
    xr.testing.assert_identical(result_raster, result_pandas)


@given(data=st.data())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_vectorized_label_indexing(data, raster_da, pandas_da):
    """Test that vectorized label indexing produces identical results for RasterIndex and PandasIndex."""
    indexers = data.draw(vectorized_label_indexers(indexes=pandas_da.xindexes))
    result_raster = raster_da.sel(indexers, method="nearest")
    result_pandas = pandas_da.sel(indexers, method="nearest")
    xr.testing.assert_identical(result_raster, result_pandas)
