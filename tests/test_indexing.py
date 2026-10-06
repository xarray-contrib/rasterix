#!/usr/bin/env python3
"""Property tests comparing RasterIndex with PandasIndex for indexing operations."""

import math

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from hypothesis import HealthCheck, assume, example, given, settings
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


def assert_identical(actual: xr.DataArray, expected: xr.DataArray):
    """Assert DataArrays are identical, comparing RasterIndex to equivalent PandasIndex."""
    xr.testing.assert_equal(actual, expected)

    for idx in actual.xindexes.get_unique():
        if isinstance(idx, RasterIndex):
            pandas_indexes = idx._as_pandas_index()
            for dim, pandas_idx in pandas_indexes.items():
                expected_idx = expected.xindexes[dim]
                assert pandas_idx.equals(expected_idx)


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
    assert_identical(result_raster, result_pandas)


@given(data=st.data())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_sel_basic_indexing_equivalence(data, raster_da, pandas_da):
    """Test that sel produces identical results for RasterIndex and PandasIndex."""
    indexers = data.draw(basic_label_indexers(indexes=pandas_da.xindexes))

    result_raster = raster_da.sel(
        indexers,
        method=("nearest" if any(np.isscalar(idxr) for idxr in indexers.values()) else None),
    )
    result_pandas = pandas_da.sel(indexers)

    if all(isinstance(idxr, slice) for idxr in indexers.values()):
        assert all(isinstance(idx, RasterIndex) for idx in result_raster.xindexes.get_unique())

    assert_identical(result_raster, result_pandas)


def test_simple_isel(raster_da, pandas_da):
    """Sanity check: simple indexing operations."""
    # Scalar indexing
    assert_identical(raster_da.isel(x=0), pandas_da.isel(x=0))
    assert_identical(raster_da.isel(y=0), pandas_da.isel(y=0))
    assert_identical(raster_da.isel(x=0, y=0), pandas_da.isel(x=0, y=0))

    # Slice indexing
    assert_identical(raster_da.isel(x=slice(2, 5)), pandas_da.isel(x=slice(2, 5)))
    assert_identical(raster_da.isel(y=slice(1, 4)), pandas_da.isel(y=slice(1, 4)))
    assert_identical(
        raster_da.isel(x=slice(2, 5), y=slice(1, 4)),
        pandas_da.isel(x=slice(2, 5), y=slice(1, 4)),
    )

    # Array indexing
    assert_identical(raster_da.isel(x=[0, 2, 4]), pandas_da.isel(x=[0, 2, 4]))
    assert_identical(raster_da.isel(y=[1, 3]), pandas_da.isel(y=[1, 3]))


@given(data=st.data())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_outer_array_indexing(data, raster_da, pandas_da):
    """Test that outer array indexing produces identical results for RasterIndex and PandasIndex."""
    sizes = dict(raster_da.sizes)
    indexers = data.draw(outer_array_indexers(sizes=sizes))

    result_raster = raster_da.isel(indexers)
    result_pandas = pandas_da.isel(indexers)

    assert_identical(result_raster, result_pandas)


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
    assert_identical(result_raster, result_pandas)


@given(data=st.data())
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_vectorized_indexing(data, raster_da, pandas_da):
    """Test that vectorized indexing produces identical results for RasterIndex and PandasIndex."""
    sizes = dict(raster_da.sizes)
    indexers = data.draw(vectorized_indexers(sizes=sizes))
    result_raster = raster_da.isel(indexers)
    result_pandas = pandas_da.isel(indexers)
    assert_identical(result_raster, result_pandas)


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
    assert_identical(result_raster, result_pandas)


PERIODIC_GRIDS = st.sampled_from(
    [
        # (origin of left edge, dx, n)
        (-180.0, 1.0, 360),
        (0.0, 1.0, 360),
        (-180.0, 0.5, 720),
        (-181.8, 3.6, 100),
        (180.0, -1.0, 360),
    ]
)


def periodic_da(origin: float, dx: float, n: int) -> xr.DataArray:
    index = RasterIndex.from_transform(
        Affine(dx, 0.0, origin, 0.0, -1.0, 90.0), width=n, height=2, x_period=360.0
    )
    data = np.broadcast_to(np.arange(n, dtype=float), (2, n)).copy()
    return xr.DataArray(data, dims=("y", "x"), coords=xr.Coordinates.from_xindex(index))


def reference_column(label: float, origin: float, dx: float, n: int) -> int:
    """Nearest cell center over several copies of the globe; independent of RasterIndex."""
    centers = origin + dx * (np.arange(n) + 0.5)
    tiled = np.concatenate([centers + k * 360.0 for k in range(-5, 6)])
    return int(np.argmin(np.abs(tiled - label))) % n


@settings(deadline=None)  # first call is slow (cold imports)
@given(grid=PERIODIC_GRIDS, label=st.floats(-1000, 1000))
def test_periodic_sel_matches_reference(grid, label):
    origin, dx, n = grid
    centers = origin + dx * (np.arange(n) + 0.5)
    # skip labels near a cell edge, where "nearest" has a tie
    frac = ((label - centers[0]) / abs(dx)) % 1.0
    assume(abs(frac - 0.5) > 1e-6)
    da = periodic_da(origin, dx, n)
    assert da.sel(x=label, method="nearest").values[0] == reference_column(label, origin, dx, n)


@settings(deadline=None)  # first call is slow (cold imports)
@given(grid=PERIODIC_GRIDS, label=st.floats(-500, 500), k=st.integers(-3, 3))
def test_periodic_sel_invariant_to_period_shift(grid, label, k):
    origin, dx, n = grid
    centers = origin + dx * (np.arange(n) + 0.5)
    frac = ((label - centers[0]) / abs(dx)) % 1.0
    assume(abs(frac - 0.5) > 1e-6)
    da = periodic_da(origin, dx, n)
    a = da.sel(x=label, method="nearest").values[0]
    b = da.sel(x=label + k * 360.0, method="nearest").values[0]
    assert a == b


@settings(deadline=None)  # first call is slow (cold imports)
@given(grid=PERIODIC_GRIDS, start=st.floats(-500, 500), width=st.floats(0.0, 360.0))
def test_periodic_sel_slice_is_contiguous(grid, start, width):
    origin, dx, n = grid
    da = periodic_da(origin, dx, n)
    stop = start + width if dx > 0 else start - width
    out = da.sel(x=slice(start, stop))
    assume(out.sizes["x"] >= 2)
    assert out.sizes["x"] <= n
    assert isinstance(out.xindexes["x"], RasterIndex)
    np.testing.assert_allclose(np.diff(out.x.values), dx)


def _off_cell_edge(v: float, origin: float, step: float, tol: float = 1e-6) -> bool:
    """True if v is more than tol cells away from every cell edge (modulo the period)."""
    frac = ((v - origin) / step) % 1.0
    return min(frac, 1.0 - frac) > tol


@settings(deadline=None, max_examples=100)  # first call is slow (cold imports)
@given(
    n=st.integers(4, 300),
    ny=st.integers(2, 300),
    roll_fraction=st.floats(0.0, 0.99),
    shift=st.sampled_from([-360.0, 0.0, 360.0]),
    base=st.sampled_from([-180.0, 0.0]),
    xs=st.lists(st.floats(-180, 180), min_size=2, max_size=2),
    ys=st.lists(st.floats(-90, 90), min_size=2, max_size=2),
)
# offset near 0 after the unwrap (as_compatible_bboxes tolerance)
@example(n=17, ny=3, roll_fraction=9.5 / 17, shift=-360.0, base=-180.0, xs=[-1.0, 1.0], ys=[-10.0, 10.0])
# selection covers a full period starting left of the extent: any rotation is valid
@example(n=10, ny=3, roll_fraction=1.5 / 10, shift=0.0, base=-180.0, xs=[-179.0, 179.0], ys=[-10.0, 10.0])
def test_periodic_roll_invariance(n, ny, roll_fraction, shift, base, xs, ys):
    """sel+isel+concat of a bbox gives the same values wherever the seam is in the array."""
    dx, dy = 360.0 / n, 180.0 / (ny - 1)  # non-dyadic dy also covers the concat bbox tolerance
    origin, y0 = base - dx / 2, -90 - dy / 2
    k = math.floor(roll_fraction * n)
    values = np.random.default_rng(0).random((ny, n))

    def build(values, x0):
        idx = RasterIndex.from_transform(Affine(dx, 0, x0, 0, dy, y0), width=n, height=ny, x_period=360.0)
        return xr.DataArray(values, dims=("y", "x"), coords=xr.Coordinates.from_xindex(idx))

    origin_b = origin + k * dx + shift
    a = build(values, origin)
    b = build(np.roll(values, -k, axis=1), origin_b)

    west, east = sorted(xs)
    south, north = sorted(ys)
    assume(east - west >= 1e-4 and north - south >= 1e-4)
    # skip bounds within 1e-6 cell of an edge (rounding ties under the moved origin)
    assume(all(_off_cell_edge(v, origin, dx) for v in (west, east)))
    assume(all(_off_cell_edge(v, y0, dy) for v in (south, north)))

    def subset(da):
        sel = da.xindexes["x"].sel({"x": slice(west, east), "y": slice(south, north)})
        xi = np.arange(n)[sel.dim_indexers["x"]]  # indexer may be a slice or an array
        runs = np.split(xi, np.flatnonzero(np.diff(xi) != 1) + 1)
        parts = [da.isel(x=slice(r[0], r[-1] + 1), y=sel.dim_indexers["y"]) for r in runs]
        return parts[0] if len(parts) == 1 else xr.concat(parts, dim="x")  # exercises concat

    # oracle: plain numpy on the unrolled values; cell i covers [origin + i*dx, origin + (i+1)*dx)
    first, last = (math.floor((v - origin) / dx) for v in (west, east))
    cols = (first + np.arange(min(last - first + 1, n))) % n
    rows = np.arange(math.floor((south - y0) / dy), math.floor((north - y0) / dy) + 1)
    expected = values[rows][:, cols]

    def first_col(da, x0, roll):  # unrolled column of the first selected cell
        return (math.floor((float(da.x[0]) - x0) / dx) + roll) % n

    for da, x0, roll in ((subset(a), origin, 0), (subset(b), origin_b, k)):
        assert da.shape == expected.shape
        if last - first + 1 >= n:
            # a full period or more: the rotation is not fixed by the labels, so align the first cell
            np.testing.assert_array_equal(
                da.values, np.roll(expected, first - first_col(da, x0, roll), axis=1)
            )
        else:
            np.testing.assert_array_equal(da.values, expected)  # same order, not only the same multiset
        np.testing.assert_allclose(np.diff(da.x.values), dx)  # concat coords are contiguous
