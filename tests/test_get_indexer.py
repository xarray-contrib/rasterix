import numpy as np
import pytest
import xarray as xr
from affine import Affine
from hypothesis import given, settings
from hypothesis import strategies as st

from rasterix import RasterIndex


def global_index(**kwargs) -> RasterIndex:
    """1-degree global grid, north-up, corners at -180..180 and 90..-90."""
    return RasterIndex.from_transform(Affine(1, 0, -180, 0, -1, 90), width=360, height=180, **kwargs)


def regional_index() -> RasterIndex:
    """20 x 20 cells of 1 degree, corners at -10..10, north-up."""
    return RasterIndex.from_transform(Affine(1, 0, -10, 0, -1, 10), width=20, height=20)


def test_get_indexer_cell_centers_and_interior():
    index = regional_index()
    xc = -10 + np.arange(20) + 0.5
    yc = 10 - np.arange(20) - 0.5
    result = index.get_indexer({"x": xc, "y": yc})
    np.testing.assert_array_equal(result["x"], np.arange(20))
    np.testing.assert_array_equal(result["y"], np.arange(20))
    assert result["x"].dtype == np.int64
    assert result["y"].dtype == np.int64

    result = index.get_indexer({"x": [-9.99, -9.01, 9.99], "y": [9.99, 9.01, -9.99]})
    np.testing.assert_array_equal(result["x"], [0, 0, 19])
    np.testing.assert_array_equal(result["y"], [0, 0, 19])


def test_get_indexer_outside_extent():
    index = regional_index()
    result = index.get_indexer({"x": [-100.0, -10.01, 10.01, 180.5], "y": [100.0, 10.01, -10.01, -100.0]})
    np.testing.assert_array_equal(result["x"], [-1, -1, -1, -1])
    np.testing.assert_array_equal(result["y"], [-1, -1, -1, -1])


def test_get_indexer_edges_are_half_open_in_position():
    # increasing x: left edge included, right edge excluded
    # decreasing y: top edge included, bottom edge excluded
    index = regional_index()
    result = index.get_indexer({"x": [-10.0, -9.0, 10.0], "y": [10.0, 9.0, -10.0]})
    np.testing.assert_array_equal(result["x"], [0, 1, -1])
    np.testing.assert_array_equal(result["y"], [0, 1, -1])


def test_get_indexer_increasing_y():
    index = RasterIndex.from_transform(Affine(1, 0, -10, 0, 1, -10), width=20, height=20)
    result = index.get_indexer({"y": [-10.0, -9.5, -9.0, 9.5, 10.0]})
    np.testing.assert_array_equal(result["y"], [0, 0, 1, 19, -1])


def test_get_indexer_single_axis():
    index = regional_index()
    result = index.get_indexer({"x": [-9.5, 0.5]})
    assert set(result) == {"x"}
    np.testing.assert_array_equal(result["x"], [0, 10])
    result = index.get_indexer({"y": [9.5]})
    assert set(result) == {"y"}
    np.testing.assert_array_equal(result["y"], [0])


def test_get_indexer_keeps_shape():
    index = regional_index()
    x = np.full((2, 3), 0.5)
    result = index.get_indexer({"x": x, "y": 0.5})
    assert result["x"].shape == (2, 3)
    assert result["y"].shape == ()
    assert result["y"] == 9


def test_get_indexer_unknown_coordinate_raises():
    with pytest.raises(KeyError, match="foo"):
        regional_index().get_indexer({"foo": [0.0]})


def test_get_indexer_periodic_wraps():
    index = global_index(x_period=360)
    result = index.get_indexer({"x": [180.5, 539.5, -180.5]})
    np.testing.assert_array_equal(result["x"], [0, 359, 359])

    x = np.array([-179.5, -0.25, 0.25, 179.9])
    expected = index.get_indexer({"x": x})["x"]
    for k in (-2, -1, 1, 2):
        np.testing.assert_array_equal(index.get_indexer({"x": x + k * 360})["x"], expected)


def test_get_indexer_periodic_subset():
    # 100 of 360 cells; a wrapped label can still miss the subset
    index = RasterIndex.from_transform(Affine(1, 0, 100, 0, -1, 90), width=100, height=180, x_period=360)
    result = index.get_indexer({"x": [100.5, 199.5, 200.5, 460.5, -259.5, 99.5]})
    np.testing.assert_array_equal(result["x"], [0, 99, -1, 0, 0, -1])


def test_get_indexer_no_period_no_wrap():
    index = regional_index()
    result = index.get_indexer({"x": [0.5, 0.5 + 360, 0.5 - 360]})
    np.testing.assert_array_equal(result["x"], [10, -1, -1])


@pytest.mark.parametrize("x_period", [None, 360])
def test_get_indexer_non_finite(x_period):
    index = global_index(x_period=x_period)
    result = index.get_indexer({"x": [np.nan, np.inf, -np.inf, 0.5], "y": [np.nan, np.inf, -np.inf, 0.5]})
    np.testing.assert_array_equal(result["x"], [-1, -1, -1, 180])
    np.testing.assert_array_equal(result["y"], [-1, -1, -1, 89])


@pytest.mark.parametrize("x_period", [None, 360])
def test_get_indexer_matches_sel_nearest(x_period):
    index = global_index(x_period=x_period)
    rng = np.random.default_rng(0)
    x = rng.uniform(-179.99, 179.99, 1000)
    y = rng.uniform(-89.99, 89.99, 1000)
    result = index.get_indexer({"x": x, "y": y})
    expected = index.sel({"x": x, "y": y}, method="nearest").dim_indexers
    np.testing.assert_array_equal(result["x"], expected["x"])
    np.testing.assert_array_equal(result["y"], expected["y"])


def rotated_index() -> RasterIndex:
    return RasterIndex.from_transform(
        Affine.translation(5, 5) * Affine.rotation(30) * Affine.scale(2, -2),
        width=10,
        height=8,
        x_coord_name="xc",
        y_coord_name="yc",
    )


def test_get_indexer_rotated():
    index = rotated_index()
    i, j = np.meshgrid(np.arange(10), np.arange(8))
    xc, yc = index.center_transform() * (i.ravel(), j.ravel())
    result = index.get_indexer({"xc": xc, "yc": yc})
    np.testing.assert_array_equal(result["x"], i.ravel())
    np.testing.assert_array_equal(result["y"], j.ravel())

    labels = {"xc": xr.Variable("pt", xc), "yc": xr.Variable("pt", yc)}
    expected = index.sel(labels, method="nearest").dim_indexers
    np.testing.assert_array_equal(result["x"], expected["x"].values)
    np.testing.assert_array_equal(result["y"], expected["y"].values)


def test_get_indexer_rotated_miss_on_both_dims():
    index = rotated_index()
    # corner positions (-0.25, 2.5) miss in x only, (3.5, 8.5) miss in y only
    xc, yc = index.transform() * (np.array([-0.25, 3.5, 3.5]), np.array([2.5, 8.5, np.nan]))
    result = index.get_indexer({"xc": xc, "yc": yc})
    np.testing.assert_array_equal(result["x"], [-1, -1, -1])
    np.testing.assert_array_equal(result["y"], [-1, -1, -1])


def test_get_indexer_rotated_requires_both_coords():
    index = rotated_index()
    with pytest.raises(ValueError, match="both"):
        index.get_indexer({"xc": [5.0]})
    with pytest.raises(ValueError, match="shape"):
        index.get_indexer({"xc": [5.0, 6.0], "yc": [5.0]})


@settings(deadline=None)  # first call is slow (cold imports)
@given(
    dx=st.floats(1e-3, 1e3).flatmap(lambda v: st.sampled_from([v, -v])),
    dy=st.floats(1e-3, 1e3).flatmap(lambda v: st.sampled_from([v, -v])),
    x0=st.floats(-1e6, 1e6),
    y0=st.floats(-1e6, 1e6),
    width=st.integers(1, 1000),
    height=st.integers(1, 1000),
    data=st.data(),
)
def test_get_indexer_cell_center_plus_offset(dx, dy, x0, y0, width, height, data):
    index = RasterIndex.from_transform(Affine(dx, 0, x0, 0, dy, y0), width=width, height=height)
    i = data.draw(st.integers(0, width - 1))
    j = data.draw(st.integers(0, height - 1))
    # stay clear of the cell edges, where round-off can move the point to a neighbour
    fx = data.draw(st.floats(-0.49, 0.49))
    fy = data.draw(st.floats(-0.49, 0.49))
    x, y = index.center_transform() * (i + fx, j + fy)
    result = index.get_indexer({"x": x, "y": y})
    assert result["x"] == i
    assert result["y"] == j
