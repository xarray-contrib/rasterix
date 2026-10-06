import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS

from rasterix import RasterIndex, assign_index
from rasterix.odc_compat import BoundingBox
from rasterix.raster_index import AxisAffineTransform


def axis(*, period=360.0, dx=1.0, origin=-180.0, size=360) -> AxisAffineTransform:
    """x axis transform with cell-center affine, as ``RasterIndex.from_transform`` builds it."""
    affine = Affine(dx, 0.0, origin, 0.0, -1.0, 90.0) * Affine.translation(0.5, 0.5)
    return AxisAffineTransform(affine, size, "x", "x", is_xaxis=True, period=period)


def labels(t: AxisAffineTransform) -> np.ndarray:
    return t.generate_coords()["x"]


def test_period_cells():
    assert axis().period_cells == 360
    assert axis(dx=0.25, size=1440).period_cells == 1440
    assert axis(dx=-1.0, origin=180.0).period_cells == 360
    assert axis(dx=3.6, origin=-181.8, size=100).period_cells == 100
    assert axis(period=None).period_cells is None


def test_period_not_integer_cells_raises():
    with pytest.raises(ValueError, match="not an integer number of cells"):
        axis(dx=0.7, size=10)
    # seam off by 2e-3 cell
    with pytest.raises(ValueError, match="not an integer number of cells"):
        axis(dx=360 / 360.002, size=360)


def test_period_cells_tolerates_geotransform_noise():
    # ctrees AGB GeoTransform dx: 360 / dx = 405000.00000044
    t = axis(dx=0.0008888888888879225, origin=-180.0, size=405000)
    assert t.period_cells == 405000
    # seam off by 1e-4 cell
    assert axis(dx=360 / 360.0001, size=360).period_cells == 360


def test_period_tolerance_options():
    import rasterix

    # seam off by 2e-3 cell: rejected by default, accepted with a looser atol or rtol
    with rasterix.set_options(period_atol=1e-2):
        assert axis(dx=360 / 360.002, size=360).period_cells == 360
    with rasterix.set_options(period_atol=0.0, period_rtol=1e-5):
        assert axis(dx=360 / 360.002, size=360).period_cells == 360
    # seam off by 1e-4 cell: rejected with a stricter atol
    with rasterix.set_options(period_atol=1e-5), pytest.raises(ValueError, match="not an integer"):
        axis(dx=360 / 360.0001, size=360)


def test_slice_keeps_period():
    t = axis().slice(slice(10, 20))
    assert (t.period, t.period_cells, t.size) == (360.0, 360, 10)
    np.testing.assert_allclose(labels(t), np.arange(-169.5, -160.0))


@pytest.mark.parametrize("period", [0.0, -360.0])
def test_period_not_positive_raises(period):
    with pytest.raises(ValueError, match="period must be positive"):
        axis(period=period)


def test_strided_slice_period():
    # 360 % 2 == 0: still periodic on the coarser grid
    t2 = axis().slice(slice(None, None, 2))
    assert (t2.period, t2.period_cells) == (360.0, 180)
    # 360 % 7 != 0: not periodic on the coarser grid; must not raise
    assert axis().slice(slice(None, None, 7)).period is None


def test_run_is_unwrapped():
    t = axis().run(358, 4)
    np.testing.assert_allclose(labels(t), [178.5, 179.5, 180.5, 181.5])
    assert t.period == 360.0
    t = axis().run(1, 3, step=-1)
    np.testing.assert_allclose(labels(t), [-178.5, -179.5, -180.5])


def test_equals_compares_period():
    assert axis().equals(axis())
    assert not axis().equals(axis(period=None))


def test_repr_shows_period():
    assert "period=360" in repr(axis())
    assert "period" not in repr(axis(period=None))


def global_index(*, origin=-180.0, dx=1.0, width=360, height=4, x_period=360.0) -> RasterIndex:
    affine = Affine(dx, 0.0, origin, 0.0, -1.0, 90.0)
    return RasterIndex.from_transform(affine, width=width, height=height, x_period=x_period)


def global_da(**kwargs) -> xr.DataArray:
    """DataArray whose values are the column numbers, so tests can see which columns were selected."""
    index = global_index(**kwargs)
    nx, ny = index.xy_shape
    data = np.broadcast_to(np.arange(nx, dtype=float), (ny, nx)).copy()
    return xr.DataArray(data, dims=("y", "x"), coords=xr.Coordinates.from_xindex(index))


def test_from_transform_sets_period():
    index = global_index()
    assert index._periods() == (360.0, None)
    assert (index.x_period, index.y_period) == (360.0, None)
    assert global_index(x_period=None)._periods() == (None, None)
    assert global_index(x_period=None).x_period is None


def test_from_transform_size_larger_than_period_raises():
    with pytest.raises(ValueError, match="Drop the duplicate seam"):
        global_index(width=361)


def test_from_transform_period_rotated_raises():
    with pytest.raises(NotImplementedError):
        RasterIndex.from_transform(Affine.rotation(45), width=10, height=10, x_period=360.0)


def test_assign_index_passes_period():
    ds = xr.Dataset({"foo": (("y", "x"), np.ones((4, 360)), {"grid_mapping": "spatial_ref"})})
    ds.coords["spatial_ref"] = ((), 0, CRS.from_epsg(4326).to_cf() | {"GeoTransform": "-180 1 0 90 0 -1"})
    out = assign_index(ds, x_period=360.0)
    assert out.xindexes["x"]._periods() == (360.0, None)


def test_proj_set_crs_keeps_period():
    index = global_index()._proj_set_crs("spatial_ref", CRS.from_epsg(4326))
    assert index._periods() == (360.0, None)


def test_new_with_bbox_keeps_period_and_allows_oversize():
    index = global_index()
    out = index._new_with_bbox(BoundingBox(-180.0, 86.0, 182.0, 90.0))
    assert out.xy_shape == (362, 4)
    assert out._periods() == (360.0, None)


def test_align_different_periods_raises():
    a = global_da()
    b = global_da(x_period=None)
    with pytest.raises(ValueError, match="different periods"):
        xr.align(a, b, join="outer")


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right"])
def test_align_seam_subset_in_other_frame_raises(join):
    da = global_da()
    subset = da.sel(x=slice(170, 190))  # 180.5 ... 189.5 are the cells at -179.5 ... -170.5
    with pytest.raises(ValueError, match="same frame"):
        xr.align(da, subset, join=join)


def test_arithmetic_seam_subset_in_other_frame_raises():
    da = global_da()
    with pytest.raises(ValueError, match="same frame"):
        da + da.sel(x=slice(170, 190))


def test_align_full_period_shift_raises():
    a = global_da()
    b = global_da(origin=180.0)  # same cells, labels shifted by +360
    with pytest.raises(ValueError, match="same frame"):
        xr.align(a, b, join="outer")


def test_align_same_frame_ok():
    da = global_da()
    a, _ = xr.align(da.isel(x=slice(0, 10)), da.isel(x=slice(5, 20)), join="outer")
    assert a.sizes["x"] == 20
    # seam subset vs a subset of itself: same frame
    seam = da.sel(x=slice(170, 190))
    a, b = xr.align(seam, seam.isel(x=slice(5, None)), join="inner")
    np.testing.assert_array_equal(a.x.values, b.x.values)


def test_concat_across_seam_still_works():
    da = global_da()
    actual = xr.concat([da.isel(x=slice(-2, None)), da.isel(x=slice(0, 2))], dim="x")
    np.testing.assert_array_equal(actual.x.values, [178.5, 179.5, 180.5, 181.5])


# (origin, dx) for [-180, 180) and [0, 360), ascending and descending
CONVENTIONS = [(-180.0, 1.0), (0.0, 1.0), (180.0, -1.0), (360.0, -1.0)]


@pytest.mark.parametrize("origin, dx", CONVENTIONS)
def test_sel_scalar_wraps(origin, dx):
    da = global_da(origin=origin, dx=dx)
    expected = da.sel(x=239.5 if origin >= 0 and origin != 180.0 else -120.5, method="nearest")
    for label in [-120.5, 239.5, 599.5, -480.5]:
        actual = da.sel(x=label, method="nearest")
        assert actual.values[0] == expected.values[0]


@pytest.mark.parametrize("origin, dx", CONVENTIONS)
def test_sel_array_wraps(origin, dx):
    da = global_da(origin=origin, dx=dx)
    out = da.sel(x=np.array([-120.5, 239.5, 599.5]), method="nearest")
    assert len(set(out.values[0])) == 1


def test_reverse_seam_label_wraps_to_first_column():
    # 180.0 is position 359.5; np.round -> 360 -> outside -> wraps to 0
    pos = axis().reverse({"x": np.array([180.0])})["x"]
    assert np.round(pos).astype(int).tolist() == [0]


def test_reverse_exact_label_wins_on_oversize_index():
    t = axis().run(0, 362)
    pos = np.round(t.reverse({"x": np.array([180.5, 181.5, -179.5, 540.5])})["x"]).astype(int)
    assert pos.tolist() == [360, 361, 0, 0]


def test_reverse_out_of_subset_after_wrap():
    t = axis().run(350, 20)  # labels 170.5 .. 189.5
    pos = np.round(t.reverse({"x": np.array([-175.5, 0.5])})["x"]).astype(int)
    assert pos[0] == 14  # -175.5 == 184.5
    assert not 0 <= pos[1] < t.size  # 0.5 is not in this subset


def test_sel_scalar_out_of_subset_raises():
    da = global_da().isel(x=slice(350, 360))
    with pytest.raises((IndexError, KeyError)):
        da.sel(x=0.5, method="nearest")


def test_non_periodic_reverse_unchanged():
    pos = axis(period=None).reverse({"x": np.array([180.0, 400.0])})["x"]
    np.testing.assert_allclose(pos, [359.5, 579.5])


def as_list(indexer):
    if isinstance(indexer, slice):
        return list(range(indexer.start, indexer.stop))
    return np.asarray(indexer).tolist()


@pytest.mark.parametrize(
    "kwargs, start, stop, expected",
    [
        # inside the extent: same result as without a period
        ({}, -10, 10, list(range(170, 191))),
        # across the seam
        ({}, 170, 190, list(range(350, 360)) + list(range(0, 11))),
        # stop < start means "go across the seam"
        ({}, 175, -175, list(range(355, 360)) + list(range(0, 6))),
        # contains the full extent: whole axis, not empty, not rolled
        ({}, -180, 180, list(range(360))),
        ({}, -200, 200, list(range(360))),
        # wider than one period: cover the axis once, from start
        ({}, 170, 600, list(range(350, 360)) + list(range(0, 350))),
        # [0, 360) convention
        # [0, 360) convention: a start left of a global extent keeps the requested frame
        ({"origin": 0.0}, -10, 10, list(range(-10, 11))),
        # descending: user writes the slice in descending label order
        ({"origin": 180.0, "dx": -1.0}, 10, -10, list(range(170, 191))),
        ({"origin": 180.0, "dx": -1.0}, -170, 170, list(range(350, 360)) + list(range(0, 11))),
    ],
)
def test_periodic_slice_indexer(kwargs, start, stop, expected):
    assert as_list(axis(**kwargs).periodic_slice_indexer(slice(start, stop))) == expected


def test_periodic_slice_indexer_on_subset_keeps_valid_cells():
    t = axis().run(350, 20)  # labels 170.5 .. 189.5
    # from 185 the long way round to 175: only cells of this subset
    # 175 rounds up to the 175.5 cell (position 5), as in the non-periodic path
    assert as_list(t.periodic_slice_indexer(slice(185, 175))) == list(range(15, 20)) + list(range(0, 6))


def test_sel_slice_across_seam_values():
    da = global_da()
    out = da.sel(x=slice(170, 190))
    assert out.values[0].tolist() == list(range(350, 360)) + list(range(0, 11))


def test_sel_slice_inside_unchanged():
    da = global_da()
    ref = global_da(x_period=None)
    xr.testing.assert_equal(da.sel(x=slice(-10, 10)), ref.sel(x=slice(-10, 10)))


def test_sel_slice_open_end_unchanged():
    da = global_da()
    ref = global_da(x_period=None)
    xr.testing.assert_equal(da.sel(x=slice(None, -170)), ref.sel(x=slice(None, -170)))


@pytest.mark.parametrize("wrap", [np.asarray, lambda v: xr.Variable("x", v)])
def test_isel_periodic_run_keeps_index(wrap):
    da = global_da()
    out = da.isel(x=wrap([358, 359, 0, 1]))
    assert isinstance(out.xindexes["x"], RasterIndex)
    np.testing.assert_allclose(out.x.values, [178.5, 179.5, 180.5, 181.5])
    assert out.values[0].tolist() == [358, 359, 0, 1]
    assert out.xindexes["x"]._periods() == (360.0, None)


def test_isel_periodic_run_descending():
    out = global_da().isel(x=[1, 0, 359])
    np.testing.assert_allclose(out.x.values, [-178.5, -179.5, -180.5])


def test_isel_non_run_drops_index():
    out = global_da().isel(x=[0, 5, 9])
    assert "x" not in out.xindexes


def test_isel_run_without_period_drops_index():
    out = global_da(x_period=None).isel(x=[358, 359, 0, 1])
    assert "x" not in out.xindexes


def test_sel_slice_across_seam_keeps_index():
    out = global_da().sel(x=slice(170, 190))
    assert isinstance(out.xindexes["x"], RasterIndex)
    np.testing.assert_allclose(out.x.values, np.arange(170.5, 191.0))
    # the subset still wraps: -175.5 == 184.5
    assert out.sel(x=-175.5, method="nearest").values[0] == 4


def issue_89_dataset(**kwargs) -> xr.Dataset:
    """Reproducer from rasterix#89: global EPSG:4326, 100 x 3.6 deg columns."""
    ds = xr.Dataset({"foo": (("x", "y"), np.ones((100, 100)))})
    ds.coords["spatial_ref"] = ((), 0, CRS.from_epsg(4326).to_cf())
    ds.spatial_ref.attrs["GeoTransform"] = "-181.8 3.6 0.0 90.9 0.0 -1.8"
    return assign_index(ds, x_dim="x", y_dim="y", **kwargs)


def test_concat_tail_head_issue_89():
    ds = issue_89_dataset(x_period=360.0)
    out = xr.concat([ds.isel(x=slice(-2, None)), ds.isel(x=slice(0, 52))], dim="x")
    index = out.xindexes["x"]
    assert out.sizes["x"] == 54
    assert index.transform().c == pytest.approx(171.0)
    assert np.all(np.diff(out.x.values) > 0)
    assert index._periods() == (360.0, None)


def test_concat_tail_head_without_period_still_raises():
    ds = issue_89_dataset()
    with pytest.raises(ValueError, match="X offsets are incompatible"):
        xr.concat([ds.isel(x=slice(-2, None)), ds.isel(x=slice(0, 52))], dim="x")


def test_concat_pad_larger_than_period():
    # xpublish-tiles pads a global grid by wrapping: [0:100] + [0:2]
    ds = issue_89_dataset(x_period=360.0)
    out = xr.concat([ds, ds.isel(x=slice(0, 2))], dim="x")
    index = out.xindexes["x"]
    assert out.sizes["x"] == 102
    assert np.all(np.diff(out.x.values) > 0)
    # exact label wins: 180.0 is the copy at position 100, -180.0 is position 0
    assert index.sel({"x": 180.0}, method="nearest").dim_indexers["x"] == 100
    assert index.sel({"x": -180.0}, method="nearest").dim_indexers["x"] == 0


def test_concat_different_periods_raises():
    a = issue_89_dataset(x_period=360.0).isel(x=slice(0, 50))
    b = issue_89_dataset().isel(x=slice(50, None))
    with pytest.raises(ValueError, match="different periods"):
        xr.concat([a, b], dim="x")


def test_concat_along_y_with_periodic_x_unchanged():
    ds = issue_89_dataset(x_period=360.0)
    out = xr.concat([ds.isel(y=slice(0, 50)), ds.isel(y=slice(50, None))], dim="y")
    assert out.sizes == ds.sizes


def test_concat_tail_head_descending():
    # x decreasing (180 -> -180): the head moves by -360
    da = global_da(origin=180.0, dx=-1.0)
    out = xr.concat([da.isel(x=slice(-2, None)), da.isel(x=slice(0, 2))], dim="x")
    np.testing.assert_allclose(out.x.values, [-178.5, -179.5, -180.5, -181.5])
    assert out.values[0].tolist() == [358, 359, 0, 1]


@pytest.mark.parametrize(
    "start, stop, expected",
    [
        (-161, -150, [-160.5]),
        (-162, -150, [-161.5, -160.5]),
        # 190 .. 200 is -170 .. -160: the full subset
        (190, 200, np.arange(-169.5, -160.0).tolist()),
        (0, 10, []),
    ],
)
def test_sel_slice_on_subset_keeps_index(start, stop, expected):
    sub = global_da().isel(x=slice(10, 20))
    out = sub.sel(x=slice(start, stop))
    assert isinstance(out.xindexes["x"], RasterIndex)
    np.testing.assert_allclose(out.x.values, expected)
    if start < 0 or stop < 10:
        ref = global_da(x_period=None).isel(x=slice(10, 20)).sel(x=slice(start, stop))
        np.testing.assert_allclose(out.x.values, ref.x.values)


def test_sel_slice_half_open_wraps():
    sub = global_da().sel(x=slice(170, 190))  # labels 170.5 .. 190.5
    np.testing.assert_allclose(sub.sel(x=slice(-175, None)).x.values, np.arange(185.5, 191.0))
    # -175 == 185 rounds up to the 185.5 cell (position 15)
    np.testing.assert_allclose(sub.sel(x=slice(None, -175)).x.values, np.arange(170.5, 186.0))


# global grids: even and odd number of cells, ascending and descending
EDGE_GRIDS = [
    {},
    {"dx": 8.0, "width": 45},
    {"origin": 180.0, "dx": -1.0},
    {"origin": 180.0, "dx": -8.0, "width": 45},
]


@pytest.mark.parametrize("kwargs", EDGE_GRIDS)
@pytest.mark.parametrize("label", [-180.0, 180.0, 179.99999999999997, -179.99999999999997])
@pytest.mark.parametrize("half", ["start", "stop"])
def test_sel_slice_half_open_at_extent_edge_unchanged(kwargs, label, half):
    # a label on the extent edges is inside the extent: no wrap, same as without a period
    key = slice(label, None) if half == "start" else slice(None, label)
    actual = global_da(**kwargs).sel(x=key)
    expected = global_da(x_period=None, **kwargs).sel(x=key)
    xr.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("kwargs", EDGE_GRIDS)
@pytest.mark.parametrize("label", [-180.0, 180.0, 179.99999999999997])
def test_sel_scalar_at_seam(kwargs, label):
    # the seam is a tie between the first and the last column
    da = global_da(**kwargs)
    actual = da.sel(x=label, method="nearest")
    assert actual.values[0] in (0, da.sizes["x"] - 1)


def test_sel_slice_left_of_global_extent_keeps_frame():
    out = global_da().sel(x=slice(-190, -170))
    assert isinstance(out.xindexes["x"], RasterIndex)
    np.testing.assert_allclose(out.x.values, np.arange(-189.5, -169.0))
    assert out.values[0].tolist() == list(range(350, 360)) + list(range(0, 11))
    out = global_da(origin=0.0).sel(x=slice(-10, 10))
    np.testing.assert_allclose(out.x.values, np.arange(-9.5, 11.0))


def test_align_outer_keeps_period():
    da = global_da()
    a, b = xr.align(da.isel(x=slice(0, 10)), da.isel(x=slice(5, 15)), join="outer")
    for out in (a, b):
        assert out.xindexes["x"]._periods() == (360.0, None)
        assert out.sizes["x"] == 15


def test_concat_two_periods():
    # a periodic axis may concat to more than one period
    da = global_da()
    out = xr.concat([da, da], dim="x")
    assert out.sizes["x"] == 720
    assert np.all(np.diff(out.x.values) > 0)
    assert out.xindexes["x"]._periods() == (360.0, None)


@pytest.mark.parametrize(
    "idx, x, data",
    [([-2, -1], [-161.5, -160.5], [18, 19]), ([-1, -2], [-160.5, -161.5], [19, 18])],
)
def test_isel_negative_run_on_subset(idx, x, data):
    # non-global subset: negative indices count from the end of the subset
    out = global_da().isel(x=slice(10, 20)).isel(x=idx)
    assert isinstance(out.xindexes["x"], RasterIndex)
    np.testing.assert_allclose(out.x.values, x)
    assert out.values[0].tolist() == data


def test_isel_negative_run_on_global_keeps_frame():
    out = global_da().isel(x=[-2, -1, 0, 1])
    assert isinstance(out.xindexes["x"], RasterIndex)
    np.testing.assert_allclose(out.x.values, [-181.5, -180.5, -179.5, -178.5])
    assert out.values[0].tolist() == [358, 359, 0, 1]
