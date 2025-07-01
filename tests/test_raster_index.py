from textwrap import dedent

import numpy as np
import pyproj
import pytest
import xarray as xr
from affine import Affine
from xarray.testing import assert_identical

from rasterix import RasterIndex, assign_index
from rasterix.utils import get_grid_mapping_var

CRS_ATTRS = pyproj.CRS.from_epsg(4326).to_cf()


def dataset_from_transform(transform: str) -> xr.Dataset:
    return xr.Dataset(
        {"foo": (("y", "x"), np.ones((4, 2)), {"grid_mapping": "spatial_ref"})},
        coords={"spatial_ref": ((), 0, CRS_ATTRS | {"GeoTransform": transform})},
    ).pipe(assign_index)


def test_grid_mapping_var():
    obj = xr.DataArray()
    assert get_grid_mapping_var(obj) is None

    obj = xr.Dataset()
    assert get_grid_mapping_var(obj) is None

    obj = xr.DataArray(attrs={"grid_mapping": "spatial_ref"})
    assert get_grid_mapping_var(obj) is None

    obj = xr.DataArray(attrs={"grid_mapping": "spatial_ref"}, coords={"spatial_ref": 0})
    assert_identical(get_grid_mapping_var(obj), obj["spatial_ref"])

    obj = xr.Dataset({"foo": ((), 0, {"grid_mapping": "spatial_ref"})})
    assert get_grid_mapping_var(obj) is None

    obj = xr.Dataset(
        {
            "foo": ((), 0, {"grid_mapping": "spatial_ref_0"}),
            "zoo": ((), 0, {"grid_mapping": "spatial_ref_1"}),
        },
        coords={"spatial_ref_1": 0},
    )
    assert_identical(get_grid_mapping_var(obj), obj["spatial_ref_1"])


def test_set_xindex() -> None:
    coords = xr.Coordinates(coords={"x": np.arange(0.5, 12.5), "y": np.arange(0.5, 10.5)}, indexes={})
    ds = xr.Dataset(coords=coords)

    with pytest.raises(NotImplementedError, match="Creating a RasterIndex from existing"):
        ds.set_xindex(["x", "y"], RasterIndex)


def test_rectilinear():
    source = "/vsicurl/https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/2024/01_Jan/S_20240101_concentration_v3.0.tif"
    da_no_raster_index = xr.open_dataarray(source, engine="rasterio")
    da_raster_index = assign_index(da_no_raster_index)
    assert da_raster_index.equals(da_no_raster_index)


def test_raster_index_properties():
    index1 = RasterIndex.from_transform(Affine.identity(), width=12, height=10)
    assert index1.xy_shape == (12, 10)
    assert index1.xy_dims == ("x", "y")
    assert index1.xy_coord_names == ("x", "y")
    assert index1.as_geotransform() == "0.0 1.0 0.0 0.0 0.0 1.0"

    index2 = RasterIndex.from_transform(Affine.identity(), width=12, height=10, x_dim="x_", y_dim="y_")
    assert index2.xy_dims == ("x_", "y_")
    assert index2.as_geotransform() == "0.0 1.0 0.0 0.0 0.0 1.0"

    index3 = RasterIndex.from_transform(Affine.rotation(45.0), width=12, height=10)
    assert index3.xy_shape == (12, 10)
    assert index3.xy_dims == ("x", "y")
    assert index3.xy_coord_names == ("xc", "yc")
    assert (
        index3.as_geotransform()
        == "0.0 0.7071067811865476 -0.7071067811865475 0.0 0.7071067811865475 0.7071067811865476"
    )
    assert index3.as_geotransform(decimals=6) == "0.000000 0.707107 -0.707107 0.000000 0.707107 0.707107"


# TODO: parameterize over
# 1. y points up;
# 2. y points down
def test_sel_slice():
    ds = xr.Dataset({"foo": (("y", "x"), np.ones((10, 12)))})
    transform = Affine.identity()
    ds.coords["spatial_ref"] = ((), 0, {"GeoTransform": " ".join(map(str, transform.to_gdal()))})
    ds = assign_index(ds)
    assert "GeoTransform" not in ds.spatial_ref.attrs
    assert ds.xindexes["x"].transform() == transform

    actual = ds.sel(x=slice(4), y=slice(3, 5))
    assert isinstance(actual.xindexes["x"], RasterIndex)
    assert isinstance(actual.xindexes["y"], RasterIndex)

    actual_transform = actual.xindexes["x"].transform()
    assert actual_transform == transform * Affine.translation(0, 3)

    reverse = ds.isel(y=slice(None, None, -1))
    assert_identical(reverse.y, ds.y[::-1])

    reverse = ds.isel(y=slice(8, 5, -1))
    assert_identical(reverse.y, ds.y[8:5:-1])

    reverse = ds.isel(y=slice(8, None, -1))
    assert_identical(reverse.y, ds.y[8::-1])

    reverse = ds.isel(y=slice(None, 5, -1))
    assert_identical(reverse.y, ds.y[:5:-1])


def test_crs() -> None:
    index = RasterIndex.from_transform(Affine.identity(), width=12, height=10)
    assert index.crs is None

    index = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:31370")
    assert index.crs == pyproj.CRS.from_user_input("epsg:31370")


# asserting (in)equality for both "x" and "y" is redundant but not harmful
@pytest.mark.parametrize("index_coord_name", ["x", "y"])
def test_equals(index_coord_name) -> None:
    index = RasterIndex.from_transform(Affine.identity(), width=12, height=10)
    ds = xr.Dataset(coords=xr.Coordinates.from_xindex(index))

    ds2 = ds.isel(x=slice(None), y=slice(None))
    assert ds.xindexes[index_coord_name].equals(ds2.xindexes[index_coord_name])

    # equal x/y coordinate labels but different index types
    ds3 = xr.Dataset(coords={"x": np.arange(0.5, 12.5), "y": np.arange(0.5, 10.5)})
    xr.testing.assert_equal(ds.drop_indexes(["x", "y"]), ds3.drop_indexes(["x", "y"]))
    assert not ds.xindexes[index_coord_name].equals(ds3.xindexes[index_coord_name])

    # same affine transform but different shape
    index4 = RasterIndex.from_transform(Affine.identity(), width=6, height=5)
    ds4 = xr.Dataset(coords=xr.Coordinates.from_xindex(index4))
    assert not ds.xindexes[index_coord_name].equals(ds4.xindexes[index_coord_name])

    # undefined vs. defined CRS
    index5 = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:31370")
    ds5 = xr.Dataset(coords=xr.Coordinates.from_xindex(index5))
    assert ds.xindexes[index_coord_name].equals(ds5.xindexes[index_coord_name])

    # conflicting CRSs
    index6 = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:27700")
    ds6 = xr.Dataset(coords=xr.Coordinates.from_xindex(index6))
    assert not ds5.xindexes[index_coord_name].equals(ds6.xindexes[index_coord_name])


def test_join() -> None:
    index_crs1 = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:31370")
    ds_crs1 = xr.Dataset(coords=xr.Coordinates.from_xindex(index_crs1))

    index_crs2 = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:27700")
    ds_crs2 = xr.Dataset(coords=xr.Coordinates.from_xindex(index_crs2))

    with pytest.raises(ValueError, match="raster indexes.*do not have the same CRS"):
        xr.align(ds_crs1, ds_crs2)


@pytest.mark.parametrize(
    "transforms, concat_dim",
    [
        (
            [
                "-50.0 5 0.0 0.0 0.0 -0.25",
                "-40.0 5 0.0 0.0 0.0 -0.25",
                "-30.0 5 0.0 0.0 0.0 -0.25",
            ],
            "x",
        ),
        (
            [
                # decreasing Δy
                "-40.0 5 0.0 2.0 0.0 -0.5",
                "-40.0 5 0.0 0.0 0.0 -0.5",
                "-40.0 5 0.0 -2.0 0.0 -0.5",
            ],
            "y",
        ),
        (
            [
                # increasing Δy
                "-40.0 5 0.0 -2.0 0.0 0.5",
                "-40.0 5 0.0 0.0 0.0 0.5",
                "-40.0 5 0.0 2.0 0.0 0.5",
            ],
            "y",
        ),
    ],
)
def test_concat_and_combine_nested_1D(transforms, concat_dim):
    """Models two side-by-side tiles"""
    datasets = list(map(dataset_from_transform, transforms))

    if concat_dim == "x":
        new_data = np.ones((4, 2 * len(transforms)))
    else:
        new_data = np.ones((4 * len(transforms), 2))
    expected = xr.Dataset(
        {"foo": (("y", "x"), new_data, {"grid_mapping": "spatial_ref"})},
        coords={"spatial_ref": ((), 0, CRS_ATTRS | {"GeoTransform": transforms[0]})},
    ).pipe(assign_index)

    for actual in [
        xr.combine_nested(datasets, concat_dim=concat_dim, combine_attrs="override"),
        xr.concat(datasets, dim=concat_dim),
    ]:
        assert_identical(actual, expected)
        concat_coord = xr.concat([ds[concat_dim] for ds in datasets], dim=concat_dim)
        assert_identical(actual[concat_dim], concat_coord)


@pytest.mark.parametrize(
    "transforms, concat_dim",
    [
        (
            [
                # out-of-order for Y
                "-40.0 5 0.0 -2.0 0.0 -0.5",
                "-40.0 5 0.0 0.0 0.0 -0.5",
                "-40.0 5 0.0 2.0 0.0 -0.5",
            ],
            "y",
        ),
        (
            [
                # incompatible, different Δx
                "-50.0 2 0.0 0.0 0.0 -0.5",
                "-40.0 5 0.0 0.0 0.0 -0.5",
            ],
            "x",
        ),
        (
            [
                # incompatible, different Δy
                "-50.0 2 0.0 0.0 0.0 -0.5",
                "-40.0 5 0.0 0.0 0.0 -0.25",
            ],
            "x",
        ),
        (
            [
                # exact same transform, makes no sense to concat in X or Y
                "-50.0 5 0.0 0.0 0.0 -0.5",
                "-50.0 5 0.0 0.0 0.0 -0.5",
            ],
            "x",
        ),
    ],
)
def test_concat_errors(transforms, concat_dim):
    datasets = list(map(dataset_from_transform, transforms))
    with pytest.raises(ValueError):
        xr.combine_nested(datasets, concat_dim=concat_dim, combine_attrs="override")
    with pytest.raises(ValueError):
        xr.concat(datasets, dim=concat_dim, combine_attrs="override")


def test_concat_error_alignment():
    transforms, concat_dim = (
        [
            # incompatible, different origins
            "-50.0 5 0.0 -2.0 0.0 -0.5",
            "-40.0 5 0.0 0.0 0.0 -0.5",
        ],
        "x",
    )
    datasets = list(map(dataset_from_transform, transforms))
    with pytest.raises(AssertionError):
        xr.combine_nested(datasets, concat_dim=concat_dim, combine_attrs="override")
    with pytest.raises(AssertionError):
        xr.concat(datasets, dim=concat_dim, combine_attrs="override")


def test_concat_different_shape_compatible_transform_error():
    crs_attrs = pyproj.CRS.from_epsg(4326).to_cf()
    concat_dim = "x"

    ds1 = xr.Dataset(
        {"foo": (("y", "x"), np.ones((4, 3)), {"grid_mapping": "spatial_ref"})},
        coords={"spatial_ref": ((), 0, crs_attrs | {"GeoTransform": "-50.0 5 0.0 0.0 0.0 -0.5"})},
    )
    ds2 = xr.Dataset(
        {"foo": (("y", "x"), np.ones((4, 2)), {"grid_mapping": "spatial_ref"})},
        coords={"spatial_ref": ((), 0, crs_attrs | {"GeoTransform": "-40.0 5 0.0 0.0 0.0 -0.5"})},
    )

    datasets = list(map(assign_index, [ds1, ds2]))
    with pytest.raises(ValueError):
        xr.combine_nested(datasets, concat_dim=concat_dim, combine_attrs="override")
    with pytest.raises(ValueError):
        xr.concat(datasets, dim=concat_dim, combine_attrs="override")


def test_concat_new_dim():
    """models concat along `time` for two tiles with same transform."""
    transforms = [
        "-50.0 0.5 0.0 0.0 0.0 -0.25",
        "-50.0 0.5 0.0 0.0 0.0 -0.25",
    ]
    datasets = list(map(dataset_from_transform, transforms))
    actual = xr.concat(datasets, dim="time", join="exact")
    expected = xr.concat(
        tuple(map(lambda ds: ds.drop_indexes(["x", "y"]), datasets)), dim="time", join="exact"
    ).pipe(assign_index)
    assert_identical(actual, expected)


def test_combine_nested_2d():
    """models 2d tiling"""
    transforms = [
        # row 1
        "-50.0 5 0.0 0.0 0.0 -0.25",
        "-40.0 5 0.0 0.0 0.0 -0.25",
        "-30.0 5 0.0 0.0 0.0 -0.25",
        # row 2
        "-50.0 5 0.0 -1 0.0 -0.25",
        "-40.0 5 0.0 -1 0.0 -0.25",
        "-30.0 5 0.0 -1 0.0 -0.25",
    ]

    datasets = list(map(dataset_from_transform, transforms))
    datasets = [datasets[:3], datasets[3:]]
    actual = xr.combine_nested(datasets, concat_dim=["y", "x"], combine_attrs="identical")
    expected = xr.Dataset(
        {"foo": (("y", "x"), np.ones((8, 6)), {"grid_mapping": "spatial_ref"})},
        coords={"spatial_ref": ((), 0, CRS_ATTRS | {"GeoTransform": transforms[0]})},
    ).pipe(assign_index)
    assert_identical(actual, expected)


@pytest.mark.skip(reason="xarray converts to PandasIndex")
def test_combine_by_coords():
    """models 2d tiling"""
    transforms = [
        # row 1
        "-50.0 5 0.0 0.0 0.0 -0.25",
        "-40.0 5 0.0 0.0 0.0 -0.25",
        "-30.0 5 0.0 0.0 0.0 -0.25",
        # row 2
        "-50.0 5 0.0 -1 0.0 -0.25",
        "-40.0 5 0.0 -1 0.0 -0.25",
        "-30.0 5 0.0 -1 0.0 -0.25",
    ]
    datasets = list(map(dataset_from_transform, transforms))
    xr.combine_by_coords(datasets)


def test_align():
    transforms = [
        "-50.0 5 0.0 0.0 0.0 -0.25",
        "-40.0 5 0.0 0.0 0.0 -0.25",
    ]
    expected_affine = Affine(5, 0, -50, 0, -0.25, 0)
    datasets = list(map(dataset_from_transform, transforms))

    aligned = xr.align(*datasets, join="outer")
    assert all(a.sizes == {"x": 4, "y": 4} for a in aligned)
    assert all(a.xindexes["x"].transform() == expected_affine for a in aligned)

    aligned = xr.align(*datasets, join="inner")
    assert all(a.sizes["x"] == 0 for a in aligned)

    with pytest.raises(xr.AlignmentError):
        aligned = xr.align(*datasets, join="exact")


def test_repr_inline() -> None:
    index1 = RasterIndex.from_transform(Affine.identity(), width=12, height=10)
    ds1 = xr.Dataset(coords=xr.Coordinates.from_xindex(index1))
    actual = ds1.xindexes["x"]._repr_inline_(70)
    expected = "RasterIndex (crs=None)"
    assert actual == expected

    index2 = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:31370")
    ds2 = xr.Dataset(coords=xr.Coordinates.from_xindex(index2))
    actual = ds2.xindexes["x"]._repr_inline_(70)
    expected = "RasterIndex (crs=EPSG:31370)"
    assert actual == expected


def test_repr() -> None:
    index1 = RasterIndex.from_transform(Affine.identity(), width=12, height=10)
    expected = dedent(
        """\
        RasterIndex(crs=None)
            AxisAffineTransformIndex(AxisAffineTransform(a=1, b=0, c=0.5, d=0, e=1, f=0.5, axis=X, dim='x'))
            AxisAffineTransformIndex(AxisAffineTransform(a=1, b=0, c=0.5, d=0, e=1, f=0.5, axis=Y, dim='y'))"""
    )
    actual = repr(index1)
    assert expected == actual

    index2 = RasterIndex.from_transform(Affine.rotation(5), width=12, height=10)
    expected = dedent(
        """\
        RasterIndex(crs=None)
            CoordinateTransformIndex(AffineTransform(a=0.9962, b=-0.08716, c=0.4545, d=0.08716, e=0.9962, f=0.5417))"""
    )
    actual = repr(index2)
    assert expected == actual

    index3 = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:31370")
    assert repr(index3).startswith("RasterIndex(crs=EPSG:31370)")
