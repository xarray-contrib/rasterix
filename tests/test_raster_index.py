import numpy as np
import pyproj
import pytest
import rioxarray  # noqa
import xarray as xr
from affine import Affine
from xarray.testing import assert_identical

from rasterix import RasterIndex, assign_index

CRS_ATTRS = pyproj.CRS.from_epsg(4326).to_cf()


def dataset_from_transform(transform: str) -> xr.Dataset:
    return xr.Dataset(
        {"foo": (("y", "x"), np.ones((4, 2)), {"grid_mapping": "spatial_ref"})},
        coords={"spatial_ref": ((), 0, CRS_ATTRS | {"GeoTransform": transform})},
    ).pipe(assign_index)


def test_rectilinear():
    source = "/vsicurl/https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/2024/01_Jan/S_20240101_concentration_v3.0.tif"
    da_no_raster_index = xr.open_dataarray(source, engine="rasterio")
    da_raster_index = assign_index(da_no_raster_index)
    assert da_raster_index.equals(da_no_raster_index)


# TODO: parameterize over
# 1. y points up;
# 2. y points down
def test_sel_slice():
    ds = xr.Dataset({"foo": (("y", "x"), np.ones((10, 12)))})
    transform = Affine.identity()
    ds = ds.rio.write_transform(transform)
    ds = assign_index(ds)

    assert ds.xindexes["x"].transform() == transform

    actual = ds.sel(x=slice(4), y=slice(3, 5))
    assert isinstance(actual.xindexes["x"], RasterIndex)
    assert isinstance(actual.xindexes["y"], RasterIndex)
    actual_transform = actual.xindexes["x"].transform()

    assert actual_transform == actual.rio.transform()
    assert actual_transform == (transform * Affine.translation(0, 3))


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
                # incompatible, different origins
                "-50.0 5 0.0 -2.0 0.0 -0.5",
                "-40.0 5 0.0 0.0 0.0 -0.5",
            ],
            "x",
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
                # exact same transform, makes no sense to concat
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
    xr.concat(datasets, dim="time", join="exact")


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
    xr.combine_nested(datasets, concat_dim=["y", "x"])
