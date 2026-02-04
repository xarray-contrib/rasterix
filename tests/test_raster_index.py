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

    index4 = RasterIndex.from_transform(
        Affine.rotation(45.0), width=12, height=10, x_coord_name="x1", y_coord_name="x2"
    )
    assert index4.xy_coord_names == ("x1", "x2")


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


def test_crs_generated_attributes() -> None:
    index = RasterIndex.from_transform(Affine.identity(), width=12, height=10)
    assert index.crs is None
    variables = index.create_variables()
    assert variables["x"].attrs == {}
    assert variables["y"].attrs == {}

    index = RasterIndex.from_transform(Affine.identity(), width=12, height=10, crs="epsg:31370")
    assert index.crs == pyproj.CRS.from_user_input("epsg:31370")
    variables = index.create_variables()
    assert variables["x"].attrs == {
        "axis": "X",
        "long_name": "Easting",
        "standard_name": "projection_x_coordinate",
        "units": "metre",
    }
    assert variables["y"].attrs == {
        "axis": "Y",
        "long_name": "Northing",
        "standard_name": "projection_y_coordinate",
        "units": "metre",
    }

    index = RasterIndex.from_transform(
        Affine.identity(),
        width=12,
        height=10,
        x_dim="lon",
        y_dim="lat",
        crs="epsg:4326",
    )
    assert index.crs == pyproj.CRS.from_user_input("epsg:4326")
    variables = index.create_variables()
    assert variables["lon"].attrs == {
        "axis": "X",
        "long_name": "longitude coordinate",
        "standard_name": "longitude",
        "units": "degrees_east",
    }
    assert variables["lat"].attrs == {
        "axis": "Y",
        "long_name": "latitude coordinate",
        "standard_name": "latitude",
        "units": "degrees_north",
    }


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


def test_assign_index_with_geotiff_metadata():
    """Test assign_index with GeoTIFF metadata (model_tiepoint and model_pixel_scale)."""
    # Example from issue #55
    # WGS 84 / UTM zone 10N with tiepoint at pixel (0, 0) -> world (323400.0, 4265400.0)
    # and pixel scale of 30.0 x 30.0 meters
    da = xr.DataArray(
        np.ones((100, 100)),
        dims=("y", "x"),
        attrs={
            "model_pixel_scale": [30.0, 30.0, 0.0],
            "model_tiepoint": [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0],
        },
    )

    result = assign_index(da)

    # Check that the index was created
    assert isinstance(result.xindexes["x"], RasterIndex)
    assert isinstance(result.xindexes["y"], RasterIndex)

    # Verify the affine transform
    expected_affine = Affine.translation(323400.0, 4265400.0) * Affine.scale(30.0, 30.0)
    actual_affine = result.xindexes["x"].transform()
    assert actual_affine == expected_affine

    # Check bbox
    bbox = result.xindexes["x"].bbox
    assert bbox.left == 323400.0
    assert bbox.top == 4265400.0 + 100 * 30.0
    assert bbox.right == 323400.0 + 100 * 30.0
    assert bbox.bottom == 4265400.0

    # Verify GeoTIFF metadata attributes are removed
    assert "model_tiepoint" not in result.attrs
    assert "model_pixel_scale" not in result.attrs


def test_assign_index_with_geotiff_metadata_nonzero_tiepoint():
    """Test assign_index with GeoTIFF metadata where tiepoint is not at (0, 0)."""
    # Tiepoint at pixel (10, 20) -> world (500.0, 1000.0)
    da = xr.DataArray(
        np.ones((50, 60)),
        dims=("y", "x"),
        attrs={
            "model_pixel_scale": [10.0, 10.0, 0.0],
            "model_tiepoint": [10.0, 20.0, 0.0, 500.0, 1000.0, 0.0],
        },
    )

    result = assign_index(da)

    # Verify the affine transform
    # c = x - i * scale_x = 500.0 - 10.0 * 10.0 = 400.0
    # f = y - j * scale_y = 1000.0 - 20.0 * 10.0 = 800.0
    expected_affine = Affine.translation(400.0, 800.0) * Affine.scale(10.0, 10.0)
    actual_affine = result.xindexes["x"].transform()
    assert actual_affine == expected_affine


def test_assign_index_with_geotiff_metadata_invalid_z_scale():
    """Test that assign_index raises error when Z pixel scale is non-zero."""
    da = xr.DataArray(
        np.ones((10, 10)),
        dims=("y", "x"),
        attrs={
            "model_pixel_scale": [30.0, 30.0, 10.0],  # Non-zero Z scale
            "model_tiepoint": [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0],
        },
    )

    with pytest.raises(AssertionError, match="Z pixel scale must be 0"):
        assign_index(da)


def test_assign_index_with_stac_proj_transform():
    """Test assign_index with STAC proj:transform attribute."""
    # STAC proj:transform is [a, b, c, d, e, f] representing affine matrix
    # Example: 30m resolution, origin at (323400.0, 4268400.0)
    da = xr.DataArray(
        np.ones((100, 100)),
        dims=("y", "x"),
        attrs={
            "proj:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0],
        },
    )

    result = assign_index(da)

    # Check that the index was created
    assert isinstance(result.xindexes["x"], RasterIndex)
    assert isinstance(result.xindexes["y"], RasterIndex)

    # Verify the affine transform
    expected_affine = Affine(30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0)
    actual_affine = result.xindexes["x"].transform()
    assert actual_affine == expected_affine

    # Verify proj:transform attribute is removed
    assert "proj:transform" not in result.attrs


def test_assign_index_with_stac_proj_transform_9_elements():
    """Test assign_index with STAC proj:transform as full 9-element array."""
    # Full 3x3 matrix in row-major order: [a, b, c, d, e, f, 0, 0, 1]
    da = xr.DataArray(
        np.ones((50, 60)),
        dims=("y", "x"),
        attrs={
            "proj:transform": [10.0, 0.0, 400.0, 0.0, 10.0, 800.0, 0.0, 0.0, 1.0],
        },
    )

    result = assign_index(da)

    # Verify the affine transform (should use first 6 elements)
    expected_affine = Affine(10.0, 0.0, 400.0, 0.0, 10.0, 800.0)
    actual_affine = result.xindexes["x"].transform()
    assert actual_affine == expected_affine


@pytest.mark.parametrize(
    "convention_spec",
    [
        {"name": "spatial:"},  # optional
        {"uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"},  # mandatory
    ],
)
def test_assign_index_with_spatial_zarr_convention(convention_spec: dict[str, str]):
    da = xr.DataArray(
        np.ones((100, 100)),
        dims=("y", "x"),
        attrs={
            "zarr_conventions": [convention_spec],
            "spatial:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0],
        },
    )

    result = assign_index(da)

    # Check that the index was created
    assert isinstance(result.xindexes["x"], RasterIndex)
    assert isinstance(result.xindexes["y"], RasterIndex)

    # Verify the affine transform
    expected_affine = Affine(30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0)
    actual_affine = result.xindexes["x"].transform()
    assert actual_affine == expected_affine

    # Verify spatial:transform attribute is removed
    assert "spatial:transform" not in result.attrs


def test_assign_index_with_spatial_zarr_convention_too_few_raises():
    da = xr.DataArray(
        np.ones((100, 100)),
        dims=("y", "x"),
        attrs={
            "zarr_conventions": [{"name": "spatial:"}],
            "spatial:transform": [30.0, 0.0, 323400.0, 0.0, 30.0],
        },
    )

    with pytest.raises(ValueError, match="spatial:transform must have at least 6 elements"):
        assign_index(da)


def test_assign_index_with_spatial_zarr_convention_transform_type_not_implemented():
    da = xr.DataArray(
        np.ones((100, 100)),
        dims=("y", "x"),
        attrs={
            "zarr_conventions": [{"name": "spatial:"}],
            "spatial:transform_type": "not_affine",
            "spatial:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0],
        },
    )

    with pytest.raises(NotImplementedError, match="Unsupported spatial:transform_type"):
        assign_index(da)


def test_assign_index_with_spatial_zarr_convention_registration_not_implemented():
    da = xr.DataArray(
        np.ones((100, 100)),
        dims=("y", "x"),
        attrs={
            "zarr_conventions": [{"name": "spatial:"}],
            "spatial:registration": "not_pixel",
            "spatial:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0],
        },
    )

    with pytest.raises(NotImplementedError, match="Unsupported spatial:registration"):
        assign_index(da)


def test_assign_index_no_coords_no_metadata():
    """Test that assign_index raises error when coords are missing and no transform metadata."""
    da = xr.DataArray(np.ones((10, 10)), dims=("y", "x"))

    with pytest.raises(ValueError, match="do not have explicit coordinate values"):
        assign_index(da)


def test_assign_index_from_coords():
    """Test assign_index when creating from coordinate arrays."""
    da = xr.DataArray(
        np.ones((10, 12)),
        dims=("y", "x"),
        coords={"x": np.arange(0.5, 12.5), "y": np.arange(0.5, 10.5)},
    )

    result = assign_index(da)

    assert isinstance(result.xindexes["x"], RasterIndex)
    assert isinstance(result.xindexes["y"], RasterIndex)

    # Verify the transform
    # Coordinates are centered at pixels, so we expect identity transform at pixel corners
    # x[0] = 0.5, dx = 1.0 -> c = 0.5 - 1.0/2 = 0.0
    # y[0] = 0.5, dy = 1.0 -> f = 0.5 - 1.0/2 = 0.0 but since y increases down, f = y[-1] - dy/2 = 9.5 - 0.5 = 9.0
    expected_affine = Affine.translation(0.0, 9.0) * Affine.scale(1.0, 1.0)
    actual_affine = result.xindexes["x"].transform()
    assert actual_affine == expected_affine


def test_assign_index_dataset():
    """Test assign_index with a Dataset."""
    ds = xr.Dataset(
        {"foo": (("y", "x"), np.ones((10, 12)))},
        coords={"x": np.arange(0.5, 12.5), "y": np.arange(0.5, 10.5)},
    )

    result = assign_index(ds)

    assert isinstance(result.xindexes["x"], RasterIndex)
    assert isinstance(result.xindexes["y"], RasterIndex)


def test_assign_index_custom_dims():
    """Test assign_index with custom dimension names."""
    da = xr.DataArray(
        np.ones((10, 12)),
        dims=("lat", "lon"),
        coords={"lon": np.arange(0.5, 12.5), "lat": np.arange(0.5, 10.5)},
    )

    result = assign_index(da, x_dim="lon", y_dim="lat")

    assert isinstance(result.xindexes["lon"], RasterIndex)
    assert isinstance(result.xindexes["lat"], RasterIndex)
    assert result.xindexes["lon"].xy_dims == ("lon", "lat")


def test_raster_index_from_tiepoint_and_scale():
    """Test RasterIndex.from_tiepoint_and_scale classmethod."""
    tiepoint = [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0]
    scale = [30.0, 30.0, 0.0]

    index = RasterIndex.from_tiepoint_and_scale(tiepoint=tiepoint, scale=scale, width=100, height=100)

    # Verify the index was created
    assert isinstance(index, RasterIndex)
    assert index.xy_shape == (100, 100)

    # Verify the transform
    expected_affine = Affine.translation(323400.0, 4265400.0) * Affine.scale(30.0, 30.0)
    assert index.transform() == expected_affine


def test_raster_index_from_tiepoint_and_scale_nonzero_tiepoint():
    """Test from_tiepoint_and_scale with tiepoint not at origin."""
    tiepoint = [10.0, 20.0, 0.0, 500.0, 1000.0, 0.0]
    scale = [10.0, 10.0, 0.0]

    index = RasterIndex.from_tiepoint_and_scale(tiepoint=tiepoint, scale=scale, width=60, height=50)

    # Verify the transform
    # c = x - i * scale_x = 500.0 - 10.0 * 10.0 = 400.0
    # f = y - j * scale_y = 1000.0 - 20.0 * 10.0 = 800.0
    expected_affine = Affine.translation(400.0, 800.0) * Affine.scale(10.0, 10.0)
    assert index.transform() == expected_affine


def test_raster_index_from_tiepoint_and_scale_invalid_z():
    """Test from_tiepoint_and_scale raises error for non-zero Z scale."""
    tiepoint = [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0]
    scale = [30.0, 30.0, 10.0]  # Non-zero Z scale

    with pytest.raises(AssertionError, match="Z pixel scale must be 0"):
        RasterIndex.from_tiepoint_and_scale(tiepoint=tiepoint, scale=scale, width=100, height=100)


def test_raster_index_from_stac_proj_metadata():
    """Test RasterIndex.from_stac_proj_metadata classmethod."""
    metadata = {"proj:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0]}

    index = RasterIndex.from_stac_proj_metadata(metadata, width=100, height=100)

    # Verify the index was created
    assert isinstance(index, RasterIndex)
    assert index.xy_shape == (100, 100)

    # Verify the transform
    expected_affine = Affine(30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0)
    assert index.transform() == expected_affine


def test_raster_index_from_stac_proj_metadata_9_elements():
    """Test from_stac_proj_metadata with full 9-element transform."""
    metadata = {"proj:transform": [10.0, 0.0, 400.0, 0.0, 10.0, 800.0, 0.0, 0.0, 1.0]}

    index = RasterIndex.from_stac_proj_metadata(metadata, width=60, height=50)

    # Verify the transform (should use first 6 elements)
    expected_affine = Affine(10.0, 0.0, 400.0, 0.0, 10.0, 800.0)
    assert index.transform() == expected_affine


def test_raster_index_from_stac_proj_metadata_missing_key():
    """Test from_stac_proj_metadata raises error when proj:transform is missing."""
    metadata = {"other_key": "value"}

    with pytest.raises(ValueError, match="metadata must contain 'proj:transform' key"):
        RasterIndex.from_stac_proj_metadata(metadata, width=100, height=100)


def test_raster_index_from_stac_proj_metadata_with_crs():
    """Test from_stac_proj_metadata with CRS parameter."""
    metadata = {"proj:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0]}

    index = RasterIndex.from_stac_proj_metadata(metadata, width=100, height=100, crs="epsg:32610")

    # Verify CRS was set
    assert index.crs is not None
    assert index.crs.to_epsg() == 32610


@pytest.mark.parametrize(
    "convention_spec",
    [
        {"name": "proj:"},  # optional
        {"uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f"},  # mandatory
    ],
)
def test_assign_index_proj_zarr_convention_code(convention_spec: dict[str, str]):
    ds = xr.DataArray(
        np.ones((3, 4)),
        dims=("y", "x"),
        attrs={
            "zarr_conventions": [convention_spec, {"name": "spatial:"}],
            "proj:code": "EPSG:4326",
            "spatial:transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    )
    indexed = assign_index(ds)
    assert indexed.xindexes["x"].crs is not None
    assert indexed.xindexes["x"].crs.to_epsg() == 4326


def test_assign_index_proj_zarr_convention_wkt2():
    crs = pyproj.CRS.from_epsg(3857)
    ds = xr.DataArray(
        np.ones((3, 4)),
        dims=("y", "x"),
        attrs={
            "zarr_conventions": [{"name": "proj:"}, {"name": "spatial:"}],
            "proj:wkt2": crs.to_wkt(),
            "spatial:transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    )
    indexed = assign_index(ds)
    assert indexed.xindexes["x"].crs is not None
    assert indexed.xindexes["x"].crs.to_epsg() == 3857


def test_assign_index_proj_zarr_convention_projjson():
    crs = pyproj.CRS.from_epsg(32610)
    ds = xr.DataArray(
        np.ones((3, 4)),
        dims=("y", "x"),
        attrs={
            "zarr_conventions": [{"name": "proj:"}, {"name": "spatial:"}],
            "proj:projjson": crs.to_json_dict(),
            "spatial:transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    )
    indexed = assign_index(ds)
    assert indexed.xindexes["x"].crs is not None
    assert indexed.xindexes["x"].crs.to_epsg() == 32610
