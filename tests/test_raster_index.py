from textwrap import dedent

import numpy as np
import pandas as pd
import pyproj
import pytest
import rioxarray  # noqa
import xarray as xr
from affine import Affine

from rasterix import RasterIndex, assign_index


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


def test_crs() -> None:
    index = RasterIndex.from_transform(Affine.identity(), 12, 10)
    assert index.crs is None

    index = RasterIndex.from_transform(Affine.identity(), 12, 10, crs="epsg:31370")
    assert index.crs == pyproj.CRS.from_user_input("epsg:31370")


# asserting (in)equality for both "x" and "y" is redundant but not harmful
@pytest.mark.parametrize("index_coord_name", ["x", "y"])
def test_equals(index_coord_name) -> None:
    index = RasterIndex.from_transform(Affine.identity(), 12, 10)
    ds = xr.Dataset(coords=xr.Coordinates.from_xindex(index))

    ds2 = ds.isel(x=slice(None), y=slice(None))
    assert ds.xindexes[index_coord_name].equals(ds2.xindexes[index_coord_name])

    # equal x/y coordinate labels but different index types
    ds3 = xr.Dataset(coords={"x": np.arange(0.5, 12.5), "y": np.arange(0.5, 10.5)})
    xr.testing.assert_equal(ds.drop_indexes(["x", "y"]), ds3.drop_indexes(["x", "y"]))
    assert not ds.xindexes[index_coord_name].equals(ds3.xindexes[index_coord_name])

    # same affine transform but different shape
    index4 = RasterIndex.from_transform(Affine.identity(), 6, 5)
    ds4 = xr.Dataset(coords=xr.Coordinates.from_xindex(index4))
    assert not ds.xindexes[index_coord_name].equals(ds4.xindexes[index_coord_name])

    # undefined vs. defined CRS
    index5 = RasterIndex.from_transform(Affine.identity(), 12, 10, crs="epsg:31370")
    ds5 = xr.Dataset(coords=xr.Coordinates.from_xindex(index5))
    assert ds.xindexes[index_coord_name].equals(ds5.xindexes[index_coord_name])

    # conflicting CRSs
    index6 = RasterIndex.from_transform(Affine.identity(), 12, 10, crs="epsg:27700")
    ds6 = xr.Dataset(coords=xr.Coordinates.from_xindex(index6))
    assert not ds5.xindexes[index_coord_name].equals(ds6.xindexes[index_coord_name])

    # different wrapped indexes
    ds7 = ds.isel(x=[0, 3])
    assert not ds.xindexes[index_coord_name].equals(ds7.xindexes[index_coord_name])


def test_join() -> None:
    index_crs1 = RasterIndex.from_transform(Affine.identity(), 12, 10, crs="epsg:31370")
    ds_crs1 = xr.Dataset(coords=xr.Coordinates.from_xindex(index_crs1))

    index_crs2 = RasterIndex.from_transform(Affine.identity(), 12, 10, crs="epsg:27700")
    ds_crs2 = xr.Dataset(coords=xr.Coordinates.from_xindex(index_crs2))

    with pytest.raises(ValueError, match="raster indexes.*do not have the same CRS"):
        xr.align(ds_crs1, ds_crs2)


def test_to_pandas_index() -> None:
    index = RasterIndex.from_transform(Affine.identity(), 12, 10)
    ds = xr.Dataset(coords=xr.Coordinates.from_xindex(index))

    with pytest.raises(ValueError, match="Cannot convert RasterIndex to pandas.Index"):
        ds.indexes["x"]

    ds2 = ds.isel(y=0)
    assert ds2.indexes["x"].equals(pd.Index(np.arange(0.5, 12.5)))

    ds3 = ds.isel(x=0, y=[0, 3])
    assert ds3.indexes["y"].equals(pd.Index([0.5, 3.5]))


def test_repr_inline() -> None:
    index1 = RasterIndex.from_transform(Affine.identity(), 12, 10)
    ds1 = xr.Dataset(coords=xr.Coordinates.from_xindex(index1))
    actual = ds1.xindexes["x"]._repr_inline_(70)
    expected = "RasterIndex (crs=None)"
    assert actual == expected

    index2 = RasterIndex.from_transform(Affine.identity(), 12, 10, crs="epsg:31370")
    ds2 = xr.Dataset(coords=xr.Coordinates.from_xindex(index2))
    actual = ds2.xindexes["x"]._repr_inline_(70)
    expected = "RasterIndex (crs=EPSG:31370)"
    assert actual == expected


def test_repr() -> None:
    index1 = RasterIndex.from_transform(Affine.identity(), 12, 10)
    ds1 = xr.Dataset(coords=xr.Coordinates.from_xindex(index1))
    expected = dedent(
        """\
        RasterIndex(crs=None)
        'x':
            AxisAffineTransformIndex(AxisAffineTransform(a=1, b=0, c=0.5, d=0, e=1, f=0.5, axis=X, dim='x'))
        'y':
            AxisAffineTransformIndex(AxisAffineTransform(a=1, b=0, c=0.5, d=0, e=1, f=0.5, axis=Y, dim='y'))"""
    )
    actual = repr(index1)
    assert expected == actual

    ds2 = ds1.isel(x=0, y=[1, 3])
    index2 = ds2.xindexes["x"]
    expected = dedent(
        """\
        RasterIndex(crs=None)
        'y':
            PandasIndex(Index([1.5, 3.5], dtype='float64', name='y'))"""
    )
    actual = repr(index2)
    assert expected == actual

    index3 = RasterIndex.from_transform(Affine.rotation(5), 12, 10)
    expected = dedent(
        """\
        RasterIndex(crs=None)
        ('x', 'y'):
            CoordinateTransformIndex(AffineTransform(a=0.9962, b=-0.08716, c=0.4545, d=0.08716, e=0.9962, f=0.5417))"""
    )
    actual = repr(index3)
    assert expected == actual

    index4 = RasterIndex.from_transform(Affine.identity(), 12, 10, crs="epsg:31370")
    assert repr(index4).startswith("RasterIndex(crs=EPSG:31370)")
