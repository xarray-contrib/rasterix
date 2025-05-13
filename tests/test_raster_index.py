import numpy as np
import rioxarray  # noqa
import xarray as xr
from affine import Affine
import pytest

from xarray.structure.merge import MergeError

from rasterix import RasterIndex, assign_index
# TODO: hook up xproj to remove need for import?
import xproj

def _open_test_raster():
    source = "/vsicurl/https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/2024/01_Jan/S_20240101_concentration_v3.0.tif"
    return xr.open_dataarray(source, engine="rasterio")


def test_rectilinear():
    da_no_raster_index = _open_test_raster()
    da_raster_index = assign_index(da_no_raster_index)
    assert da_raster_index.equals(da_no_raster_index)


def test_different_crs_same_geotransform():
    da = _open_test_raster()
    da1 = assign_index(da)
    da2 = da1.copy()
    da1 = da1.proj.assign_crs(spatial_ref='EPSG:32610')
    da2 = da2.proj.assign_crs(spatial_ref='EPSG:32611')
    with pytest.raises(
        MergeError, match="conflicting values/indexes on objects to be combined for coordinate"
    ):
        da1 + da2

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
