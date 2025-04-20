import numpy as np
import rioxarray  # noqa
import xarray as xr
from affine import Affine

from rasterix import RasterIndex


def set_raster_index(obj):
    x_dim = obj.rio.x_dim
    y_dim = obj.rio.y_dim

    index = RasterIndex.from_transform(
        obj.rio.transform(), obj.sizes[x_dim], obj.sizes[y_dim], x_dim=x_dim, y_dim=y_dim
    )
    coords = xr.Coordinates.from_xindex(index)
    return obj.assign_coords(coords)


def test_rectilinear():
    source = "/vsicurl/https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/2024/01_Jan/S_20240101_concentration_v3.0.tif"
    da_no_raster_index = xr.open_dataarray(source, engine="rasterio")
    da_raster_index = set_raster_index(da_no_raster_index)
    assert da_raster_index.equals(da_no_raster_index)


# TODO: parameterize over
# 1. y points up;
# 2. y points down
def test_sel_slice():
    ds = xr.Dataset({"foo": (("y", "x"), np.ones((10, 12)))})
    transform = Affine.identity()
    ds = ds.rio.write_transform(transform)
    ds = set_raster_index(ds)

    assert ds.xindexes["x"].transform() == transform

    actual = ds.sel(x=slice(4), y=slice(3, 5))
    assert isinstance(actual.xindexes["x"], RasterIndex)
    assert isinstance(actual.xindexes["y"], RasterIndex)
    actual_transform = actual.xindexes["x"].transform()

    assert actual_transform == actual.rio.transform()
    assert actual_transform == (transform * Affine.translation(0, 3))
