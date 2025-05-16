import numpy as np
import pyproj
import rioxarray  # noqa
import xarray as xr
from affine import Affine

from rasterix import RasterIndex, assign_index


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


def test_combine_nested():
    transforms = [
        "-50.0 0.5 0.0 0.0 0.0 -0.25",
        "-40.0 0.5 0.0 0.0 0.0 -0.25",
    ]
    crs_attrs = pyproj.CRS.from_epsg(4326).to_cf()

    datasets = [
        xr.Dataset(
            {"foo": (("y", "x"), np.ones((4, 2)), {"grid_mapping": "spatial_ref"})},
            coords={"spatial_ref": ((), 0, crs_attrs | {"GeoTransform": transform})},
        )
        for transform in transforms
    ]
    datasets = list(map(assign_index, datasets))
    xr.combine_nested(datasets, concat_dim="x")
