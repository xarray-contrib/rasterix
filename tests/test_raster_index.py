import xarray as xr

from rasterix import RasterIndex


def test_rectilinear():
    source = "/vsicurl/https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/2024/01_Jan/S_20240101_concentration_v3.0.tif"
    da_no_raster_index = xr.open_dataarray(source, engine="rasterio")
    x_dim = da_no_raster_index.rio.x_dim
    y_dim = da_no_raster_index.rio.y_dim

    index = RasterIndex.from_transform(
        da_no_raster_index.rio.transform(),
        da_no_raster_index.sizes[x_dim],
        da_no_raster_index.sizes[y_dim],
        x_dim=x_dim,
        y_dim=y_dim,
    )
    coords = xr.Coordinates.from_xindex(index)
    da_raster_index = da_no_raster_index.assign_coords(coords)
    assert da_raster_index.equals(da_no_raster_index)
