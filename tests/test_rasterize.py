import dask_geopandas as dgpd
import geodatasets
import geopandas as gpd
import rioxarray  # noqa
import xarray as xr
from xarray.tests import raise_if_dask_computes

from rasterix.rasterize import coverage_ee, rasterize_rio


def test_coverage():
    ds = xr.tutorial.open_dataset("eraint_uvz")
    ds = ds.rio.write_crs("epsg:4326")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    rasterized = coverage_ee(ds, world[["geometry"]], xdim="longitude", ydim="latitude")

    chunked = ds.chunk(latitude=119, longitude=-1)
    with raise_if_dask_computes():
        drasterized = coverage_ee(chunked, world[["geometry"]], xdim="longitude", ydim="latitude")
    xr.testing.assert_allclose(rasterized, drasterized)

    dask_geoms = dgpd.from_geopandas(world, chunksize=5)
    with raise_if_dask_computes():
        drasterized = coverage_ee(chunked, dask_geoms[["geometry"]], xdim="longitude", ydim="latitude")
    xr.testing.assert_allclose(rasterized, drasterized)


def test_rasterize():
    ds = xr.tutorial.open_dataset("eraint_uvz")
    ds = ds.rio.write_crs("epsg:4326")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    rasterized = rasterize_rio(ds, world[["geometry"]], xdim="longitude", ydim="latitude")

    chunked = ds.chunk(latitude=119, longitude=-1)
    with raise_if_dask_computes():
        drasterized = rasterize_rio(chunked, world[["geometry"]], xdim="longitude", ydim="latitude")
    xr.testing.assert_identical(rasterized, drasterized)

    dask_geoms = dgpd.from_geopandas(world, chunksize=5)
    with raise_if_dask_computes():
        drasterized = rasterize_rio(chunked, dask_geoms[["geometry"]], xdim="longitude", ydim="latitude")
    xr.testing.assert_identical(rasterized, drasterized)
