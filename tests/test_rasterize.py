import dask_geopandas as dgpd
import geodatasets
import geopandas as gpd
import pytest
import rioxarray  # noqa
import xarray as xr
from xarray.tests import raise_if_dask_computes

from rasterix.rasterize.rasterio import geometry_mask, rasterize


@pytest.mark.parametrize("clip", [True, False])
def test_rasterize(clip):
    fname = "rasterize_snapshot.nc"
    try:
        snapshot = xr.load_dataarray(fname)
    except FileNotFoundError:
        snapshot = xr.load_dataarray(f"./tests/{fname}")
    if clip:
        snapshot = snapshot.sel(latitude=slice(83.25, None))

    ds = xr.tutorial.open_dataset("eraint_uvz")
    ds = ds.rio.write_crs("epsg:4326")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    kwargs = dict(xdim="longitude", ydim="latitude", clip=clip)
    rasterized = rasterize(ds, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(rasterized, snapshot)

    chunked = ds.chunk(latitude=119, longitude=-1)
    with raise_if_dask_computes():
        drasterized = rasterize(chunked, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(rasterized, drasterized)

    if not clip:
        # clipping not supported with dask geometries
        dask_geoms = dgpd.from_geopandas(world, chunksize=5)
        with raise_if_dask_computes():
            drasterized = rasterize(chunked, dask_geoms[["geometry"]], **kwargs)
        xr.testing.assert_identical(drasterized, snapshot)


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("clip", [False, True])
def test_geometry_mask(clip, invert):
    fname = "geometry_mask_snapshot.nc"
    try:
        snapshot = xr.load_dataarray(fname)
    except FileNotFoundError:
        snapshot = xr.load_dataarray(f"./tests/{fname}")
    if clip:
        snapshot = snapshot.sel(latitude=slice(83.25, None))
    if invert:
        snapshot = ~snapshot

    ds = xr.tutorial.open_dataset("eraint_uvz")
    ds = ds.rio.write_crs("epsg:4326")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    kwargs = dict(xdim="longitude", ydim="latitude", clip=clip, invert=invert)
    rasterized = geometry_mask(ds, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(rasterized, snapshot)

    chunked = ds.chunk(latitude=119, longitude=-1)
    with raise_if_dask_computes():
        drasterized = geometry_mask(chunked, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(drasterized, snapshot)

    if not clip:
        # clipping not supported with dask geometries
        dask_geoms = dgpd.from_geopandas(world, chunksize=5)
        with raise_if_dask_computes():
            drasterized = geometry_mask(chunked, dask_geoms[["geometry"]], **kwargs)
        xr.testing.assert_identical(drasterized, snapshot)
