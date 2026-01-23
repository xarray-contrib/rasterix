import dask_geopandas as dgpd
import geodatasets
import geopandas as gpd
import pytest
import xarray as xr
import xproj  # noqa
from xarray.tests import raise_if_dask_computes

from rasterix.rasterize import geometry_mask, rasterize


@pytest.fixture
def dataset():
    ds = xr.tutorial.open_dataset("eraint_uvz")
    ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
    ds["spatial_ref"].attrs = ds.proj.crs.to_cf()
    return ds


@pytest.mark.parametrize("clip", [False, True])
def test_rasterize(clip, engine, dataset):
    fname = "rasterize_snapshot.nc"
    try:
        snapshot = xr.load_dataarray(fname)
    except FileNotFoundError:
        fname = f"./tests/{fname}"
        snapshot = xr.load_dataarray(fname)
    if clip:
        snapshot = snapshot.sel(latitude=slice(83.25, None))

    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    kwargs = dict(xdim="longitude", ydim="latitude", clip=clip, engine=engine)
    rasterized = rasterize(dataset, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(rasterized, snapshot)

    chunked = dataset.chunk(latitude=119, longitude=-1)
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
def test_geometry_mask(clip, invert, engine, dataset):
    fname = "geometry_mask_snapshot.nc"
    try:
        snapshot = xr.load_dataarray(fname)
    except FileNotFoundError:
        snapshot = xr.load_dataarray(f"./tests/{fname}")
    if clip:
        snapshot = snapshot.sel(latitude=slice(83.25, None))
    if invert:
        snapshot = ~snapshot

    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    kwargs = dict(xdim="longitude", ydim="latitude", clip=clip, invert=invert, engine=engine)
    rasterized = geometry_mask(dataset, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(rasterized, snapshot)

    chunked = dataset.chunk(latitude=119, longitude=-1)
    with raise_if_dask_computes():
        drasterized = geometry_mask(chunked, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(drasterized, snapshot)

    if not clip:
        # clipping not supported with dask geometries
        dask_geoms = dgpd.from_geopandas(world, chunksize=5)
        with raise_if_dask_computes():
            drasterized = geometry_mask(chunked, dask_geoms[["geometry"]], **kwargs)
        xr.testing.assert_identical(drasterized, snapshot)


# geometry_clip is rasterio-specific
def test_geometry_clip(dataset):
    pytest.importorskip("rasterio")

    from rasterix.rasterize.rasterio import geometry_clip

    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    clipped = geometry_clip(dataset, world[["geometry"]], xdim="longitude", ydim="latitude")
    assert clipped is not None
    # Basic check that clipping worked - masked values outside geometries
    assert clipped["u"].isnull().any()
