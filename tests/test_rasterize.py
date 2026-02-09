import dask_geopandas as dgpd
import geodatasets
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import xproj  # noqa
from xarray.tests import raise_if_dask_computes

from rasterix.rasterize import geometry_clip, geometry_mask, rasterize

pytestmark = pytest.mark.filterwarnings(
    "ignore:variable '.*' has non-conforming '_FillValue'"
)


@pytest.fixture
def dataset():
    with xr.tutorial.open_dataset("eraint_uvz") as ds:
        ds = ds.load()
        ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
        ds["spatial_ref"].attrs = ds.proj.crs.to_cf()
        return ds


@pytest.mark.parametrize("clip", [False, True])
def test_rasterize(clip, engine, dataset):
    # Use engine-specific snapshots due to pixel boundary differences:
    # - rasterio: default (center-point) rasterization
    # - rusterize: has its own boundary handling
    # - exactextract: equivalent to all_touched=True (any coverage counts)
    if engine == "rusterize":
        suffix = "_rusterize"
    elif engine == "exactextract":
        suffix = "_all_touched"  # exactextract matches rasterio all_touched=True
    else:
        suffix = ""
    fname = f"rasterize_snapshot{suffix}.nc"
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
    assert drasterized.chunks is not None, "Output should be chunked when input is dask"
    if not clip:
        # When not clipping, chunks should match input exactly
        expected_chunks = (
            chunked.chunksizes["latitude"],
            chunked.chunksizes["longitude"],
        )
        assert drasterized.chunks == expected_chunks
    xr.testing.assert_identical(rasterized, drasterized)

    if not clip:
        # clipping not supported with dask geometries
        dask_geoms = dgpd.from_geopandas(world, chunksize=5)
        with raise_if_dask_computes():
            drasterized = rasterize(chunked, dask_geoms[["geometry"]], **kwargs)
        assert (
            drasterized.chunks is not None
        ), "Output should be chunked when input is dask"
        xr.testing.assert_identical(drasterized, snapshot)


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("clip", [False, True])
def test_geometry_mask(clip, invert, engine, dataset):
    # Use engine-specific snapshots due to pixel boundary differences:
    # - rasterio: default (center-point) rasterization
    # - rusterize: has its own boundary handling
    # - exactextract: equivalent to all_touched=True (any coverage counts)
    if engine == "rusterize":
        suffix = "_rusterize"
    elif engine == "exactextract":
        suffix = "_all_touched"  # exactextract matches rasterio all_touched=True
    else:
        suffix = ""
    fname = f"geometry_mask_snapshot{suffix}.nc"
    try:
        snapshot = xr.load_dataarray(fname)
    except FileNotFoundError:
        snapshot = xr.load_dataarray(f"./tests/{fname}")
    if clip:
        snapshot = snapshot.sel(latitude=slice(83.25, None))
    if invert:
        snapshot = ~snapshot

    world = gpd.read_file(geodatasets.get_path("naturalearth land"))

    kwargs = dict(
        xdim="longitude", ydim="latitude", clip=clip, invert=invert, engine=engine
    )
    rasterized = geometry_mask(dataset, world[["geometry"]], **kwargs)
    xr.testing.assert_identical(rasterized, snapshot)

    chunked = dataset.chunk(latitude=119, longitude=-1)
    with raise_if_dask_computes():
        drasterized = geometry_mask(chunked, world[["geometry"]], **kwargs)
    assert drasterized.chunks is not None, "Output should be chunked when input is dask"
    if not clip:
        # When not clipping, chunks should match input exactly
        expected_chunks = (
            chunked.chunksizes["latitude"],
            chunked.chunksizes["longitude"],
        )
        assert drasterized.chunks == expected_chunks
    xr.testing.assert_identical(drasterized, snapshot)

    if not clip:
        # clipping not supported with dask geometries
        dask_geoms = dgpd.from_geopandas(world, chunksize=5)
        with raise_if_dask_computes():
            drasterized = geometry_mask(chunked, dask_geoms[["geometry"]], **kwargs)
        assert (
            drasterized.chunks is not None
        ), "Output should be chunked when input is dask"
        xr.testing.assert_identical(drasterized, snapshot)


def test_geometry_clip(engine, dataset):
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    clipped = geometry_clip(
        dataset, world[["geometry"]], xdim="longitude", ydim="latitude", engine=engine
    )
    assert clipped is not None
    # Basic check that clipping worked - masked values outside geometries
    assert clipped["u"].isnull().any()


def test_rasterize_field(engine, dataset):
    """Test burning feature field values into the raster."""
    if engine != "rasterio":
        pytest.skip("field burning only supported with rasterio engine")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    world = world[["geometry"]].copy()
    world["pop"] = np.arange(len(world), dtype=np.float64) * 1.5

    kwargs = dict(xdim="longitude", ydim="latitude", engine=engine, field="pop")
    rasterized = rasterize(dataset, world, **kwargs)

    # dtype should match the field values
    assert rasterized.dtype == np.float64
    # burned values should include actual field values, not just integer indices
    unique_burned = np.unique(rasterized.values)
    # 0.0 is fill, other values should be multiples of 1.5
    non_fill = unique_burned[unique_burned != 0.0]
    assert len(non_fill) > 0
    for v in non_fill:
        assert v % 1.5 == 0.0, f"Expected multiples of 1.5, got {v}"

    # dask path
    chunked = dataset.chunk(latitude=119, longitude=-1)
    drasterized = rasterize(chunked, world, **kwargs)
    assert drasterized.chunks is not None
    xr.testing.assert_identical(rasterized, drasterized)


def test_rasterize_field_int(engine, dataset):
    """Test burning integer field values."""
    if engine != "rasterio":
        pytest.skip("field burning only supported with rasterio engine")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    world = world[["geometry"]].copy()
    world["code"] = np.arange(len(world), dtype=np.int32) * 10

    rasterized = rasterize(
        dataset, world, xdim="longitude", ydim="latitude", engine=engine, field="code"
    )
    assert np.issubdtype(rasterized.dtype, np.integer)

    unique_burned = np.unique(rasterized.values)
    non_fill = unique_burned[unique_burned != 0]
    assert len(non_fill) > 0
    for v in non_fill:
        assert v % 10 == 0, f"Expected multiples of 10, got {v}"


def test_rasterize_field_missing_column(engine, dataset):
    """Test that a missing field column raises ValueError."""
    if engine != "rasterio":
        pytest.skip("field burning only supported with rasterio engine")
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        rasterize(
            dataset,
            world[["geometry"]],
            xdim="longitude",
            ydim="latitude",
            engine=engine,
            field="nonexistent",
        )
