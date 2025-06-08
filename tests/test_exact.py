from typing import Any

import dask_geopandas as dgpd
import geodatasets
import geopandas as gpd
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import sparse
import xarray as xr
import xarray.testing as xrt
from exactextract import exact_extract
from hypothesis import example, given, settings
from xarray.tests import raise_if_dask_computes

from rasterix.rasterize.exact import CoverageWeights, coverage, xy_to_raster_source

dataset = xr.tutorial.open_dataset("eraint_uvz").rename({"latitude": "y", "longitude": "x"})
dataset = dataset.rio.write_crs("epsg:4326")
world = gpd.read_file(geodatasets.get_path("naturalearth land"))
XSIZE = dataset.x.size
YSIZE = dataset.y.size
GEOMSIZE = len(world)


def np_exact_extract(
    x: np.ndarray,
    y: np.ndarray,
    *,
    geometries: gpd.GeoDataFrame,
    coverage_weight: CoverageWeights,
) -> np.ndarray[Any, Any]:
    """
    Parameters
    ----------

    """
    dtype = np.uint8 if coverage_weight == "none" else np.float64
    raster = xy_to_raster_source(x, y, srs_wkt=geometries.crs.to_wkt())
    result = exact_extract(
        rast=raster,
        vec=geometries,
        ops=["cell_id", f"coverage(coverage_weight={coverage_weight})"],
        output="pandas",
    )
    out = np.zeros((len(geometries), y.size, x.size), dtype=dtype)
    for i in range(len(geometries)):
        res = result.loc[i]
        out[i, ...].flat[res.cell_id] = res.coverage
    as_sparse = sparse.COO.from_numpy(out, fill_value=0)
    return as_sparse


def test_geoseries():
    pass


def test_geodataframe_multiple_columns_error():
    pass


@st.composite
def slice_and_chunksize(draw, size: int) -> tuple[slice, int | None]:
    start = draw(st.integers(min_value=0, max_value=size - 2))
    end = draw(st.integers(min_value=start + 2, max_value=size))
    # stride > 1 does not work with how I construct `expected`
    # stride = draw(st.integers(1, max_value=(end - start) // 2))
    if end - start > 5:
        chunksize = draw(st.none() | st.integers(min_value=(end - start) // 2, max_value=(end - start - 2)))
        if isinstance(chunksize, int):
            while (end - start) % chunksize == 1:
                chunksize += 1
    else:
        chunksize = None
    return slice(start, end), chunksize


@settings(deadline=None)
@example(
    x=(slice(None), None), y=(slice(None), None), coverage_weight="fraction", geom_chunks=None, indexed=True
)
@example(x=(slice(None), -1), y=(slice(120), 111), coverage_weight="fraction", geom_chunks=23, indexed=True)
@given(
    x=slice_and_chunksize(XSIZE),
    y=slice_and_chunksize(YSIZE),
    coverage_weight=st.sampled_from(
        [
            "fraction",
            "area_cartesian",
            "none",
            "area_spherical_m2",
            "area_spherical_km2",
        ]
    ),
    geom_chunks=st.sampled_from([None, -1, 23]),
    indexed=st.booleans(),
)
def test_coverage_weights(
    coverage_weight: CoverageWeights,
    geom_chunks: int | None,
    indexed: bool,
    x: tuple[slice, int | None],
    y: tuple[slice, int | None],
) -> None:
    xslicer, xchunks = x
    yslicer, ychunks = y
    geometries = world.copy(deep=True)
    ds = dataset.copy(deep=True)
    if not indexed:
        ds = ds.drop_indexes(["x", "y"])

    expected = np_exact_extract(
        x=ds.x.data, y=ds.y.data, geometries=geometries, coverage_weight=coverage_weight
    )[:, yslicer, xslicer]

    if geom_chunks:
        geometries = dgpd.from_geopandas(geometries, chunksize=geom_chunks)

    ds = ds.isel(x=xslicer, y=yslicer)
    if xchunks is not None or ychunks is not None:
        ds = ds.chunk({"x": xchunks, "y": ychunks})

    with raise_if_dask_computes():
        actual = coverage(ds, geometries[["geometry"]], coverage_weight=coverage_weight)

    if not indexed:
        assert "x" not in actual.xindexes
        assert "y" not in actual.xindexes
    else:
        assert ds.xindexes["x"].equals(actual.xindexes["x"])
        assert ds.xindexes["y"].equals(actual.xindexes["y"])
    assert expected.dtype == actual.data.dtype

    actual_sparse = actual.compute().data
    npt.assert_equal(expected.todense(), actual_sparse.todense())
    # TODO: This doesn't work for area_spherical_*. A few 0 values sneak through
    if "area_spherical" not in coverage_weight:
        assert expected.nnz == actual_sparse.nnz
    xrt.assert_equal(dataset["spatial_ref"], actual["spatial_ref"])
