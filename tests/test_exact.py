from typing import Any

import geodatasets
import geopandas as gpd
import numpy as np
import numpy.testing as npt
import pytest
import sparse
import xarray as xr
import xarray.testing as xrt
from exactextract import exact_extract
from xarray.tests import raise_if_dask_computes

from rasterix.rasterize.exact import CoverageWeights, coverage, xy_to_raster_source

dataset = xr.tutorial.open_dataset("eraint_uvz").rename({"latitude": "y", "longitude": "x"})
dataset = dataset.rio.write_crs("epsg:4326").isel(y=slice(200, None))
world = gpd.read_file(geodatasets.get_path("naturalearth land"))


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
    shape = (y.size, x.size)
    out = np.zeros((len(geometries), *shape), dtype=dtype)
    for i in range(len(geometries)):
        res = result.loc[i]
        out[i, ...].flat[res.cell_id] = res.coverage
    as_sparse = sparse.COO.from_numpy(out, fill_value=0)
    return as_sparse


def test_geoseries():
    pass


def test_geodataframe_multiple_columns_error():
    pass


@pytest.mark.parametrize("xslicer", [slice(None), slice(40, 90)])
@pytest.mark.parametrize("yslicer", [slice(None)])
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize(
    "coverage_weight", ["area_spherical_m2", "area_cartesian", "area_spherical_km2", "fraction", "none"]
)
def test_coverage_weights(
    coverage_weight: CoverageWeights, chunked: bool, indexed: bool, xslicer, yslicer
) -> None:
    geometries = world.copy(deep=True)
    ds = dataset.copy(deep=True)
    if not indexed:
        ds = ds.drop_indexes(["x", "y"])
    if chunked:
        pass

    expected = np_exact_extract(
        x=ds.x.data, y=ds.y.data, geometries=geometries, coverage_weight=coverage_weight
    )[:, yslicer, xslicer]

    ds = ds.isel(x=xslicer, y=yslicer)
    with raise_if_dask_computes():
        actual = coverage(ds, geometries[["geometry"]], coverage_weight=coverage_weight)

    if not indexed:
        assert "x" not in actual.xindexes
        assert "y" not in actual.xindexes
    else:
        assert ds.xindexes["x"].equals(actual.xindexes["x"])
        assert ds.xindexes["y"].equals(actual.xindexes["y"])
    assert expected.dtype == actual.data.dtype
    npt.assert_equal(expected.todense(), actual.data.todense())
    # TODO: This doesn't work for area_spherical_*. A few 0 values sneak through
    if "area_spherical" not in coverage_weight:
        assert expected.nnz == actual.data.nnz
    xrt.assert_equal(dataset["spatial_ref"], actual["spatial_ref"])
