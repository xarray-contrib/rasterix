# exactexact wrappers
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import sparse
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from .utils import geometries_as_dask_array, is_in_memory

if TYPE_CHECKING:
    import dask.array
    import dask_geopandas

MIN_CHUNK_SIZE = 2  # exactextract cannot handle arrays of size 1.
GEOM_AXIS = 0
X_AXIS = 1
Y_AXIS = 2

CoverageWeights = Literal["area_spherical_m2", "area_cartesian_m2", "area_spherical_km2", "fraction", "none"]

__all__ = [
    "coverage",
]


def get_dtype(coverage_weight: CoverageWeights, geometries):
    if coverage_weight.lower() == "none":
        dtype = np.min_scalar_type(len(geometries))
    else:
        dtype = np.float64
    return dtype


def np_coverage(
    x: np.ndarray,
    y: np.ndarray,
    *,
    geometries: gpd.GeoDataFrame,
    coverage_weight: CoverageWeights = "fraction",
) -> np.ndarray[Any, Any]:
    """
    Parameters
    ----------

    """
    assert x.ndim == 1
    assert y.ndim == 1

    dtype = get_dtype(coverage_weight, geometries)

    if len(geometries.columns) > 1:
        raise ValueError("Require a single geometries column or a GeoSeries.")

    xsize = x.size
    ysize = y.size

    # we need the top left corner, and the bottom right corner
    dx0 = (x[1] - x[0]) / 2
    dx1 = (x[-1] - x[-2]) / 2
    dy0 = np.abs(y[1] - y[0]) / 2
    dy1 = np.abs(y[-1] - y[-2]) / 2
    if y[0] > y[-1]:
        dy0, dy1 = dy1, dy0

    shape = (ysize, xsize)
    raster = NumPyRasterSource(
        np.broadcast_to([1], shape),
        xmin=x.min() - dx0,
        xmax=x.max() + dx1,
        ymin=y.min() - dy0,
        ymax=y.max() + dy1,
        srs_wkt=geometries.crs.to_wkt(),
    )
    result = exact_extract(
        rast=raster,
        vec=geometries,
        ops=["cell_id", f"coverage(coverage_weight={coverage_weight})"],
        output="pandas",
        # max_cells_in_memory=2*x.size * y.size
    )

    lens = np.vectorize(len)(result.cell_id.values)
    nnz = np.sum(lens)

    # Notes on GCXS vs COO,  For N data points in 263 geoms by 4000 x by 4000 y
    # 1. GCXS cannot compress _all_ axes. This is relevant here.
    # 2. GCXS: indptr is 4000*4000 + 1, N per indices & N per data
    # 3. COO: 4*N
    # It is not obvious that there is much improvement to GCXS at least currently
    geom_idxs = np.empty((nnz,), dtype=np.int64)
    xy_idxs = np.empty((nnz,), dtype=np.int64)
    data = np.empty((nnz,), dtype=dtype)

    off = 0
    for i in range(len(geometries)):
        cell_id = result.cell_id.values[i]
        if cell_id.size == 0:
            continue
        geom_idxs[off : off + cell_id.size] = i
        xy_idxs[off : off + cell_id.size] = cell_id
        data[off : off + cell_id.size] = result.coverage.values[i]
        off += cell_id.size
    return sparse.COO(
        (geom_idxs, *np.unravel_index(xy_idxs, shape=shape)),
        data=data,
        sorted=True,
        fill_value=0,
        shape=(len(geometries), *shape),
    )


def coverage_np_dask_wrapper(
    geom_array: np.ndarray, x: np.ndarray, y: np.ndarray, coverage_weight: CoverageWeights, crs
) -> np.ndarray:
    return np_coverage(
        x=x.squeeze(axis=(GEOM_AXIS, Y_AXIS)),
        y=y.squeeze(axis=(GEOM_AXIS, X_AXIS)),
        geometries=gpd.GeoDataFrame(geometry=geom_array.squeeze(axis=(X_AXIS, Y_AXIS)), crs=crs),
        coverage_weight=coverage_weight,
    )


def dask_coverage(
    x: dask.array.Array,
    y: dask.array.Array,
    *,
    geom_array: dask.array.Array,
    coverage_weight: CoverageWeights = "fraction",
    crs: Any,
) -> dask.array.Array:
    import dask.array

    if any(c == 1 for c in x.chunks) or any(c == 1 for c in y.chunks):
        raise ValueError("exactextract does not support a chunksize of 1. Please rechunk to avoid this")

    return dask.array.map_blocks(
        coverage_np_dask_wrapper,
        geom_array[:, np.newaxis, np.newaxis],
        x[np.newaxis, :, np.newaxis],
        y[np.newaxis, np.newaxis, :],
        crs=crs,
        coverage_weight=coverage_weight,
        meta=sparse.COO(
            [], data=np.array([], dtype=get_dtype(coverage_weight, geom_array)), shape=(0, 0, 0), fill_value=0
        ),
    )


def coverage(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    xdim="x",
    ydim="y",
    coverage_weight: CoverageWeights = "fraction",
) -> xr.DataArray:
    """
    Returns "coverage" fractions for each pixel for each geometry calculated using exactextract.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        Xarray object used to extract the grid
    geometries: GeoDataFrame | DaskGeoDataFrame
        Geometries used for to calculate coverage
    xdim: str
        Name of the "x" dimension on ``obj``.
    ydim: str
        Name of the "y" dimension on ``obj``.
    coverage_weight: {"fraction", "none", "area_cartesian", "area_spherical_m2", "area_spherical_km2"}
        Weights to estimate, passed directly to exactextract.

    Returns
    -------
    DataArray
        3D dataarray with coverage fraction. The additional dimension is "geometry".
    """
    if "spatial_ref" not in obj.coords:
        raise ValueError("Xarray object must contain the `spatial_ref` variable.")
    # FIXME: assert obj.crs == geometries.crs
    if is_in_memory(obj=obj, geometries=geometries):
        out = np_coverage(
            x=obj[xdim].data,
            y=obj[ydim].data,
            geometries=geometries,
            coverage_weight=coverage_weight,
        )
        geom_array = geometries.to_numpy().squeeze(axis=1)
    else:
        from dask.array import from_array

        geom_dask_array = geometries_as_dask_array(geometries)
        out = dask_coverage(
            x=from_array(obj[xdim].data, chunks=obj.chunksizes.get(xdim, -1)),
            y=from_array(obj[ydim].data, chunks=obj.chunksizes.get(ydim, -1)),
            geom_array=geom_dask_array,
            crs=geometries.crs,
            coverage_weight=coverage_weight,
        )
        if isinstance(geometries, gpd.GeoDataFrame):
            geom_array = geometries.to_numpy().squeeze(axis=1)
        else:
            geom_array = geom_dask_array

    name = "coverage"
    attrs = {}
    if "area" in coverage_weight:
        name = "area"
    if "_m2" in coverage_weight:
        attrs["long_name"] = coverage_weight.removesuffix("_m2")
        attrs["units"] = "m2"
    elif "_km2" in coverage_weight:
        attrs["long_name"] = coverage_weight.removesuffix("_km2")
        attrs["units"] = "km2"

    indexes = {dim: obj.xindexes.get(dim) for dim in (xdim, ydim) if obj.xindexes.get(dim) is not None}
    coverage = xr.DataArray(
        dims=("geometry", ydim, xdim),
        data=out,
        coords=xr.Coordinates(
            coords={
                "spatial_ref": obj.spatial_ref,
                "geometry": geom_array,
            },
            indexes=indexes,
        ),
        attrs=attrs,
        name=name,
    )
    return coverage
