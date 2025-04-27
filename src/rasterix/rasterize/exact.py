# exactexact wrappers
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from .utils import geometries_as_dask_array, is_in_memory

if TYPE_CHECKING:
    import dask.array
    import dask_geopandas

MIN_CHUNK_SIZE = 2  # exactextract cannot handle arrays of size 1.

__all__ = [
    "coverage",
]


def get_dtype(coverage_weight, geometries):
    if coverage_weight.lower() == "fraction":
        dtype = "float64"
    elif coverage_weight.lower() == "none":
        dtype = np.min_scalar_type(len(geometries))
    else:
        raise NotImplementedError
    return dtype


def np_coverage(
    x: np.ndarray,
    y: np.ndarray,
    *,
    geometries: gpd.GeoDataFrame,
    coverage_weight: Literal["fraction", "none"] = "fraction",
) -> np.ndarray[Any, Any]:
    """
    Parameters
    ----------

    """
    assert x.ndim == 1
    assert y.ndim == 1

    dtype = get_dtype(coverage_weight, geometries)

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
    out = np.zeros((len(geometries), *shape), dtype=dtype)
    # TODO: vectorized assignment?
    for i in range(len(geometries)):
        res = result.loc[i]
        # indices = np.unravel_index(res.cell_id, shape=shape)
        # out[(i, *indices)] = offset + i + 1 # 0 is the fill value
        out[i, ...].flat[res.cell_id] = res.coverage
    return out


def coverage_np_dask_wrapper(
    x: np.ndarray, y: np.ndarray, geom_array: np.ndarray, coverage_weight, crs
) -> np.ndarray:
    return np_coverage(
        x=x,
        y=y,
        geometries=gpd.GeoDataFrame(geometry=geom_array, crs=crs),
        coverage_weight=coverage_weight,
    )


def dask_coverage(
    x: dask.array.Array,
    y: dask.array.Array,
    *,
    geom_array: dask.array.Array,
    coverage_weight: Literal["fraction", "none"] = "fraction",
    crs: Any,
) -> dask.array.Array:
    import dask.array

    if any(c == 1 for c in x.chunks) or any(c == 1 for c in y.chunks):
        raise ValueError("exactextract does not support a chunksize of 1. Please rechunk to avoid this")

    return dask.array.blockwise(
        coverage_np_dask_wrapper,
        "gji",
        x,
        "i",
        y,
        "j",
        geom_array,
        "g",
        crs=crs,
        coverage_weight=coverage_weight,
        dtype=get_dtype(coverage_weight, geom_array),
    )


def coverage(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    xdim="x",
    ydim="y",
    coverage_weight="fraction",
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

    coverage = xr.DataArray(
        dims=("geometry", ydim, xdim),
        data=out,
        coords=xr.Coordinates(
            coords={
                xdim: obj.coords[xdim],
                ydim: obj.coords[ydim],
                "spatial_ref": obj.spatial_ref,
                "geometry": geom_array,
            },
            indexes={xdim: obj.xindexes[xdim], ydim: obj.xindexes[ydim]},
        ),
    )
    return coverage
