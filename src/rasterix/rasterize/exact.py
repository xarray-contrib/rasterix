# exactexact wrappers
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import sparse
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from .utils import clip_to_bbox, geometries_as_dask_array, is_in_memory

if TYPE_CHECKING:
    import dask.array
    import dask_geopandas

MIN_CHUNK_SIZE = 2  # exactextract cannot handle arrays of size 1.
GEOM_AXIS = 0
Y_AXIS = 1
X_AXIS = 2

DEFAULT_STRATEGY = "feature-sequential"
Strategy = Literal["feature-sequential", "raster-sequential", "raster-parallel"]
CoverageWeights = Literal["area_spherical_m2", "area_cartesian", "area_spherical_km2", "fraction", "none"]

__all__ = ["coverage"]


def affine_to_xy_coords(affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Convert affine transform and shape to x and y coordinate arrays.

    Parameters
    ----------
    affine : Affine
        Affine transform for the raster grid.
    shape : tuple[int, int]
        Shape of the raster (nrows, ncols).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x and y coordinate arrays (1D) representing pixel centers.
    """
    nrows, ncols = shape
    # Generate pixel center coordinates
    cols = np.arange(ncols)
    rows = np.arange(nrows)
    # affine.c is x origin, affine.a is x pixel size
    # affine.f is y origin, affine.e is y pixel size (typically negative)
    x = affine.c + (cols + 0.5) * affine.a
    y = affine.f + (rows + 0.5) * affine.e
    return x, y


def xy_to_raster_source(x: np.ndarray, y: np.ndarray, *, srs_wkt: str | None) -> NumPyRasterSource:
    assert x.ndim == 1
    assert y.ndim == 1

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
        srs_wkt=srs_wkt,
    )

    return raster


def get_dtype(coverage_weight: CoverageWeights, geometries):
    if coverage_weight.lower() == "none":
        dtype = np.uint8
    else:
        dtype = np.float64
    return dtype


def np_coverage(
    x: np.ndarray,
    y: np.ndarray,
    *,
    geometries: gpd.GeoDataFrame,
    strategy: Strategy = DEFAULT_STRATEGY,
    coverage_weight: CoverageWeights = "fraction",
) -> np.ndarray[Any, Any]:
    dtype = get_dtype(coverage_weight, geometries)

    if len(geometries.columns) > 1:
        raise ValueError("Require a single geometries column or a GeoSeries.")

    shape = (y.size, x.size)
    raster = xy_to_raster_source(x, y, srs_wkt=geometries.crs.to_wkt())
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
    geom_array: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    coverage_weight: CoverageWeights,
    strategy: Strategy,
    crs,
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
    strategy: Strategy = DEFAULT_STRATEGY,
    crs: Any,
) -> dask.array.Array:
    import dask.array

    if any(c == 1 for c in x.chunks[0]) or any(c == 1 for c in y.chunks[0]):
        raise ValueError("exactextract does not support a chunksize of 1. Please rechunk to avoid this")

    out = dask.array.map_blocks(
        coverage_np_dask_wrapper,
        geom_array[:, np.newaxis, np.newaxis],
        x[np.newaxis, np.newaxis, :],
        y[np.newaxis, :, np.newaxis],
        crs=crs,
        coverage_weight=coverage_weight,
        strategy=strategy,
        chunks=(*geom_array.chunks, *y.chunks, *x.chunks),
        meta=sparse.COO(
            [], data=np.array([], dtype=get_dtype(coverage_weight, geom_array)), shape=(0, 0, 0), fill_value=0
        ),
    )
    return out


def coverage(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    xdim="x",
    ydim="y",
    strategy: Strategy = "feature-sequential",
    coverage_weight: CoverageWeights = "fraction",
    clip: bool = False,
) -> xr.DataArray:
    """Calculate pixel coverage fractions for geometries using exactextract.

    This function computes how much of each raster pixel is covered by each geometry,
    returning precise fractional coverage values or area measurements. It supports
    both in-memory and dask-based computation for large datasets.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        Xarray object defining the raster grid. Must contain a 'spatial_ref'
        coordinate variable with CRS information.
    geometries : geopandas.GeoDataFrame or dask_geopandas.GeoDataFrame
        Vector geometries for which to calculate coverage. CRS should match
        the raster object (though this is not currently enforced).
    xdim : str, default "x"
        Name of the x (longitude/easting) dimension in the raster object.
    ydim : str, default "y"
        Name of the y (latitude/northing) dimension in the raster object.
    strategy : {"feature-sequential", "raster-sequential", "raster-parallel"}, default "feature-sequential"
        Processing strategy passed to exactextract. Controls how computation
        is parallelized and memory is managed.
    coverage_weight : {"fraction", "none", "area_cartesian", "area_spherical_m2", "area_spherical_km2"}, default "fraction"
        Type of coverage measurement to compute:

        - "fraction": Fractional coverage (0-1)
        - "none": Binary coverage (0 or 1)
        - "area_cartesian": Area in map units squared
        - "area_spherical_m2": Spherical area in square meters
        - "area_spherical_km2": Spherical area in square kilometers
    clip: bool
       If True, clip raster to the bounding box of the geometries.
       Ignored for dask-geopandas geometries.

    Returns
    -------
    xarray.DataArray
        3D DataArray with dimensions (geometry, y, x) containing coverage values.
        Data type depends on coverage_weight: uint8 for "none", float64 otherwise.
        Includes appropriate units and long_name attributes for area measurements.

    Raises
    ------
    ValueError
        If the raster object lacks a 'spatial_ref' coordinate or if exactextract
        encounters chunks of size 1 (not supported by exactextract).

    See Also
    --------
    exactextract.exact_extract : Underlying exactextract function used for coverage calculation

    Examples
    --------
    Calculate fractional coverage:

    >>> import rasterix.rasterize.exact as exact
    >>> import xarray as xr
    >>> import geopandas as gpd
    >>> # Load raster data with CRS info
    >>> raster = xr.open_dataarray("data.tif")
    >>> # Load vector geometries
    >>> geometries = gpd.read_file("polygons.shp")
    >>> # Calculate coverage fractions
    >>> coverage = exact.coverage(raster, geometries)
    >>> print(coverage.dims)  # ('geometry', 'y', 'x')

    Calculate area in square meters:

    >>> area_coverage = exact.coverage(raster, geometries, coverage_weight="area_spherical_m2")
    >>> print(area_coverage.units)  # 'm2'

    """
    if "spatial_ref" not in obj.coords:
        raise ValueError("Xarray object must contain the `spatial_ref` variable.")

    # FIXME: assert obj.crs == geometries.crs

    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)
    if is_in_memory(obj=obj, geometries=geometries):
        out = np_coverage(
            x=obj[xdim].data,
            y=obj[ydim].data,
            geometries=geometries,
            coverage_weight=coverage_weight,
            strategy=strategy,
        )
        geom_array = geometries.to_numpy().squeeze(axis=1)
    else:
        from dask.array import Array, from_array

        geom_dask_array = geometries_as_dask_array(geometries)
        if not isinstance(obj[xdim].data, Array):
            dask_x = from_array(obj[xdim].data, chunks=obj.chunksizes.get(xdim, -1))
        else:
            dask_x = obj[xdim].data

        if not isinstance(obj[ydim].data, Array):
            dask_y = from_array(obj[ydim].data, chunks=obj.chunksizes.get(ydim, -1))
        else:
            dask_y = obj[ydim].data

        out = dask_coverage(
            x=dask_x,
            y=dask_y,
            geom_array=geom_dask_array,
            crs=geometries.crs,
            coverage_weight=coverage_weight,
            strategy=strategy,
        )
        if isinstance(geometries, gpd.GeoDataFrame):
            geom_array = geometries.to_numpy().squeeze(axis=1)
        else:
            geom_array = geom_dask_array

    name = "coverage"
    attrs = {}
    if "area" in coverage_weight:
        name = "area"
    if "_m2" in coverage_weight or coverage_weight == "area_cartesian":
        attrs["long_name"] = coverage_weight.removesuffix("_m2")
        attrs["units"] = "m2"
    elif "_km2" in coverage_weight:
        attrs["long_name"] = coverage_weight.removesuffix("_km2")
        attrs["units"] = "km2"

    xy_coords = [
        xr.Coordinates.from_xindex(obj.xindexes.get(dim))
        for dim in (xdim, ydim)
        if obj.xindexes.get(dim) is not None
    ]
    coords = xr.Coordinates(
        coords={
            "spatial_ref": obj.spatial_ref,
            "geometry": geom_array,
        },
        indexes={},
    )
    if xy_coords:
        for c in xy_coords:
            coords = coords.merge(c)
        coords = coords.coords
    coverage = xr.DataArray(dims=("geometry", ydim, xdim), data=out, coords=coords, attrs=attrs, name=name)
    return coverage


# ============================================================================
# Engine functions for use with rasterize() in core.py
# These provide the same interface as rasterio.py and rusterize.py engines
# ============================================================================


def rasterize_geometries(
    geometries,
    *,
    dtype: np.dtype,
    shape: tuple[int, int],
    affine,
    offset: int,
    all_touched: bool = False,
    merge_alg: str = "last",
    fill: Any = 0,
    **kwargs,
) -> np.ndarray:
    """
    Rasterize geometries using exactextract for precise coverage.

    Uses exactextract to compute which pixels are covered by each geometry,
    then reduces to geometry indices using the specified merge algorithm.

    Parameters
    ----------
    geometries : Sequence[Geometry]
        Shapely geometries to rasterize.
    dtype : np.dtype
        Output data type.
    shape : tuple[int, int]
        Output shape (nrows, ncols).
    affine : Affine
        Affine transform for the output grid.
    offset : int
        Starting value for geometry indices.
    all_touched : bool
        If True, all pixels touched by geometries will be burned in.
        Note: exactextract does not support this parameter directly.
    merge_alg : str
        Merge algorithm: "last" (replace) or "sum" (add).
    fill : Any
        Fill value for pixels not covered by any geometry.
    **kwargs
        Additional arguments (ignored for compatibility).

    Returns
    -------
    np.ndarray
        Rasterized array with shape (nrows, ncols).
    """

    if all_touched:
        raise NotImplementedError(
            "all_touched=True is not supported by the exactextract engine. "
            "Use engine='rasterio' if you need all_touched support."
        )

    if len(geometries) == 0:
        return np.full(shape, fill, dtype=dtype)

    # Convert affine to x,y coordinates
    x, y = affine_to_xy_coords(affine, shape)

    # Create GeoDataFrame (exactextract requires it)
    # Use a dummy CRS - exactextract needs one but the algorithm doesn't depend on it
    gdf = gpd.GeoDataFrame(geometry=list(geometries), crs="EPSG:4326")

    # Use exactextract to get coverage (binary mode for efficiency)
    raster = xy_to_raster_source(x, y, srs_wkt=gdf.crs.to_wkt())
    result = exact_extract(
        rast=raster,
        vec=gdf,
        ops=["cell_id", "coverage(coverage_weight=none)"],
        output="pandas",
    )

    # Initialize output with fill value
    out = np.full(shape, fill, dtype=dtype)

    # Process based on merge algorithm
    if merge_alg == "last":
        # Later geometries overwrite earlier ones (MergeAlg.replace behavior)
        for i in range(len(geometries)):
            cell_ids = result.cell_id.values[i]
            if len(cell_ids) > 0:
                # Burn geometry index (with offset) into covered pixels
                np.put(out, cell_ids, offset + i)
    elif merge_alg == "sum":
        # Sum values where geometries overlap
        for i in range(len(geometries)):
            cell_ids = result.cell_id.values[i]
            if len(cell_ids) > 0:
                flat_out = out.ravel()
                flat_out[cell_ids] += offset + i
    else:
        raise ValueError(f"Unsupported merge_alg: {merge_alg}. Must be 'last' or 'sum'.")

    return out


def dask_rasterize_wrapper(
    geom_array: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    x_sizes: np.ndarray,
    y_sizes: np.ndarray,
    offset_array: np.ndarray,
    *,
    fill: Any,
    affine,
    all_touched: bool,
    merge_alg: str,
    dtype_: np.dtype,
    **kwargs,
) -> np.ndarray:
    """Dask wrapper for exactextract rasterization."""
    offset = offset_array.item()

    return rasterize_geometries(
        geom_array[:, 0, 0].tolist(),
        affine=affine * affine.translation(x_offsets.item(), y_offsets.item()),
        shape=(y_sizes.item(), x_sizes.item()),
        offset=offset,
        all_touched=all_touched,
        merge_alg=merge_alg,
        fill=fill,
        dtype=dtype_,
    )[np.newaxis, :, :]


def np_geometry_mask(
    geometries,
    *,
    shape: tuple[int, int],
    affine,
    all_touched: bool = False,
    invert: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Create a geometry mask using exactextract.

    Parameters
    ----------
    geometries : Sequence[Geometry]
        Shapely geometries for masking.
    shape : tuple[int, int]
        Output shape (nrows, ncols).
    affine : Affine
        Affine transform for the output grid.
    all_touched : bool
        If True, all pixels touched by geometries will be included.
        Note: exactextract does not support this parameter directly.
    invert : bool
        If True, pixels inside geometries are True (unmasked).
        If False (default), pixels inside geometries are False (masked).
    **kwargs
        Additional arguments (ignored for compatibility).

    Returns
    -------
    np.ndarray
        Boolean mask array with shape (nrows, ncols).
    """
    if all_touched:
        raise NotImplementedError(
            "all_touched=True is not supported by the exactextract engine. "
            "Use engine='rasterio' if you need all_touched support."
        )

    if len(geometries) == 0:
        # No geometries: all pixels are outside (masked=True) or inside (masked=False)
        return np.full(shape, not invert, dtype=bool)

    # Convert affine to x,y coordinates
    x, y = affine_to_xy_coords(affine, shape)

    # Create GeoDataFrame (exactextract requires it)
    gdf = gpd.GeoDataFrame(geometry=list(geometries), crs="EPSG:4326")

    # Use exactextract to get coverage
    raster = xy_to_raster_source(x, y, srs_wkt=gdf.crs.to_wkt())
    result = exact_extract(
        rast=raster,
        vec=gdf,
        ops=["cell_id", "coverage(coverage_weight=none)"],
        output="pandas",
    )

    # Collect all covered cell IDs
    all_cell_ids = set()
    for i in range(len(geometries)):
        cell_ids = result.cell_id.values[i]
        all_cell_ids.update(cell_ids)

    # Create mask: True = outside geometry (masked), False = inside
    inside = np.zeros(shape, dtype=bool)
    if all_cell_ids:
        np.put(inside, list(all_cell_ids), True)

    if invert:
        return inside  # True = inside
    else:
        return ~inside  # True = outside (masked)


def dask_mask_wrapper(
    geom_array: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    x_sizes: np.ndarray,
    y_sizes: np.ndarray,
    *,
    affine,
    **kwargs,
) -> np.ndarray:
    """Dask wrapper for exactextract geometry masking."""
    res = np_geometry_mask(
        geom_array[:, 0, 0].tolist(),
        shape=(y_sizes.item(), x_sizes.item()),
        affine=affine * affine.translation(x_offsets.item(), y_offsets.item()),
        **kwargs,
    )
    return res[np.newaxis, :, :]
