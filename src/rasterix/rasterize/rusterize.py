# rusterize-specific rasterization helpers
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely import Geometry

__all__: list[str] = []


def _affine_to_extent_and_res(
    affine: Affine, shape: tuple[int, int]
) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    """Convert affine transform and shape to extent and resolution for rusterize."""
    nrows, ncols = shape
    # affine maps pixel (col, row) to (x, y)
    # top-left corner of pixel (0, 0)
    xmin = affine.c
    ymax = affine.f
    xres = affine.a
    yres = affine.e  # typically negative

    xmax = xmin + ncols * xres
    ymin = ymax + nrows * yres

    # Ensure proper ordering
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    return (xmin, ymin, xmax, ymax), (abs(xres), abs(yres))


def rasterize_geometries(
    geometries: Sequence[Geometry],
    *,
    dtype: np.dtype,
    shape: tuple[int, int],
    affine: Affine,
    offset: int,
    all_touched: bool = False,
    merge_alg: str = "replace",
    fill: Any = 0,
    **kwargs,
) -> np.ndarray:
    """
    Rasterize geometries using rusterize.

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
        Note: rusterize may not support this parameter directly.
    merge_alg : str
        Merge algorithm: "replace" or "add".
    fill : Any
        Fill value for pixels not covered by any geometry.
    **kwargs
        Additional arguments (ignored for compatibility).

    Returns
    -------
    np.ndarray
        Rasterized array with shape (nrows, ncols).
    """
    from rusterize import rusterize

    if all_touched:
        raise NotImplementedError(
            "all_touched=True is not supported by the rusterize engine. "
            "Use engine='rasterio' if you need all_touched support."
        )

    # Create GeoDataFrame with index values
    # Dummy CRS required by rusterize but not used by the algorithm
    # https://github.com/ttrotto/rusterize/issues/10
    values = list(range(offset, offset + len(geometries)))
    gdf = gpd.GeoDataFrame({"value": values}, geometry=list(geometries), crs="EPSG:4326")

    extent, (xres, yres) = _affine_to_extent_and_res(affine, shape)

    # Translate merge_alg to rusterize's native names
    rusterize_merge_alg = {"replace": "last", "add": "sum"}.get(merge_alg)
    if rusterize_merge_alg is None:
        raise ValueError(f"Unsupported merge_alg: {merge_alg}. Must be 'replace' or 'add'.")

    result = rusterize(
        gdf,
        res=(xres, yres),
        extent=extent,
        out_shape=shape,
        field="value",
        fun=rusterize_merge_alg,
        background=fill,
        encoding="numpy",
        dtype=str(dtype),
    )

    # rusterize returns (1, nrows, ncols), squeeze to (nrows, ncols) if needed
    if result.ndim == 3 and result.shape[0] == 1:
        result = result.squeeze(axis=0)
    assert result.shape == shape
    return result


def dask_rasterize_wrapper(
    geom_array: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    x_sizes: np.ndarray,
    y_sizes: np.ndarray,
    offset_array: np.ndarray,
    *,
    fill: Any,
    affine: Affine,
    all_touched: bool,
    merge_alg: str,
    dtype_: np.dtype,
    **kwargs,
) -> np.ndarray:
    """Dask wrapper for rusterize rasterization."""
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
    geometries: Sequence[Geometry],
    *,
    shape: tuple[int, int],
    affine: Affine,
    all_touched: bool = False,
    invert: bool = False,
    **kwargs,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Create a geometry mask using rusterize.

    Rasterizes geometries with burn value 1, then converts to boolean mask.

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
        Note: rusterize may not support this parameter directly.
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
    from rusterize import rusterize

    if all_touched:
        raise NotImplementedError(
            "all_touched=True is not supported by the rusterize engine. "
            "Use engine='rasterio' if you need all_touched support."
        )

    # Create GeoDataFrame with burn value
    # Dummy CRS required by rusterize but not used by the algorithm
    # https://github.com/ttrotto/rusterize/issues/10
    gdf = gpd.GeoDataFrame(geometry=list(geometries), crs="EPSG:4326")

    extent, (xres, yres) = _affine_to_extent_and_res(affine, shape)

    result = rusterize(
        gdf,
        res=(xres, yres),
        extent=extent,
        out_shape=shape,
        burn=1,
        fun="any",
        background=0,
        encoding="numpy",
        dtype="uint8",
    )

    # rusterize returns (1, nrows, ncols), squeeze to (nrows, ncols) if needed
    if result.ndim == 3 and result.shape[0] == 1:
        result = result.squeeze(axis=0)

    # Convert to boolean mask
    # rasterio convention: True = outside geometry (masked), False = inside geometry
    # invert=True flips this
    inside = result > 0
    if invert:
        return inside
    else:
        return ~inside


def dask_mask_wrapper(
    geom_array: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    x_sizes: np.ndarray,
    y_sizes: np.ndarray,
    *,
    affine: Affine,
    **kwargs,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Dask wrapper for rusterize geometry masking."""
    res = np_geometry_mask(
        geom_array[:, 0, 0].tolist(),
        shape=(y_sizes.item(), x_sizes.item()),
        affine=affine * affine.translation(x_offsets.item(), y_offsets.item()),
        **kwargs,
    )
    return res[np.newaxis, :, :]
