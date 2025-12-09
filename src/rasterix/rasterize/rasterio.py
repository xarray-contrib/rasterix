# rasterio-specific rasterization helpers
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from affine import Affine

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    import dask_geopandas
    import geopandas as gpd
    import rasterio as rio
    import xarray as xr
    from rasterio.features import MergeAlg

__all__ = ["geometry_clip"]


def with_rio_env(func: F) -> F:
    """
    Decorator that handles the 'env' and 'clear_cache' kwargs.
    """
    import rasterio as rio

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        env = kwargs.pop("env", None)
        clear_cache = kwargs.pop("clear_cache", False)

        if env is None:
            env = rio.Env()

        with env:
            result = func(*args, **kwargs)

        if clear_cache:
            with rio.Env(GDAL_CACHEMAX=0):
                # attempt to force-clear the GDAL cache
                pass

        return result

    return wrapper


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
    merge_alg: MergeAlg,
    dtype_: np.dtype,
    env: rio.Env | None = None,
) -> np.ndarray:
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
        env=env,
    )[np.newaxis, :, :]


@with_rio_env
def rasterize_geometries(
    geometries: Sequence[Any],
    *,
    dtype: np.dtype,
    shape: tuple[int, int],
    affine: Affine,
    offset: int,
    env: rio.Env | None = None,
    clear_cache: bool = False,
    **kwargs,
):
    from rasterio.features import rasterize as rasterize_rio

    res = rasterize_rio(
        zip(geometries, range(offset, offset + len(geometries)), strict=True),
        out_shape=shape,
        transform=affine,
        **kwargs,
    )
    assert res.shape == shape
    return res


# ===========> geometry_mask helpers


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
    res = np_geometry_mask(
        geom_array[:, 0, 0].tolist(),
        shape=(y_sizes.item(), x_sizes.item()),
        affine=affine * affine.translation(x_offsets.item(), y_offsets.item()),
        **kwargs,
    )
    return res[np.newaxis, :, :]


@with_rio_env
def np_geometry_mask(
    geometries: Sequence[Any],
    *,
    shape: tuple[int, int],
    affine: Affine,
    env: rio.Env | None = None,
    clear_cache: bool = False,
    **kwargs,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    from rasterio.features import geometry_mask as geometry_mask_rio

    res = geometry_mask_rio(geometries, out_shape=shape, transform=affine, **kwargs)
    assert res.shape == shape
    return res


# ===========> geometry_clip (rasterio-specific)


def geometry_clip(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    xdim: str = "x",
    ydim: str = "y",
    all_touched: bool = False,
    invert: bool = False,
    geoms_rechunk_size: int | None = None,
    env: rio.Env | None = None,
    clip: bool = True,
) -> xr.DataArray:
    """
    Dask-ified version of rioxarray.clip

    This function is rasterio-specific.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Xarray object used to extract the grid
    geometries : GeoDataFrame or DaskGeoDataFrame
        Geometries used for clipping
    xdim : str
        Name of the "x" dimension on ``obj``.
    ydim : str
        Name of the "y" dimension on ``obj``
    all_touched : bool
        Passed to rasterio
    invert : bool
        Whether to preserve values inside the geometry.
    geoms_rechunk_size : int or None
        Chunksize for geometry dimension of the output.
    env : rasterio.Env
        Rasterio Environment configuration. For example, use set ``GDAL_CACHEMAX``
        by passing ``env = rio.Env(GDAL_CACHEMAX=100 * 1e6)``.
    clip : bool
        If True, clip raster to the bounding box of the geometries.
        Ignored for dask-geopandas geometries.

    Returns
    -------
    DataArray
        Clipped DataArray.

    See Also
    --------
    rasterio.features.geometry_mask
    """
    from .core import geometry_mask
    from .utils import clip_to_bbox

    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)
    mask = geometry_mask(
        obj,
        geometries,
        engine="rasterio",
        all_touched=all_touched,
        invert=not invert,  # rioxarray clip convention -> rasterio geometry_mask convention
        xdim=xdim,
        ydim=ydim,
        geoms_rechunk_size=geoms_rechunk_size,
        clip=False,
        env=env,
    )
    return obj.where(mask)
