# rasterio wrappers
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from affine import Affine
from rasterio.features import MergeAlg
from rasterio.features import geometry_mask as geometry_mask_rio
from rasterio.features import rasterize as rasterize_rio

from .utils import XAXIS, YAXIS, clip_to_bbox, get_affine, is_in_memory, prepare_for_dask

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    import dask_geopandas

__all__ = ["geometry_mask", "rasterize", "geometry_clip"]


def with_rio_env(func: F) -> F:
    """
    Decorator that handles the 'env' and 'clear_cache' kwargs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        env = kwargs.pop("env", None)
        clear_cache = kwargs.pop("clear_cache", False)

        if env is None:
            env = rio.Env()

        with env:
            # Remove env and clear_cache from kwargs before calling the wrapped function
            # since the function shouldn't handle the context management
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
    res = rasterize_rio(
        zip(geometries, range(offset, offset + len(geometries)), strict=True),
        out_shape=shape,
        transform=affine,
        **kwargs,
    )
    assert res.shape == shape
    return res


def rasterize(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    xdim="x",
    ydim="y",
    all_touched: bool = False,
    merge_alg: MergeAlg = MergeAlg.replace,
    geoms_rechunk_size: int | None = None,
    env: rio.Env | None = None,
    clip: bool = False,
) -> xr.DataArray:
    """
    Dask-aware wrapper around ``rasterio.features.rasterize``.

    Returns a 2D DataArray with integer codes for cells that are within the provided geometries.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Xarray object, whose grid to rasterize
    geometries: GeoDataFrame
        Either a geopandas or dask_geopandas GeoDataFrame
    xdim: str
        Name of the "x" dimension on ``obj``.
    ydim: str
        Name of the "y" dimension on ``obj``.
    all_touched: bool = False
        Passed to ``rasterio.features.rasterize``
    merge_alg: rasterio.MergeAlg
        Passed to ``rasterio.features.rasterize``.
    geoms_rechunk_size: int | None = None
        Size to rechunk the geometry array to *after* conversion from dataframe.
    env: rasterio.Env
        Rasterio Environment configuration. For example, use set ``GDAL_CACHEMAX``
        by passing ``env = rio.Env(GDAL_CACHEMAX=100 * 1e6)``.
    clip: bool
       If True, clip raster to the bounding box of the geometries.
       Ignored for dask-geopandas geometries.

    Returns
    -------
    DataArray
        2D DataArray with geometries "burned in"

    See Also
    --------
    rasterio.features.rasterize
    """
    if xdim not in obj.dims or ydim not in obj.dims:
        raise ValueError(f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}")

    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)

    rasterize_kwargs = dict(
        all_touched=all_touched, merge_alg=merge_alg, affine=get_affine(obj, xdim=xdim, ydim=ydim), env=env
    )
    # FIXME: box.crs == geometries.crs

    if is_in_memory(obj=obj, geometries=geometries):
        geom_array = geometries.to_numpy().squeeze(axis=1)
        rasterized = rasterize_geometries(
            geom_array.tolist(),
            shape=(obj.sizes[ydim], obj.sizes[xdim]),
            offset=0,
            dtype=np.min_scalar_type(len(geometries)),
            fill=len(geometries),
            **rasterize_kwargs,
        )
    else:
        from dask.array import from_array, map_blocks

        map_blocks_args, chunks, geom_array = prepare_for_dask(
            obj,
            geometries,
            xdim=xdim,
            ydim=ydim,
            geoms_rechunk_size=geoms_rechunk_size,
        )
        # DaskGeoDataFrame.len() computes!
        num_geoms = geom_array.size
        # with dask, we use 0 as a fill value and replace it later
        dtype = np.min_scalar_type(num_geoms)
        # add 1 to the offset, to account for 0 as fill value
        npoffsets = np.cumsum(np.array([0, *geom_array.chunks[0][:-1]])) + 1
        offsets = from_array(npoffsets, chunks=1)

        rasterized = map_blocks(
            dask_rasterize_wrapper,
            *map_blocks_args,
            offsets[:, np.newaxis, np.newaxis],
            chunks=((1,) * geom_array.numblocks[0], chunks[YAXIS], chunks[XAXIS]),
            meta=np.array([], dtype=dtype),
            fill=0,  # good identity value for both sum & replace.
            **rasterize_kwargs,
            dtype_=dtype,
        )
        if merge_alg is MergeAlg.replace:
            rasterized = rasterized.max(axis=0)
        elif merge_alg is MergeAlg.add:
            rasterized = rasterized.sum(axis=0)

        # and reduce every other value by 1
        rasterized = rasterized.map_blocks(partial(replace_values, to=num_geoms))

    return xr.DataArray(
        dims=(ydim, xdim),
        data=rasterized,
        coords=xr.Coordinates(
            coords={
                xdim: obj.coords[xdim],
                ydim: obj.coords[ydim],
                "spatial_ref": obj.spatial_ref,
                # TODO: figure out how to propagate geometry array
                # "geometry": geom_array,
            },
            indexes={xdim: obj.xindexes[xdim], ydim: obj.xindexes[ydim]},
        ),
        name="rasterized",
    )


def replace_values(array: np.ndarray, to, *, from_=0) -> np.ndarray:
    mask = array == from_
    array[~mask] -= 1
    array[mask] = to
    return array


# ===========> geometry_mask


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
    res = geometry_mask_rio(geometries, out_shape=shape, transform=affine, **kwargs)
    assert res.shape == shape
    return res


def geometry_mask(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    xdim="x",
    ydim="y",
    all_touched: bool = False,
    invert: bool = False,
    geoms_rechunk_size: int | None = None,
    env: rio.Env | None = None,
    clip: bool = False,
) -> xr.DataArray:
    """
    Dask-ified version of ``rasterio.features.geometry_mask``

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        Xarray object used to extract the grid
    geometries: GeoDataFrame | DaskGeoDataFrame
        Geometries used for clipping
    xdim: str
        Name of the "x" dimension on ``obj``.
    ydim: str
        Name of the "y" dimension on ``obj``
    all_touched: bool
        Passed to rasterio
    invert: bool
        Whether to preserve values inside the geometry.
    geoms_rechunk_size: int | None = None,
        Chunksize for geometry dimension of the output.
    env: rasterio.Env
        Rasterio Environment configuration. For example, use set ``GDAL_CACHEMAX``
        by passing ``env = rio.Env(GDAL_CACHEMAX=100 * 1e6)``.
    clip: bool
       If True, clip raster to the bounding box of the geometries.
       Ignored for dask-geopandas geometries.

    Returns
    -------
    DataArray
        3D dataarray with coverage fraction. The additional dimension is "geometry".

    See Also
    --------
    rasterio.features.geometry_mask
    """
    if xdim not in obj.dims or ydim not in obj.dims:
        raise ValueError(f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}")
    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)

    geometry_mask_kwargs = dict(
        all_touched=all_touched, affine=get_affine(obj, xdim=xdim, ydim=ydim), env=env
    )

    if is_in_memory(obj=obj, geometries=geometries):
        geom_array = geometries.to_numpy().squeeze(axis=1)
        mask = np_geometry_mask(
            geom_array.tolist(),
            shape=(obj.sizes[ydim], obj.sizes[xdim]),
            invert=invert,
            **geometry_mask_kwargs,
        )
    else:
        from dask.array import map_blocks

        map_blocks_args, chunks, geom_array = prepare_for_dask(
            obj,
            geometries,
            xdim=xdim,
            ydim=ydim,
            geoms_rechunk_size=geoms_rechunk_size,
        )
        mask = map_blocks(
            dask_mask_wrapper,
            *map_blocks_args,
            chunks=((1,) * geom_array.numblocks[0], chunks[YAXIS], chunks[XAXIS]),
            meta=np.array([], dtype=bool),
            **geometry_mask_kwargs,
        )
        mask = mask.all(axis=0)
        if invert:
            mask = ~mask

    return xr.DataArray(
        dims=(ydim, xdim),
        data=mask,
        coords=xr.Coordinates(
            coords={
                xdim: obj.coords[xdim],
                ydim: obj.coords[ydim],
                "spatial_ref": obj.spatial_ref,
            },
            indexes={xdim: obj.xindexes[xdim], ydim: obj.xindexes[ydim]},
        ),
        name="mask",
    )


def geometry_clip(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    xdim="x",
    ydim="y",
    all_touched: bool = False,
    invert: bool = False,
    geoms_rechunk_size: int | None = None,
    env: rio.Env | None = None,
    clip: bool = True,
) -> xr.DataArray:
    """
    Dask-ified version of rioxarray.clip

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        Xarray object used to extract the grid
    geometries: GeoDataFrame | DaskGeoDataFrame
        Geometries used for clipping
    xdim: str
        Name of the "x" dimension on ``obj``.
    ydim: str
        Name of the "y" dimension on ``obj``
    all_touched: bool
        Passed to rasterio
    invert: bool
        Whether to preserve values inside the geometry.
    geoms_rechunk_size: int | None = None,
        Chunksize for geometry dimension of the output.
    env: rasterio.Env
        Rasterio Environment configuration. For example, use set ``GDAL_CACHEMAX``
        by passing ``env = rio.Env(GDAL_CACHEMAX=100 * 1e6)``.
    clip: bool
       If True, clip raster to the bounding box of the geometries.
       Ignored for dask-geopandas geometries.

    Returns
    -------
    DataArray
        3D dataarray with coverage fraction. The additional dimension is "geometry".

    See Also
    --------
    rasterio.features.geometry_mask
    """
    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)
    mask = geometry_mask(
        obj,
        geometries,
        all_touched=all_touched,
        invert=not invert,  # rioxarray clip convention -> rasterio geometry_mask convention
        env=env,
        xdim=xdim,
        ydim=ydim,
        geoms_rechunk_size=geoms_rechunk_size,
        clip=False,
    )
    return obj.where(mask)
