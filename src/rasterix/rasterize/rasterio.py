# rasterio wrappers
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa
import rasterio as rio
import xarray as xr
from rasterio.features import MergeAlg, geometry_mask
from rasterio.features import rasterize as rasterize_rio

from .utils import is_in_memory, prepare_for_dask

if TYPE_CHECKING:
    import dask_geopandas


def dask_rasterize_wrapper(
    geom_array: np.ndarray,
    tile_array: np.ndarray,
    offset_array: np.ndarray,
    *,
    fill: Any,
    all_touched: bool,
    merge_alg: MergeAlg,
    dtype_: np.dtype,
    env: rio.Env | None = None,
) -> np.ndarray:
    tile = tile_array.item()
    offset = offset_array.item()

    return rasterize_geometries(
        geom_array[:, 0, 0].tolist(),
        tile=tile,
        offset=offset,
        all_touched=all_touched,
        merge_alg=merge_alg,
        fill=fill,
        dtype=dtype_,
        env=env,
    )[np.newaxis, :, :]


def rasterize_geometries(
    geometries: Sequence[Any],
    *,
    dtype: np.dtype,
    tile,
    offset,
    env: rio.Env | None = None,
    clear_cache: bool = False,
    **kwargs,
):
    # From https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.rasterize
    #    The out array will be copied and additional temporary raster memory equal to 2x the smaller of out data
    #    or GDAL’s max cache size (controlled by GDAL_CACHEMAX, default is 5% of the computer’s physical memory) is required.
    #    If GDAL max cache size is smaller than the output data, the array of shapes will be iterated multiple times.
    #    Performance is thus a linear function of buffer size. For maximum speed, ensure that GDAL_CACHEMAX
    #    is larger than the size of out or out_shape.
    if env is None:
        # out_size = dtype.itemsize * math.prod(tile.shape)
        # env = rio.Env(GDAL_CACHEMAX=1.2 * out_size)
        # FIXME: figure out a good default
        env = rio.Env()
    with env:
        res = rasterize_rio(
            zip(geometries, range(offset, offset + len(geometries)), strict=True),
            out_shape=tile.shape,
            transform=tile.affine,
            **kwargs,
        )
    if clear_cache:
        with rio.Env(GDAL_CACHEMAX=0):
            try:
                from osgeo import gdal

                # attempt to force-clear the GDAL cache
                assert gdal.GetCacheMax() == 0
            except ImportError:
                pass
    assert res.shape == tile.shape
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
        Rasterio Environment configuration. For example, use set ``GDAL_CACHEMAX`
        by passing ``env = rio.Env(GDAL_CACHEMAX=100 * 1e6)``.

    Returns
    -------
    DataArray
        2D DataArray with geometries "burned in"
    """
    if xdim not in obj.dims or ydim not in obj.dims:
        raise ValueError(f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}")
    box = obj.odc.geobox
    rasterize_kwargs = dict(all_touched=all_touched, merge_alg=merge_alg)
    # FIXME: box.crs == geometries.crs
    if is_in_memory(obj=obj, geometries=geometries):
        geom_array = geometries.to_numpy().squeeze(axis=1)
        rasterized = rasterize_geometries(
            geom_array.tolist(),
            tile=box,
            offset=0,
            dtype=np.min_scalar_type(len(geometries)),
            fill=len(geometries),
            env=env,
            **rasterize_kwargs,
        )
    else:
        from dask.array import from_array, map_blocks

        chunks, tiles_array, geom_array = prepare_for_dask(
            obj, geometries, xdim=xdim, ydim=ydim, geoms_rechunk_size=geoms_rechunk_size
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
            geom_array[:, np.newaxis, np.newaxis],
            tiles_array[np.newaxis, :, :],
            offsets[:, np.newaxis, np.newaxis],
            chunks=((1,) * geom_array.numblocks[0], chunks[0], chunks[1]),
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
    tile_array: np.ndarray,
    *,
    all_touched: bool,
    invert: bool,
    env: rio.Env | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    tile = tile_array.item()

    return np_geometry_mask(
        geom_array[:, 0, 0].tolist(),
        tile=tile,
        all_touched=all_touched,
        invert=invert,
        env=env,
    )[np.newaxis, :, :]


def np_geometry_mask(
    geometries: Sequence[Any],
    *,
    tile,
    env: rio.Env | None = None,
    clear_cache: bool = False,
    **kwargs,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    # From https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.rasterize
    #    The out array will be copied and additional temporary raster memory equal to 2x the smaller of out data
    #    or GDAL’s max cache size (controlled by GDAL_CACHEMAX, default is 5% of the computer’s physical memory) is required.
    #    If GDAL max cache size is smaller than the output data, the array of shapes will be iterated multiple times.
    #    Performance is thus a linear function of buffer size. For maximum speed, ensure that GDAL_CACHEMAX
    #    is larger than the size of out or out_shape.
    if env is None:
        # out_size = np.bool_.itemsize * math.prod(tile.shape)
        # env = rio.Env(GDAL_CACHEMAX=1.2 * out_size)
        # FIXME: figure out a good default
        env = rio.Env()
    with env:
        res = geometry_mask(geometries, out_shape=tile.shape, transform=tile.affine, **kwargs)
    if clear_cache:
        with rio.Env(GDAL_CACHEMAX=0):
            try:
                from osgeo import gdal

                # attempt to force-clear the GDAL cache
                assert gdal.GetCacheMax() == 0
            except ImportError:
                pass
    assert res.shape == tile.shape
    return res


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
        Rasterio Environment configuration. For example, use set ``GDAL_CACHEMAX`
        by passing ``env = rio.Env(GDAL_CACHEMAX=100 * 1e6)``.

    Returns
    -------
    DataArray
        3D dataarray with coverage fraction. The additional dimension is "geometry".
    """
    invert = not invert  # rioxarray clip convention -> rasterio geometry_mask convention
    if xdim not in obj.dims or ydim not in obj.dims:
        raise ValueError(f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}")
    box = obj.odc.geobox
    geometry_mask_kwargs = dict(all_touched=all_touched, invert=invert)

    if is_in_memory(obj=obj, geometries=geometries):
        geom_array = geometries.to_numpy().squeeze(axis=1)
        mask = np_geometry_mask(geom_array.tolist(), tile=box, env=env, **geometry_mask_kwargs)
    else:
        from dask.array import map_blocks

        chunks, tiles_array, geom_array = prepare_for_dask(
            obj, geometries, xdim=xdim, ydim=ydim, geoms_rechunk_size=geoms_rechunk_size
        )
        mask = map_blocks(
            dask_mask_wrapper,
            geom_array[:, np.newaxis, np.newaxis],
            tiles_array[np.newaxis, :, :],
            chunks=((1,) * geom_array.numblocks[0], chunks[0], chunks[1]),
            meta=np.array([], dtype=bool),
            **geometry_mask_kwargs,
        )
        mask = mask.any(axis=0)

    mask_da = xr.DataArray(
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
    )
    return obj.where(mask_da)
