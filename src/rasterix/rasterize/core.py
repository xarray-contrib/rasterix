# Engine-agnostic rasterization API
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import xarray as xr

from ..utils import get_affine
from .utils import XAXIS, YAXIS, clip_to_bbox, is_in_memory, prepare_for_dask

if TYPE_CHECKING:
    import dask_geopandas

__all__ = ["rasterize", "geometry_mask", "geometry_clip"]

Engine = Literal["rasterio", "rusterize"]


def _get_engine(engine: Engine | None) -> Engine:
    """Determine which engine to use based on availability."""
    if engine is not None:
        # Validate explicitly requested engine
        if engine == "rusterize":
            try:
                import rusterize as _  # noqa: F401
            except ImportError as e:
                raise ImportError("rusterize is not installed. Install it with: pip install rusterize") from e
        elif engine == "rasterio":
            try:
                import rasterio as _  # noqa: F401
            except ImportError as e:
                raise ImportError("rasterio is not installed. Install it with: pip install rasterio") from e
        return engine

    # Auto-detect: prefer rusterize, fall back to rasterio
    try:
        import rusterize as _  # noqa: F401

        return "rusterize"
    except ImportError:
        pass

    try:
        import rasterio as _  # noqa: F401

        return "rasterio"
    except ImportError:
        pass

    raise ImportError(
        "Neither rusterize nor rasterio is installed. "
        "Install one with: pip install rusterize  OR  pip install rasterio"
    )


def _get_rasterize_funcs(engine: Engine):
    """Get the engine-specific rasterize functions."""
    if engine == "rasterio":
        from . import rasterio as engine_module
    else:
        from . import rusterize as engine_module

    return (
        engine_module.rasterize_geometries,
        engine_module.dask_rasterize_wrapper,
    )


def _get_mask_funcs(engine: Engine):
    """Get the engine-specific geometry_mask functions."""
    if engine == "rasterio":
        from . import rasterio as engine_module
    else:
        from . import rusterize as engine_module

    return (
        engine_module.np_geometry_mask,
        engine_module.dask_mask_wrapper,
    )


def _normalize_merge_alg(merge_alg: str, engine: Engine) -> Any:
    """Normalize merge_alg string to engine-specific value."""
    if engine == "rasterio":
        from rasterio.features import MergeAlg

        mapping = {
            "replace": MergeAlg.replace,
            "add": MergeAlg.add,
        }
        if merge_alg not in mapping:
            raise ValueError(f"Invalid merge_alg {merge_alg!r}. Must be one of: {list(mapping.keys())}")
        return mapping[merge_alg]
    else:
        # rusterize uses different names
        mapping = {
            "replace": "last",
            "add": "sum",
        }
        if merge_alg not in mapping:
            raise ValueError(f"Invalid merge_alg {merge_alg!r}. Must be one of: {list(mapping.keys())}")
        return mapping[merge_alg]


def replace_values(array: np.ndarray, to, *, from_=0) -> np.ndarray:
    """Replace fill values and adjust offsets after dask rasterization."""
    mask = array == from_
    array[~mask] -= 1
    array[mask] = to
    return array


def rasterize(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    engine: Engine | None = None,
    xdim: str = "x",
    ydim: str = "y",
    all_touched: bool = False,
    merge_alg: str = "replace",
    geoms_rechunk_size: int | None = None,
    clip: bool = False,
    **engine_kwargs,
) -> xr.DataArray:
    """
    Dask-aware rasterization of geometries.

    Returns a 2D DataArray with integer codes for cells that are within the provided geometries.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Xarray object whose grid to rasterize onto.
    geometries : GeoDataFrame
        Either a geopandas or dask_geopandas GeoDataFrame.
    engine : {"rasterio", "rusterize"} or None
        Rasterization engine to use. If None, auto-detects based on installed
        packages (prefers rusterize if available, falls back to rasterio).
    xdim : str
        Name of the "x" dimension on ``obj``.
    ydim : str
        Name of the "y" dimension on ``obj``.
    all_touched : bool
        If True, all pixels touched by geometries will be burned in.
        If False, only pixels whose center is within the geometry are burned.
    merge_alg : {"replace", "add"}
        Merge algorithm when geometries overlap.
        - "replace": later geometries overwrite earlier ones
        - "add": values are summed where geometries overlap
    geoms_rechunk_size : int or None
        Size to rechunk the geometry array to *after* conversion from dataframe.
    clip : bool
        If True, clip raster to the bounding box of the geometries.
        Ignored for dask-geopandas geometries.
    **engine_kwargs
        Additional keyword arguments passed to the engine.
        For rasterio: ``env`` (rasterio.Env for GDAL configuration).

    Returns
    -------
    DataArray
        2D DataArray with geometries "burned in" as integer codes.

    Notes
    -----
    Different engines may produce slightly different results at pixel boundaries
    due to differences in how they handle geometry-pixel intersection tests.

    See Also
    --------
    rasterio.features.rasterize
    rusterize.rusterize
    """
    if xdim not in obj.dims or ydim not in obj.dims:
        raise ValueError(f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}")

    resolved_engine = _get_engine(engine)

    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)

    affine = get_affine(obj, x_dim=xdim, y_dim=ydim)
    engine_merge_alg = _normalize_merge_alg(merge_alg, resolved_engine)

    rasterize_geometries, dask_rasterize_wrapper = _get_rasterize_funcs(resolved_engine)

    rasterize_kwargs = dict(
        all_touched=all_touched,
        merge_alg=engine_merge_alg,
        affine=affine,
        **engine_kwargs,
    )

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
        if merge_alg == "replace":
            rasterized = rasterized.max(axis=0)
        elif merge_alg == "add":
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
            },
            indexes={xdim: obj.xindexes[xdim], ydim: obj.xindexes[ydim]},
        ),
        name="rasterized",
    )


def geometry_mask(
    obj: xr.Dataset | xr.DataArray,
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
    *,
    engine: Engine | None = None,
    xdim: str = "x",
    ydim: str = "y",
    all_touched: bool = False,
    invert: bool = False,
    geoms_rechunk_size: int | None = None,
    clip: bool = False,
    **engine_kwargs,
) -> xr.DataArray:
    """
    Dask-aware geometry masking.

    Creates a boolean mask from geometries.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Xarray object used to extract the grid.
    geometries : GeoDataFrame or DaskGeoDataFrame
        Geometries used for masking.
    engine : {"rasterio", "rusterize"} or None
        Rasterization engine to use. If None, auto-detects based on installed
        packages (prefers rusterize if available, falls back to rasterio).
    xdim : str
        Name of the "x" dimension on ``obj``.
    ydim : str
        Name of the "y" dimension on ``obj``.
    all_touched : bool
        If True, all pixels touched by geometries will be included in mask.
    invert : bool
        If True, pixels inside geometries are True (unmasked).
        If False (default), pixels inside geometries are False (masked).
    geoms_rechunk_size : int or None
        Chunksize for geometry dimension of the output.
    clip : bool
        If True, clip raster to the bounding box of the geometries.
        Ignored for dask-geopandas geometries.
    **engine_kwargs
        Additional keyword arguments passed to the engine.
        For rasterio: ``env`` (rasterio.Env for GDAL configuration).

    Returns
    -------
    DataArray
        2D boolean DataArray mask.

    See Also
    --------
    rasterio.features.geometry_mask
    """
    if xdim not in obj.dims or ydim not in obj.dims:
        raise ValueError(f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}")

    resolved_engine = _get_engine(engine)

    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)

    affine = get_affine(obj, x_dim=xdim, y_dim=ydim)

    np_geometry_mask, dask_mask_wrapper = _get_mask_funcs(resolved_engine)

    geometry_mask_kwargs = dict(
        all_touched=all_touched,
        affine=affine,
        **engine_kwargs,
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
    engine: Engine | None = None,
    xdim: str = "x",
    ydim: str = "y",
    all_touched: bool = False,
    invert: bool = False,
    geoms_rechunk_size: int | None = None,
    clip: bool = True,
    **engine_kwargs,
) -> xr.DataArray:
    """
    Dask-aware geometry clipping.

    Clips an xarray object to geometries by masking values outside the geometries.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Xarray object to clip.
    geometries : GeoDataFrame or DaskGeoDataFrame
        Geometries used for clipping.
    engine : {"rasterio", "rusterize"} or None
        Rasterization engine to use. If None, auto-detects based on installed
        packages (prefers rusterize if available, falls back to rasterio).
    xdim : str
        Name of the "x" dimension on ``obj``.
    ydim : str
        Name of the "y" dimension on ``obj``.
    all_touched : bool
        If True, all pixels touched by geometries will be included.
    invert : bool
        If True, preserve values outside the geometry (invert the clip).
        If False (default), preserve values inside the geometry.
    geoms_rechunk_size : int or None
        Chunksize for geometry dimension of the output.
    clip : bool
        If True, clip raster to the bounding box of the geometries.
        Ignored for dask-geopandas geometries.
    **engine_kwargs
        Additional keyword arguments passed to the engine.
        For rasterio: ``env`` (rasterio.Env for GDAL configuration).

    Returns
    -------
    DataArray
        Clipped DataArray with values outside geometries set to NaN.

    See Also
    --------
    geometry_mask
    rasterio.features.geometry_mask
    """
    if clip:
        obj = clip_to_bbox(obj, geometries, xdim=xdim, ydim=ydim)

    mask = geometry_mask(
        obj,
        geometries,
        engine=engine,
        all_touched=all_touched,
        invert=not invert,  # rioxarray clip convention -> geometry_mask convention
        xdim=xdim,
        ydim=ydim,
        geoms_rechunk_size=geoms_rechunk_size,
        clip=False,
        **engine_kwargs,
    )
    return obj.where(mask)
