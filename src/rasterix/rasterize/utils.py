from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, overload

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine

if TYPE_CHECKING:
    import dask.array
    import dask_geopandas


YAXIS = 0
XAXIS = 1


@overload
def clip_to_bbox(obj: xr.Dataset, geometries: gpd.GeoDataFrame) -> xr.Dataset: ...
@overload
def clip_to_bbox(obj: xr.DataArray, geometries: gpd.GeoDataFrame) -> xr.DataArray: ...
def clip_to_bbox(
    obj: xr.Dataset | xr.DataArray, geometries: gpd.GeoDataFrame, *, xdim: str, ydim: str
) -> xr.Dataset | xr.DataArray:
    bbox = geometries.total_bounds
    if not hasattr(bbox, "chunks"):
        y = obj[ydim].data
        if y[0] < y[-1]:
            obj = obj.sel({xdim: slice(bbox[0], bbox[2]), ydim: slice(bbox[1], bbox[3])})
        else:
            obj = obj.sel({xdim: slice(bbox[0], bbox[2]), ydim: slice(bbox[3], bbox[1])})
    return obj


def get_affine(obj: xr.Dataset | xr.DataArray, *, xdim="x", ydim="y") -> Affine:
    spatial_ref = obj.coords["spatial_ref"]
    if "GeoTransform" in spatial_ref.attrs:
        return Affine.from_gdal(*map(float, spatial_ref.attrs["GeoTransform"].split(" ")))
    else:
        x = obj.coords[xdim]
        y = obj.coords[ydim]
        dx = (x[1] - x[0]).item()
        dy = (y[1] - y[0]).item()
        return Affine.translation(
            x[0].item() - dx / 2, (y[0] if dy < 0 else y[-1]).item() - dy / 2
        ) * Affine.scale(dx, dy)


def is_in_memory(*, obj, geometries) -> bool:
    return not obj.chunks and isinstance(geometries, gpd.GeoDataFrame)


def geometries_as_dask_array(
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
) -> dask.array.Array:
    from dask.array import from_array

    if isinstance(geometries, gpd.GeoDataFrame):
        return from_array(
            geometries.geometry.to_numpy(),
            chunks=-1,
            # This is what dask-geopandas does
            # It avoids pickling geometries, which can be expensive (calls to_wkb)
            name=uuid.uuid4().hex,
        )
    else:
        divisions = geometries.divisions
        if any(d is None for d in divisions):
            print("computing current divisions, this may be expensive.")
            divisions = geometries.compute_current_divisions()
        chunks = np.diff(divisions).tolist()
        chunks[-1] += 1
        return geometries.to_dask_array(lengths=chunks).squeeze(axis=1)


def prepare_for_dask(
    obj: xr.Dataset | xr.DataArray,
    geometries,
    *,
    xdim: str,
    ydim: str,
    geoms_rechunk_size: int | None,
):
    from dask.array import from_array

    chunks = (
        obj.chunksizes.get(ydim, obj.sizes[ydim]),
        obj.chunksizes.get(xdim, obj.sizes[ydim]),
    )
    geom_array = geometries_as_dask_array(geometries)
    if geoms_rechunk_size is not None:
        geom_array = geom_array.rechunk({0: geoms_rechunk_size})

    x_sizes = from_array(chunks[XAXIS], chunks=1)
    y_sizes = from_array(chunks[YAXIS], chunks=1)
    y_offsets = from_array(np.cumulative_sum(chunks[YAXIS][:-1], include_initial=True), chunks=1)
    x_offsets = from_array(np.cumulative_sum(chunks[XAXIS][:-1], include_initial=True), chunks=1)

    map_blocks_args = (
        geom_array[:, np.newaxis, np.newaxis],
        x_offsets[np.newaxis, np.newaxis, :],
        y_offsets[np.newaxis, :, np.newaxis],
        x_sizes[np.newaxis, np.newaxis, :],
        y_sizes[np.newaxis, :, np.newaxis],
    )
    return map_blocks_args, chunks, geom_array
