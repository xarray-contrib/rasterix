from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import xarray as xr
from odc.geo.geobox import GeoboxTiles

if TYPE_CHECKING:
    import dask.array
    import dask_geopandas


def tiles_to_array(tiles: GeoboxTiles) -> np.ndarray:
    shape = tiles.shape
    array = np.empty(shape=(shape.y, shape.x), dtype=object)
    for i in range(shape.x):
        for j in range(shape.y):
            array[j, i] = tiles[j, i]

    assert array.shape == tiles.shape
    return array


def is_in_memory(*, obj, geometries) -> bool:
    return not obj.chunks and isinstance(geometries, gpd.GeoDataFrame)


def geometries_as_dask_array(
    geometries: gpd.GeoDataFrame | dask_geopandas.GeoDataFrame,
) -> dask.array.Array:
    from dask.array import from_array

    if isinstance(geometries, gpd.GeoDataFrame):
        return from_array(geometries.geometry.to_numpy(), chunks=-1)
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

    box = obj.odc.geobox

    chunks = (
        obj.chunksizes.get(ydim, obj.sizes[ydim]),
        obj.chunksizes.get(xdim, obj.sizes[ydim]),
    )
    tiles = GeoboxTiles(box, tile_shape=chunks)
    tiles_array = from_array(tiles_to_array(tiles), chunks=(1, 1))
    geom_array = geometries_as_dask_array(geometries)
    if geoms_rechunk_size is not None:
        geom_array = geom_array.rechunk({0: geoms_rechunk_size})
    return chunks, tiles_array, geom_array
