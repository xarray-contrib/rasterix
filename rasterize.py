from functools import partial
from collections.abc import Sequence
from typing import Any, Literal, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from odc.geo.geobox import GeoboxTiles
import odc.geo.xr  # noqa
import rasterio as rio
from rasterio.features import MergeAlg, rasterize, geometry_mask
from osgeo import gdal

MIN_CHUNK_SIZE = 2  # exactextract cannot handle arrays of size 1.


def is_in_memory(*, obj, geometries) -> bool:
    return not obj.chunks and isinstance(geometries, gpd.GeoDataFrame)


def geometries_as_dask_array(geometries) -> "dask.array.Array":
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
    x: "dask.array.Array",
    y: "dask.array.Array",
    *,
    geom_array: "dask.array.Array",
    coverage_weight: Literal["fraction", "none"] = "fraction",
    crs: Any,
) -> "dask.array.Array":
    import dask.array

    if any(c == 1 for c in x.chunks) or any(c == 1 for c in y.chunks):
        raise ValueError(
            "exactextract does not support a chunksize of 1. Please rechunk to avoid this"
        )

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
        dtype=get_dtype(coverage_weight),
    )


def coverage_ee(
    obj: xr.Dataset | xr.DataArray,
    geometries: Union["gpd.GeoDataFrame", "dask_geopandas.GeoDataFrame"],
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

        geom_array = geometries_as_dask_array(geometries)
        out = dask_coverage(
            x=from_array(obj[xdim].data, chunks=obj.chunksizes.get(xdim, -1)),
            y=from_array(obj[ydim].data, chunks=obj.chunksizes.get(ydim, -1)),
            geom_array=geom_array,
            crs=geometries.crs,
            coverage_weight=coverage_weight,
        )

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


# ========> RASTERIO RASTERIZE
def tiles_to_array(tiles: GeoboxTiles) -> np.ndarray:
    shape = tiles.shape
    array = np.empty(shape=(shape.y, shape.x), dtype=object)
    for i in range(shape.x):
        for j in range(shape.y):
            array[j, i] = tiles[j, i]

    assert array.shape == tiles.shape
    return array


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
        res = rasterize(
            zip(geometries, range(offset, offset + len(geometries)), strict=True),
            out_shape=tile.shape,
            transform=tile.affine,
            **kwargs,
        )
    if clear_cache:
        with rio.Env(GDAL_CACHEMAX=0):
            # attempt to force-clear the GDAL cache
            assert gdal.GetCacheMax() == 0
    assert res.shape == tile.shape
    return res


def rasterize_rio(
    obj: xr.Dataset | xr.DataArray,
    geometries: Union["gpd.GeoDataFrame", "dask_geopandas.GeoDataFrame"],
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
        raise ValueError(
            f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}"
        )
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


def replace_values(array: np.ndarray, to) -> np.ndarray:
    mask = array == 0
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
        res = geometry_mask(
            geometries, out_shape=tile.shape, transform=tile.affine, **kwargs
        )
    if clear_cache:
        with rio.Env(GDAL_CACHEMAX=0):
            # attempt to force-clear the GDAL cache
            assert gdal.GetCacheMax() == 0
    assert res.shape == tile.shape
    return res


def geometry_clip_rio(
    obj: xr.Dataset | xr.DataArray,
    geometries: Union["gpd.GeoDataFrame", "dask_geopandas.GeoDataFrame"],
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
        raise ValueError(
            f"Received {xdim=!r}, {ydim=!r} but obj.dims={tuple(obj.dims)}"
        )
    box = obj.odc.geobox
    geometry_mask_kwargs = dict(all_touched=all_touched, invert=invert)

    if is_in_memory(obj=obj, geometries=geometries):
        geom_array = geometries.to_numpy().squeeze(axis=1)
        mask = np_geometry_mask(
            geom_array.tolist(), tile=box, env=env, **geometry_mask_kwargs
        )
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
