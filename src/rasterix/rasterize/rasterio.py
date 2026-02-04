# rasterio-specific rasterization helpers
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from affine import Affine

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    import rasterio as rio
    from rasterio.features import MergeAlg

__all__: list[str] = []


def with_rio_env(func: F) -> F:
    """
    Decorator that handles the 'env' and 'clear_cache' kwargs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        env = kwargs.pop("env", None)
        clear_cache = kwargs.pop("clear_cache", False)

        if env is None:
            import rasterio as rio

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
