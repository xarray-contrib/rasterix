from typing import NotRequired, TypedDict

import xarray as xr
from affine import Affine
from pyproj import CRS

from rasterix.lib import (
    affine_from_spatial_zarr_convention,
    affine_from_stac_proj_metadata,
    affine_from_tiepoint_and_scale,
    logger,
)

# https://github.com/zarr-conventions/geo-proj
_ZARR_GEO_PROJ_CONVENTION_UUID = "f17cb550-5864-4468-aeb7-f3180cfb622f"


def get_grid_mapping_var(obj: xr.Dataset | xr.DataArray) -> xr.DataArray | None:
    grid_mapping_var = None
    if isinstance(obj, xr.DataArray):
        if maybe := obj.attrs.get("grid_mapping", None):
            if maybe in obj.coords:
                grid_mapping_var = maybe
    else:
        # for datasets, grab the first one for simplicity
        for var in obj.data_vars.values():
            if maybe := var.attrs.get("grid_mapping"):
                if maybe in obj.coords:
                    # make sure it exists and is not an out-of-date attribute
                    grid_mapping_var = maybe
                    break
    if grid_mapping_var is None and "spatial_ref" in obj.coords:
        # hardcode this
        grid_mapping_var = "spatial_ref"
    if grid_mapping_var is not None:
        return obj[grid_mapping_var]
    return None


def get_affine(
    obj: xr.Dataset | xr.DataArray, *, x_dim="x", y_dim="y", clear_transform: bool = False
) -> Affine:
    """
    Grabs an affine transform from an Xarray object.

    This method will first look for the ``"GeoTransform"`` attribute on a variable named
    ``"spatial_ref"``. If not, it will look for STAC ``proj:transform`` attribute, then
    GeoTIFF metadata (``model_tiepoint`` and ``model_pixel_scale``). Finally, it will
    auto-guess the transform from the provided ``x_dim`` and ``y_dim``.

    Parameters
    ----------
    obj: xr.DataArray or xr.Dataset
    x_dim: str, optional
        Name of the X dimension coordinate variable.
    y_dim: str, optional
        Name of the Y dimension coordinate variable.
    clear_transform: bool
       Whether to delete the transform attributes if detected.

    Returns
    -------
    affine: Affine
    """
    grid_mapping_var = get_grid_mapping_var(obj)
    if grid_mapping_var is not None and (transform := grid_mapping_var.attrs.get("GeoTransform")):
        logger.trace("Creating affine from GeoTransform attribute")
        if clear_transform:
            del grid_mapping_var.attrs["GeoTransform"]
        return Affine.from_gdal(*map(float, transform.split(" ")))

    # Check for STAC, GeoTIFF, or spatial zarr convention metadata in DataArray attrs
    attrs = obj.attrs if isinstance(obj, xr.DataArray) else {}

    # Try to extract affine from STAC proj:transform
    if affine := affine_from_stac_proj_metadata(attrs):
        logger.trace("Creating affine from STAC proj:transform attribute")
        if clear_transform:
            del attrs["proj:transform"]
        return affine

    # Try to extract affine from GeoTIFF model_tiepoint and model_pixel_scale
    if "model_tiepoint" in attrs and "model_pixel_scale" in attrs:
        logger.trace("Creating affine from GeoTIFF model_tiepoint and model_pixel_scale attributes")
        affine = affine_from_tiepoint_and_scale(attrs["model_tiepoint"], attrs["model_pixel_scale"])

        # Clean up GeoTIFF metadata attributes after using them
        if clear_transform:
            del attrs["model_tiepoint"]
            del attrs["model_pixel_scale"]

        return affine

    # Try to extract from spatial zarr convention attributes
    if affine := affine_from_spatial_zarr_convention(attrs):
        logger.trace("Creating affine from spatial zarr convention attributes")
        if clear_transform:
            del attrs["spatial:transform"]
        return affine

    # Fall back to computing from coordinate arrays
    logger.trace(f"Creating affine from coordinate arrays {x_dim=!r} and {y_dim=!r}")
    if x_dim not in obj.coords or y_dim not in obj.coords:
        raise ValueError(
            f"Cannot create affine transform: dimensions {x_dim=!r} and {y_dim=!r} "
            f"do not have explicit coordinate values and no transform metadata found."
        )

    x = obj.coords[x_dim]
    y = obj.coords[y_dim]
    if x.ndim != 1:
        raise ValueError(f"Coordinate variable {x_dim=!r} must be 1D.")
    if y.ndim != 1:
        raise ValueError(f"Coordinate variable {y_dim=!r} must be 1D.")

    # Check that coordinates have actual values (not just dimension placeholders)
    if len(x) == 0 or len(y) == 0:
        raise ValueError(
            f"Cannot create affine transform from empty coordinate arrays for {x_dim=!r} and {y_dim=!r}."
        )

    dx = (x[1] - x[0]).item()
    dy = (y[1] - y[0]).item()
    return Affine.translation(
        x[0].item() - dx / 2, (y[0] if dy < 0 else y[-1]).item() - dy / 2
    ) * Affine.scale(dx, dy)


_ZarrConventionRegistration = TypedDict("_ZarrConventionRegistration", {"proj:": str})

_ZarrProjMetadata = TypedDict(
    "_ZarrProjMetadata",
    {
        "zarr_conventions": NotRequired[list[_ZarrConventionRegistration | dict]],
        "proj:code": NotRequired[str],
        "proj:wkt2": NotRequired[str],
        "proj:projjson": NotRequired[object],
    },
)


def _has_proj_zarr_convention(metadata: _ZarrProjMetadata) -> bool:
    zarr_conventions = metadata.get("zarr_conventions")
    if not zarr_conventions:
        return False
    for entry in zarr_conventions:
        if isinstance(entry, dict) and (
            entry.get("uuid") == _ZARR_GEO_PROJ_CONVENTION_UUID or entry.get("name") == "proj:"
        ):
            return True
    return False


def get_crs_from_proj_zarr_convention(obj: xr.Dataset | xr.DataArray) -> CRS | None:
    """Extract CRS from Zarr proj: convention metadata if present.

    See https://github.com/zarr-conventions/geo-proj for more details.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        The Xarray object to extract CRS from.

    Returns
    -------
    CRS or None
        The extracted CRS object, or None if not found.
    """
    metadata: _ZarrProjMetadata = obj.attrs  # type: ignore[assignment]

    if not _has_proj_zarr_convention(metadata):
        return None

    if code := metadata.get("proj:code"):
        return CRS.from_string(code)
    if wkt2 := metadata.get("proj:wkt2"):
        return CRS.from_wkt(wkt2)
    if projjson := metadata.get("proj:projjson"):
        return CRS.from_user_input(projjson)
    return None
