import xarray as xr
from affine import Affine

from rasterix.lib import logger


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

    # Check for STAC and GeoTIFF metadata in DataArray attrs
    attrs = obj.attrs if isinstance(obj, xr.DataArray) else {}

    # Try to extract affine from STAC proj:transform
    if "proj:transform" in attrs:
        logger.trace("Creating affine from STAC proj:transform attribute")
        transform = attrs["proj:transform"]
        # proj:transform is a 3x3 matrix in row-major order, but typically only 6 elements
        # [a, b, c, d, e, f, 0, 0, 1] where the affine is constructed from first 6 elements
        if len(transform) >= 6:
            a, b, c, d, e, f = transform[:6]
            if clear_transform:
                del attrs["proj:transform"]
            return Affine(a, b, c, d, e, f)

    # Try to extract affine from GeoTIFF model_tiepoint and model_pixel_scale
    if "model_tiepoint" in attrs and "model_pixel_scale" in attrs:
        logger.trace("Creating affine from GeoTIFF model_tiepoint and model_pixel_scale attributes")
        tiepoint = attrs["model_tiepoint"]
        pixel_scale = attrs["model_pixel_scale"]

        # model_tiepoint format: [I, J, K, X, Y, Z]
        # where (I, J, K) are pixel coords and (X, Y, Z) are world coords
        # model_pixel_scale format: [ScaleX, ScaleY, ScaleZ]
        if len(tiepoint) >= 6 and len(pixel_scale) >= 3:
            i, j, k, x, y, z = tiepoint[:6]
            scale_x, scale_y, scale_z = pixel_scale[:3]

            # We only support 2D rasters
            assert scale_z == 0, f"Z pixel scale must be 0 for 2D rasters, got {scale_z}"

            # The tiepoint gives us the world coordinates at pixel (I, J)
            # Affine transform: x_world = c + i * a, y_world = f + j * e
            # So: c = x - i * scale_x, f = y - j * scale_y
            c = x - i * scale_x
            f = y - j * scale_y

            # Clean up GeoTIFF metadata attributes after using them
            if clear_transform:
                del attrs["model_tiepoint"]
                del attrs["model_pixel_scale"]

            return Affine.translation(c, f) * Affine.scale(scale_x, scale_y)

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
