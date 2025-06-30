import xarray as xr
from affine import Affine


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


def get_affine(obj: xr.Dataset | xr.DataArray, *, x_dim="x", y_dim="y") -> Affine:
    """
    Grabs an affine transform from an Xarray object.

    This method will first look for the ``"GeoTransform"`` attribute on a variable named
    ``"spatial_ref"``. If not, it will auto-guess the transform from the provided ``x_dim``,
    and ``y_dim``.

    Parameters
    ----------
    obj: xr.DataArray or xr.Dataset
    x_dim: str, optional
        Name of the X dimension coordinate variable.
    y_dim: str, optional
        Name of the Y dimension coordinate variable.

    Returns
    -------
    affine: Affine
    """
    grid_mapping_var = get_grid_mapping_var(obj)
    if grid_mapping_var is not None and (transform := grid_mapping_var.attrs.get("GeoTransform")):
        return Affine.from_gdal(*map(float, transform.split(" ")))
    else:
        x = obj.coords[x_dim]
        y = obj.coords[y_dim]
        if x.ndim != 1:
            raise ValueError(f"Coordinate variable {x_dim=!r} must be 1D.")
        if y.ndim != 1:
            raise ValueError(f"Coordinate variable {y_dim=!r} must be 1D.")
        dx = (x[1] - x[0]).item()
        dy = (y[1] - y[0]).item()
        return Affine.translation(
            x[0].item() - dx / 2, (y[0] if dy < 0 else y[-1]).item() - dy / 2
        ) * Affine.scale(dx, dy)
