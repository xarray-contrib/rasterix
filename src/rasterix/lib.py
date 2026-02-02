"""Shared library utilities for rasterix."""

import logging
from typing import NotRequired, TypedDict

from affine import Affine

# Define TRACE level (lower than DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class TraceLogger(logging.Logger):
    """Logger with trace level support."""

    def trace(self, message, *args, **kwargs):
        """Log a message with severity 'TRACE'."""
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kwargs)


# Set the custom logger class
logging.setLoggerClass(TraceLogger)

# Create logger for the rasterix package
logger = logging.getLogger("rasterix")


def affine_from_tiepoint_and_scale(
    tiepoint: list[float] | tuple[float, ...],
    scale: list[float] | tuple[float, ...],
) -> Affine:
    """Create an Affine transform from GeoTIFF tiepoint and pixel scale.

    Parameters
    ----------
    tiepoint : list or tuple
        GeoTIFF model tiepoint in format [I, J, K, X, Y, Z]
        where (I, J, K) are pixel coords and (X, Y, Z) are world coords.
    scale : list or tuple
        GeoTIFF model pixel scale in format [ScaleX, ScaleY, ScaleZ].

    Returns
    -------
    Affine
        Affine transformation matrix.

    Raises
    ------
    AssertionError
        If ScaleZ is not 0 (only 2D rasters are supported).

    Examples
    --------
    >>> tiepoint = [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0]
    >>> scale = [30.0, 30.0, 0.0]
    >>> affine = affine_from_tiepoint_and_scale(tiepoint, scale)
    """
    if len(tiepoint) < 6:
        raise ValueError(f"tiepoint must have at least 6 elements, got {len(tiepoint)}")
    if len(scale) < 3:
        raise ValueError(f"scale must have at least 3 elements, got {len(scale)}")

    i, j, k, x, y, z = tiepoint[:6]
    scale_x, scale_y, scale_z = scale[:3]

    # We only support 2D rasters
    assert scale_z == 0, f"Z pixel scale must be 0 for 2D rasters, got {scale_z}"

    # The tiepoint gives us the world coordinates at pixel (I, J)
    # Affine transform: x_world = c + i * a, y_world = f + j * e
    # So: c = x - i * scale_x, f = y - j * scale_y
    c = x - i * scale_x
    f = y - j * scale_y

    return Affine.translation(c, f) * Affine.scale(scale_x, scale_y)


def affine_from_stac_proj_metadata(metadata: dict) -> Affine | None:
    """Extract Affine transform from STAC projection metadata.

    Parameters
    ----------
    metadata : dict
        Dictionary containing STAC metadata. Should contain a 'proj:transform' key.

    Returns
    -------
    Affine or None
        Affine transformation matrix if 'proj:transform' is found, None otherwise.

    Examples
    --------
    >>> metadata = {"proj:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0]}
    >>> affine = affine_from_stac_proj_metadata(metadata)
    """
    if "proj:transform" not in metadata:
        return None

    transform = metadata["proj:transform"]
    # proj:transform is a 3x3 matrix in row-major order, but typically only 6 elements
    # [a, b, c, d, e, f, 0, 0, 1] where the affine is constructed from first 6 elements
    if len(transform) < 6:
        raise ValueError(f"proj:transform must have at least 6 elements, got {len(transform)}")

    a, b, c, d, e, f = transform[:6]
    return Affine(a, b, c, d, e, f)


_ZarrConventionRegistration = TypedDict("_ZarrConventionRegistration", {"spatial:": str})

_ZarrSpatialMetadata = TypedDict(
    "_ZarrSpatialMetadata",
    {
        "zarr_conventions": NotRequired[list[_ZarrConventionRegistration | dict]],
        "spatial:transform": NotRequired[list[float]],
        "spatial:transform_type": NotRequired[str],
        "spatial:registration": NotRequired[str],
    },
)


def affine_from_spatial_zarr_convention(metadata: dict) -> Affine | None:
    """Extract Affine transform from Zarr spatial convention metadata.

    See https://github.com/zarr-conventions/spatial for the full specification.

    Parameters
    ----------
    metadata : dict
        Dictionary containing Zarr spatial convention metadata.

    Returns
    -------
    Affine or None
        Affine transformation matrix if minimal Zarr spatial metadata is found, None otherwise.

    Examples
    --------
    >>> ds: xr.Dataset = ...
    >>> affine = affine_from_spatial_zarr_convention(ds.attrs)
    """
    possibly_spatial_metadata: _ZarrSpatialMetadata = metadata  # type: ignore[assignment]

    if "zarr_conventions" in possibly_spatial_metadata:
        conventions = possibly_spatial_metadata["zarr_conventions"]
        if any("spatial:" in convention for convention in conventions):
            if transform := possibly_spatial_metadata.get("spatial:transform"):
                if len(transform) < 6:
                    raise ValueError(f"spatial:transform must have at least 6 elements, got {len(transform)}")

                transform_type = possibly_spatial_metadata.get("spatial:transform_type", "affine")
                if transform_type != "affine":
                    raise NotImplementedError(
                        f"Unsupported spatial:transform_type {transform_type!r}; only 'affine' is supported."
                    )

                registration = possibly_spatial_metadata.get("spatial:registration", "pixel")
                if registration != "pixel":
                    raise NotImplementedError(
                        f"Unsupported spatial:registration {registration!r}; only 'pixel' is supported."
                    )

                return Affine(*map(float, transform[:6]))

    return None
