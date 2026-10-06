"""Options for rasterix with context manager support."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

OPTIONS: dict[str, Any] = {
    "transform_rtol": 1e-12,
    "transform_atol": 0.0,
    "period_rtol": 0.0,
    "period_atol": 1e-3,
}


def _validate_tolerance(value: Any) -> bool:
    """Validate that value is a non-negative float."""
    return isinstance(value, int | float) and value >= 0


_VALIDATORS = {
    "transform_rtol": _validate_tolerance,
    "transform_atol": _validate_tolerance,
    "period_rtol": _validate_tolerance,
    "period_atol": _validate_tolerance,
}


@contextmanager
def set_options(**kwargs):
    """
    Set options for rasterix in a controlled context.

    Parameters
    ----------
    transform_rtol : float, default: 1e-12
        Relative tolerance for comparing affine transform parameters
        during alignment and concatenation operations. This small default
        handles typical floating-point representation noise.
    transform_atol : float, default: 0.0
        Absolute tolerance for comparing affine transform parameters.
    period_rtol : float, default: 0.0
        Relative tolerance for checking that ``period / |dx|`` is an integer number of cells.
    period_atol : float, default: 1e-3
        Absolute tolerance, in cells, for checking that ``period / |dx|`` is an integer.
        GeoTransform ``dx`` often has ~1e-12 relative round-off.

    Examples
    --------
    Use as a context manager:

    >>> import rasterix
    >>> import xarray as xr
    >>> with rasterix.set_options(transform_rtol=1e-9):
    ...     result = xr.concat([ds1, ds2], dim="x")
    """
    old = {}
    for k, v in kwargs.items():
        if k not in OPTIONS:
            raise ValueError(f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}")
        if k in _VALIDATORS and not _VALIDATORS[k](v):
            raise ValueError(f"option {k!r} given an invalid value: {v!r}. Expected a non-negative number.")
        old[k] = OPTIONS[k]
    OPTIONS.update(kwargs)
    try:
        yield
    finally:
        OPTIONS.update(old)


def get_options() -> dict[str, Any]:
    """
    Get current options for rasterix.

    Returns
    -------
    dict
        Dictionary of current option values.

    See Also
    --------
    set_options
    """
    return OPTIONS.copy()
