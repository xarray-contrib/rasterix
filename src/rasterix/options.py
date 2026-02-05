"""Options for rasterix with context manager support."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

OPTIONS: dict[str, Any] = {
    "transform_rtol": 1e-12,
    "transform_atol": 0.0,
}


def _validate_tolerance(value: Any) -> bool:
    """Validate that value is a non-negative float."""
    return isinstance(value, int | float) and value >= 0


_VALIDATORS = {
    "transform_rtol": _validate_tolerance,
    "transform_atol": _validate_tolerance,
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
