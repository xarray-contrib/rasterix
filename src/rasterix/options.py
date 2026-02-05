"""Options for rasterix with context manager support."""

from __future__ import annotations

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


class set_options:
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

    Or set global options:

    >>> rasterix.set_options(transform_rtol=1e-9)  # doctest: +ELLIPSIS
    <rasterix.options.set_options object at 0x...>
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}")
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(
                    f"option {k!r} given an invalid value: {v!r}. Expected a non-negative number."
                )
            self.old[k] = OPTIONS[k]
        OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        OPTIONS.update(self.old)


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
