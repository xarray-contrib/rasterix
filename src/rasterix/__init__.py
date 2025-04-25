from .raster_index import RasterIndex, assign_index


def _get_version():
    __version__ = "999"
    try:
        from ._version import __version__
    except ImportError:
        pass
    return __version__


__version__ = _get_version()

__all__ = ["RasterIndex", "assign_index"]
