import pytest


def _engine_available(engine: str) -> bool:
    """Check if a rasterization engine is available."""
    if engine == "rasterio":
        try:
            import rasterio  # noqa: F401

            return True
        except ImportError:
            return False
    elif engine == "rusterize":
        try:
            import rusterize  # noqa: F401

            return True
        except ImportError:
            return False
    elif engine == "exactextract":
        try:
            import exactextract  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests that use the 'engine' fixture."""
    if "engine" in metafunc.fixturenames:
        engines = []
        # Only add engines that are available
        if _engine_available("rasterio"):
            engines.append("rasterio")
        if _engine_available("rusterize"):
            engines.append("rusterize")
        if _engine_available("exactextract"):
            engines.append("exactextract")

        if not engines:
            pytest.skip("No rasterization engine available (need rasterio, rusterize, or exactextract)")

        metafunc.parametrize("engine", engines)
