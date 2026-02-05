"""Tests for rasterix options and tolerance support."""

import pytest

from rasterix import get_options, set_options
from rasterix.odc_compat import BoundingBox


def test_set_options():
    """Test set_options/get_options and context manager."""
    # Check defaults
    opts = get_options()
    assert opts["transform_rtol"] == 1e-12
    assert opts["transform_atol"] == 0.0

    # Context manager scoping
    with set_options(transform_rtol=1e-6):
        assert get_options()["transform_rtol"] == 1e-6
    assert get_options()["transform_rtol"] == 1e-12

    # Validation
    with pytest.raises(ValueError, match="not in the set of valid options"):
        with set_options(invalid_option=1.0):
            pass
    with pytest.raises(ValueError, match="non-negative"):
        with set_options(transform_rtol=-1e-6):
            pass


def test_boundingbox_isclose():
    """Test BoundingBox.isclose() method."""
    bbox1 = BoundingBox(0.0, 0.0, 10.0, 10.0)

    # Equal and nearly-equal boxes
    assert bbox1.isclose(BoundingBox(0.0, 0.0, 10.0, 10.0))
    assert bbox1.isclose(BoundingBox(0.0, 0.0, 10.0 + 1e-14, 10.0))

    # Different boxes
    assert not bbox1.isclose(BoundingBox(0.0, 0.0, 10.1, 10.0))

    # Custom tolerance
    assert bbox1.isclose(BoundingBox(0.0, 0.0, 10.001, 10.0), rtol=1e-3)

    # Non-BoundingBox returns False
    assert not bbox1.isclose((0.0, 0.0, 10.0, 10.0))
