import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Any

from affine import Affine


def _snap_edge_pos(x0: float, x1: float, res: float, tol: float) -> tuple[float, int]:
    assert res > 0
    assert x1 >= x0
    _x0 = math.floor(maybe_int(x0 / res, tol))
    _x1 = math.ceil(maybe_int(x1 / res, tol))
    nx = max(1, _x1 - _x0)
    return _x0 * res, nx


def _snap_edge(x0: float, x1: float, res: float, tol: float) -> tuple[float, int]:
    assert x1 >= x0
    if res > 0:
        return _snap_edge_pos(x0, x1, res, tol)
    _tx, nx = _snap_edge_pos(x0, x1, -res, tol)
    tx = _tx + nx * (-res)
    return tx, nx


def snap_grid(
    x0: float, x1: float, res: float, off_pix: float | None = 0, tol: float = 1e-6
) -> tuple[float, int]:
    """
    Compute grid snapping for single axis.

    :param x0: In point ``x0 <= x1``
    :param x1: Out point ``x0 <= x1``
    :param res: Pixel size and direction (can be negative)
    :param off_pix:
       Pixel fraction to align to ``x=0``.
       0 - edge aligned
       0.5 - center aligned
       None - don't snap

    :return: ``tx, nx`` that defines 1-d grid, such that ``x0`` and ``x1`` are within edge pixels.
    """
    assert (off_pix is None) or (0 <= off_pix < 1)
    if off_pix is None:
        if res > 0:
            nx = math.ceil(maybe_int((x1 - x0) / res, tol))
            return x0, max(1, nx)
        nx = math.ceil(maybe_int((x1 - x0) / (-res), tol))
        return x1, max(nx, 1)

    off = off_pix * abs(res)
    _tx, nx = _snap_edge(x0 - off, x1 - off, res, tol)
    return _tx + off, nx


def split_float(x: float) -> tuple[float, float]:
    """
    Split float number into whole and fractional parts.

    Adding the two numbers back together should result in the original value.
    Fractional part is always in the ``(-0.5, +0.5)`` interval, and whole part
    is equivalent to ``round(x)``.

    :param x: floating point number
    :return: ``whole, fraction``
    """
    if not math.isfinite(x):
        return (x, 0)

    x_part = math.fmod(x, 1.0)
    x_whole = x - x_part
    if x_part > 0.5:
        x_part -= 1
        x_whole += 1
    elif x_part < -0.5:
        x_part += 1
        x_whole -= 1
    return (x_whole, x_part)


def maybe_int(x: float, tol: float) -> int | float:
    """
    Turn almost ints to actual ints.

    pass through other values unmodified.
    """
    if not math.isfinite(x):
        return x

    x_whole, x_part = split_float(x)

    if abs(x_part) < tol:  # almost int
        return int(x_whole)
    return x


class BoundingBox(Sequence[float]):
    """Bounding box, defining extent in cartesian coordinates."""

    __slots__ = "_box"

    def __init__(self, left: float, bottom: float, right: float, top: float):
        self._box = (left, bottom, right, top)

    @property
    def left(self):
        return self._box[0]

    @property
    def bottom(self):
        return self._box[1]

    @property
    def right(self):
        return self._box[2]

    @property
    def top(self):
        return self._box[3]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._box

    def __iter__(self):
        return self._box.__iter__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BoundingBox):
            return self._box == other._box
        return self._box == other

    def __hash__(self) -> int:
        return hash(self._box)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx):
        return self._box[idx]

    def __repr__(self) -> str:
        return f"BoundingBox(left={self.left}, bottom={self.bottom}, right={self.right}, top={self.top})"

    def __str__(self) -> str:
        return self.__repr__()

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
        return bbox_intersection([self, other])

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        return bbox_union([self, other])

    @property
    def shape(self) -> tuple[int, int]:
        """``(int(span_y), int(span_x))``."""
        return (self.height, self.width)

    @property
    def points(self) -> list[tuple[float, float]]:
        """Extract four corners of the bounding box."""
        x0, y0, x1, y1 = self._box
        return list(itertools.product((x0, x1), (y0, y1)))

    def transform(self, transform: Affine) -> "BoundingBox":
        """
        Map bounding box through a linear transform.

        Apply linear transform on four points of the bounding box and compute bounding box of these
        four points.
        """
        pts = [transform * pt for pt in self.points]
        xx = [x for x, _ in pts]
        yy = [y for _, y in pts]
        return BoundingBox(min(xx), min(yy), max(xx), max(yy))

    @staticmethod
    def from_xy(x: tuple[float, float], y: tuple[float, float]) -> "BoundingBox":
        """
        Construct :py:class:`BoundingBox` from x and y ranges.

        :param x: (left, right)
        :param y: (bottom, top)
        """
        x1, x2 = sorted(x)
        y1, y2 = sorted(y)
        return BoundingBox(x1, y1, x2, y2)

    @staticmethod
    def from_points(p1: tuple[float, float], p2: tuple[float, float]) -> "BoundingBox":
        """
        Construct :py:class:`BoundingBox` from two points.

        :param p1: (x, y)
        :param p2: (x, y)
        """
        return BoundingBox.from_xy((p1[0], p2[0]), (p1[1], p2[1]))

    @staticmethod
    def from_transform(shape: tuple[int, int], transform: Affine) -> "BoundingBox":
        """
        Construct :py:class:`BoundingBox` from image shape and transform.

        :param shape: image dimensions
        :param transform: Affine mapping from pixel to world
        """
        ny, nx = shape
        pts = [(0, 0), (nx, 0), (nx, ny), (0, ny)]
        transform.itransform(pts)
        xx, yy = list(zip(*pts))
        return BoundingBox.from_xy((min(xx), max(xx)), (min(yy), max(yy)))

    def round(self) -> "BoundingBox":
        """
        Expand bounding box to nearest integer on all sides.
        """
        x0, y0, x1, y1 = self._box
        return BoundingBox(math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))


def bbox_union(bbs: Iterable[BoundingBox]) -> BoundingBox:
    """
    Compute union of bounding boxes.

    Given a stream of bounding boxes compute enclosing :py:class:`~odc.geo.geom.BoundingBox`.
    """
    # pylint: disable=invalid-name
    try:
        bb, *bbs = bbs
    except ValueError:
        raise ValueError("Union of empty stream is undefined") from None

    L, B, R, T = bb

    for bb in bbs:
        l, b, r, t = bb  # noqa: E741
        L = min(l, L)
        B = min(b, B)
        R = max(r, R)
        T = max(t, T)

    return BoundingBox(L, B, R, T)


def bbox_intersection(bbs: Iterable[BoundingBox]) -> BoundingBox:
    """
    Compute intersection of bounding boxes.

    Given a stream of bounding boxes compute the overlap :py:class:`~odc.geo.geom.BoundingBox`.
    """
    # pylint: disable=invalid-name
    try:
        bb, *bbs = bbs
    except ValueError:
        raise ValueError("Intersection of empty stream is undefined") from None

    L, B, R, T = bb

    for bb in bbs:
        l, b, r, t = bb  # noqa: E741
        L = max(l, L)
        B = max(b, B)
        R = min(r, R)
        T = min(t, T)

    return BoundingBox(L, B, R, T)
