import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Any

from affine import Affine


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

    def buffered(self, xbuff: float, ybuff: float | None = None) -> "BoundingBox":
        """
        Return a new BoundingBox, buffered in the x and y dimensions.

        :param xbuff: X dimension buffering amount
        :param ybuff: Y dimension buffering amount
        :return: new BoundingBox
        """
        if ybuff is None:
            ybuff = xbuff

        return BoundingBox(
            left=self.left - xbuff,
            bottom=self.bottom - ybuff,
            right=self.right + xbuff,
            top=self.top + ybuff,
        )

    @property
    def span_x(self) -> float:
        """Span of the bounding box along x axis."""
        return self.right - self.left

    @property
    def span_y(self) -> float:
        """Span of the bounding box along y axis."""
        return self.top - self.bottom

    @property
    def aspect(self) -> float:
        """Aspect ratio."""
        return self.span_x / self.span_y

    @property
    def width(self) -> int:
        """``int(span_x)``"""
        return int(self.right - self.left)

    @property
    def height(self) -> int:
        """``int(span_y)``"""
        return int(self.top - self.bottom)

    @property
    def shape(self) -> tuple[int, int]:
        """``(int(span_y), int(span_x))``."""
        return (self.height, self.width)

    @property
    def range_x(self) -> tuple[float, float]:
        """``left, right``"""
        return (self.left, self.right)

    @property
    def range_y(self) -> tuple[float, float]:
        """``bottom, top``"""
        return (self.bottom, self.top)

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

    def map_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Convert to bounds in folium/ipyleaflet style.

        Returns SW, and NE corners in lat/lon order.
        ``((lat_w, lon_s), (lat_e, lon_n))``.
        """
        x0, y0, x1, y1 = self._box
        return (y0, x0), (y1, x1)

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
