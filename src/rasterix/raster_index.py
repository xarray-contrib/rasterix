from __future__ import annotations

import math
import textwrap
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Self, TypeVar

import numpy as np
import pandas as pd
from affine import Affine
from xarray import Coordinates, DataArray, Dataset, Index, Variable
from xarray.core.coordinate_transform import CoordinateTransform

# TODO: import from public API once it is available
from xarray.core.indexes import CoordinateTransformIndex, PandasIndex
from xarray.core.indexing import IndexSelResult, merge_sel_results

from rasterix.odc_compat import BoundingBox, bbox_intersection, bbox_union, maybe_int, snap_grid
from rasterix.rioxarray_compat import guess_dims

T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")

__all__ = ["assign_index", "RasterIndex"]


def assign_index(obj: T_Xarray, *, x_dim: str | None = None, y_dim: str | None = None) -> T_Xarray:
    """Assign a RasterIndex to an Xarray DataArray or Dataset.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The object to assign the index to. Must have a rio accessor with a transform.
    x_dim : str, optional
        Name of the x dimension. If None, will be automatically detected.
    y_dim : str, optional
        Name of the y dimension. If None, will be automatically detected.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The input object with RasterIndex coordinates assigned.

    Examples
    --------
    >>> import xarray as xr
    >>> import rioxarray  # Required for rio accessor
    >>> da = xr.open_dataset("path/to/raster.tif", engine="rasterio")
    >>> indexed_da = assign_index(da)
    """
    import rioxarray  # noqa

    if x_dim is None or y_dim is None:
        guessed_x, guessed_y = guess_dims(obj)
    x_dim = x_dim or guessed_x
    y_dim = y_dim or guessed_y

    index = RasterIndex.from_transform(
        obj.rio.transform(), obj.sizes[x_dim], obj.sizes[y_dim], x_dim=x_dim, y_dim=y_dim
    )
    coords = Coordinates.from_xindex(index)
    return obj.assign_coords(coords)


def _assert_transforms_are_compatible(*affines) -> None:
    A1 = affines[0]
    for index, A2 in enumerate(affines[1:]):
        if A1.a != A2.a or A1.b != A2.b or A1.d != A2.d or A1.e != A2.e:
            raise ValueError(
                f"Transform parameters are not compatible for affine 0: {A1}, and affine {index + 1} {A2}"
            )


class AffineTransform(CoordinateTransform):
    """Affine 2D transform wrapper."""

    affine: Affine
    xy_dims: tuple[str, str]

    def __init__(
        self,
        affine: Affine,
        width: int,
        height: int,
        x_coord_name: Hashable = "xc",
        y_coord_name: Hashable = "yc",
        x_dim: str = "x",
        y_dim: str = "y",
        dtype: Any = np.dtype(np.float64),
    ):
        super().__init__((x_coord_name, y_coord_name), {x_dim: width, y_dim: height}, dtype=dtype)
        self.affine = affine

        # array dimensions in reverse order (y = rows, x = cols)
        self.xy_dims = self.dims[0], self.dims[1]
        self.dims = self.dims[1], self.dims[0]

    def forward(self, dim_positions):
        positions = tuple(dim_positions[dim] for dim in self.xy_dims)
        x_labels, y_labels = self.affine * positions

        results = {}
        for name, labels in zip(self.coord_names, [x_labels, y_labels]):
            results[name] = labels

        return results

    def reverse(self, coord_labels):
        labels = tuple(coord_labels[name] for name in self.coord_names)
        x_positions, y_positions = ~self.affine * labels

        results = {}
        for dim, positions in zip(self.xy_dims, [x_positions, y_positions]):
            results[dim] = positions

        return results

    def equals(self, other, *, exclude):
        if exclude is not None:
            raise NotImplementedError
        if not isinstance(other, AffineTransform):
            return False
        return self.affine == other.affine and self.dim_size == other.dim_size


class AxisAffineTransform(CoordinateTransform):
    """Axis-independent wrapper of an affine 2D transform with no skew/rotation."""

    affine: Affine
    is_xaxis: bool
    coord_name: Hashable
    dim: str
    size: int

    def __init__(
        self,
        affine: Affine,
        size: int,
        coord_name: Hashable,
        dim: str,
        is_xaxis: bool,
        dtype: Any = np.dtype(np.float64),
    ):
        assert affine.is_rectilinear and (affine.b == affine.d == 0)

        super().__init__((coord_name,), {dim: size}, dtype=dtype)
        self.affine = affine
        self.is_xaxis = is_xaxis
        self.coord_name = coord_name
        self.dim = dim
        self.size = size

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        positions = np.asarray(dim_positions[self.dim])

        if self.is_xaxis:
            labels, _ = self.affine * (positions, np.zeros_like(positions))
        else:
            _, labels = self.affine * (np.zeros_like(positions), positions)

        return {self.coord_name: labels}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        labels = np.asarray(coord_labels[self.coord_name])

        if self.is_xaxis:
            positions, _ = ~self.affine * (labels, np.zeros_like(labels))
        else:
            _, positions = ~self.affine * (np.zeros_like(labels), labels)

        return {self.dim: positions}

    def equals(self, other, *, exclude):
        if not isinstance(other, AxisAffineTransform):
            return False
        if exclude is not None:
            raise NotImplementedError

        # only compare the affine parameters of the relevant axis
        if self.is_xaxis:
            affine_match = self.affine.a == other.affine.a and self.affine.c == other.affine.c
        else:
            affine_match = self.affine.e == other.affine.e and self.affine.f == other.affine.f

        return affine_match and self.size == other.size

    def generate_coords(self, dims: tuple[str, ...] | None = None) -> dict[Hashable, Any]:
        assert dims is None or dims == self.dims
        return self.forward({self.dim: np.arange(self.size)})

    def slice(self, slice: slice) -> AxisAffineTransform:
        start = max(slice.start or 0, 0)
        stop = min(slice.stop or self.size, self.size)
        step = slice.step or 1

        # TODO: support reverse transform (i.e., start > stop)?
        assert start < stop

        size = (stop - start) // step
        scale = float(step)

        if self.is_xaxis:
            affine = self.affine * Affine.translation(start, 0.0) * Affine.scale(scale, 1.0)
        else:
            affine = self.affine * Affine.translation(0.0, start) * Affine.scale(1.0, scale)

        return type(self)(
            affine,
            size,
            self.coord_name,
            self.dim,
            is_xaxis=self.is_xaxis,
            dtype=self.dtype,
        )


class AxisAffineTransformIndex(CoordinateTransformIndex):
    """Axis-independent Xarray Index for an affine 2D transform with no
    skew/rotation.

    For internal use only.

    This Index class provides specific behavior on top of
    Xarray's `CoordinateTransformIndex`:

    - Data slicing computes a new affine transform and returns a new
      `AxisAffineTransformIndex` object

    - Otherwise data selection creates and returns a new Xarray
      `PandasIndex` object for non-scalar indexers

    - The index can be converted to a `pandas.Index` object (useful for Xarray
      operations that don't work with Xarray indexes yet).

    """

    axis_transform: AxisAffineTransform
    dim: str

    def __init__(self, transform: AxisAffineTransform):
        assert isinstance(transform, AxisAffineTransform)
        super().__init__(transform)
        self.axis_transform = transform
        self.dim = transform.dim

    def isel(  # type: ignore[override]
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> AxisAffineTransformIndex | PandasIndex | None:
        idxer = indexers[self.dim]

        # generate a new index with updated transform if a slice is given
        if isinstance(idxer, slice):
            return AxisAffineTransformIndex(self.axis_transform.slice(idxer))
        # no index for vectorized (fancy) indexing with n-dimensional Variable
        elif isinstance(idxer, Variable) and idxer.ndim > 1:
            return None
        # no index for scalar value
        elif np.ndim(idxer) == 0:
            return None
        # otherwise return a PandasIndex with values computed by forward transformation
        else:
            values = self.axis_transform.forward({self.dim: idxer})[self.axis_transform.coord_name]
            if isinstance(idxer, Variable):
                new_dim = idxer.dims[0]
            else:
                new_dim = self.dim
            return PandasIndex(values, new_dim, coord_dtype=values.dtype)

    def sel(self, labels, method=None, tolerance=None):
        coord_name = self.axis_transform.coord_name
        label = labels[coord_name]

        if isinstance(label, slice):
            if label.start is None:
                label = slice(0, label.stop, label.step)
            if label.step is None:
                # continuous interval slice indexing (preserves the index)
                pos = self.transform.reverse({coord_name: np.array([label.start, label.stop])})
                # np.round rounds to even, this way we round upwards
                pos = np.floor(pos[self.dim] + 0.5).astype("int")
                new_start = max(pos[0], 0)
                new_stop = min(pos[1], self.axis_transform.size)
                return IndexSelResult({self.dim: slice(new_start, new_stop)})
            else:
                # otherwise convert to basic (array) indexing
                label = np.arange(label.start, label.stop, label.step)

        # support basic indexing (in the 1D case basic vs. vectorized indexing
        # are pretty much similar)
        unwrap_xr = False
        if not isinstance(label, Variable | DataArray):
            # basic indexing -> either scalar or 1-d array
            try:
                var = Variable("_", label)
            except ValueError:
                var = Variable((), label)
            labels = {self.dim: var}
            unwrap_xr = True

        result = super().sel(labels, method=method, tolerance=tolerance)

        if unwrap_xr:
            dim_indexers = {self.dim: result.dim_indexers[self.dim].values}
            result = IndexSelResult(dim_indexers)

        return result

    def to_pandas_index(self) -> pd.Index:
        import pandas as pd

        values = self.transform.generate_coords()
        return pd.Index(values[self.dim])


# The types of Xarray indexes that may be wrapped by RasterIndex
WrappedIndex = AxisAffineTransformIndex | PandasIndex | CoordinateTransformIndex
WrappedIndexCoords = Hashable | tuple[Hashable, Hashable]


def _filter_dim_indexers(index: WrappedIndex, indexers: Mapping) -> Mapping:
    if isinstance(index, CoordinateTransformIndex):
        dims = index.transform.dims
    else:
        # PandasIndex
        dims = (str(index.dim),)

    return {dim: indexers[dim] for dim in dims if dim in indexers}


class RasterIndex(Index):
    """Xarray index for raster coordinate indexing and spatial operations.

    RasterIndex provides spatial indexing capabilities for raster data by wrapping
    one or more Xarray indexes that handle coordinate transformations. It supports
    both rectilinear and non-rectilinear (rotated/skewed) raster grids.

    The internal structure depends on the affine transformation:

    - **Non-rectilinear or rotated grids**: Uses a single 2D CoordinateTransformIndex
      for coupled x/y coordinates that handles rotation and skew.

    - **Rectilinear grids**: Uses separate 1D indexes for independent x/y axes,
      enabling more efficient slicing operations.

    Parameters
    ----------
    indexes : Mapping[WrappedIndexCoords, WrappedIndex]
        Dictionary mapping coordinate names to their corresponding index objects.
        Keys are either single coordinate names or tuples for coupled coordinates.

    Attributes
    ----------
    bbox : BoundingBox
        Spatial bounding box of the raster index.

    Methods
    -------
    from_transform
        Create RasterIndex from affine transform and dimensions.
    transform
        Get affine transform for pixel top-left corners.
    center_transform
        Get affine transform for pixel centers.

    Examples
    --------
    Create a RasterIndex from an affine transform:

    >>> from affine import Affine
    >>> transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    >>> index = RasterIndex.from_transform(transform, width=100, height=100)
    >>> print(index.bbox)
    BoundingBox(left=0.0, bottom=0.0, right=100.0, top=100.0)

    Notes
    -----
    For rectilinear grids without rotation, RasterIndex creates separate 1D indexes
    for x and y coordinates, which enables efficient slicing operations. For grids
    with rotation or skew, it uses a coupled 2D transform.
    """

    _wrapped_indexes: dict[WrappedIndexCoords, WrappedIndex]
    _shape: dict[str, int]

    def __init__(self, indexes: Mapping[WrappedIndexCoords, WrappedIndex]):
        idx_keys = list(indexes)
        idx_vals = list(indexes.values())

        # either one or the other configuration (dependent vs. independent x/y axes)
        axis_dependent = (
            len(indexes) == 1
            and isinstance(idx_keys[0], tuple)
            and isinstance(idx_vals[0], CoordinateTransformIndex)
        )
        axis_independent = len(indexes) in (1, 2) and all(
            isinstance(idx, AxisAffineTransformIndex | PandasIndex) for idx in idx_vals
        )
        assert axis_dependent ^ axis_independent

        self._wrapped_indexes = dict(indexes)
        if axis_independent:
            self._shape = {
                "x": self._wrapped_indexes["x"].axis_transform.size,
                "y": self._wrapped_indexes["y"].axis_transform.size,
            }
        else:
            self._shape = next(iter(self._wrapped_indexes.values())).dim_size

    @classmethod
    def from_transform(
        cls, affine: Affine, width: int, height: int, x_dim: str = "x", y_dim: str = "y"
    ) -> RasterIndex:
        """Create a RasterIndex from an affine transform and raster dimensions.

        Parameters
        ----------
        affine : affine.Affine
            Affine transformation matrix defining the mapping from pixel coordinates
            to spatial coordinates. Should represent pixel top-left corners.
        width : int
            Number of pixels in the x direction.
        height : int
            Number of pixels in the y direction.
        x_dim : str, default "x"
            Name for the x dimension.
        y_dim : str, default "y"
            Name for the y dimension.

        Returns
        -------
        RasterIndex
            A new RasterIndex object with appropriate internal structure.

        Notes
        -----
        For rectilinear transforms (no rotation/skew), separate AxisAffineTransformIndex
        objects are created for x and y coordinates. For non-rectilinear transforms,
        a single coupled CoordinateTransformIndex is used.

        Examples
        --------
        Create a simple rectilinear index:

        >>> from affine import Affine
        >>> transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
        >>> index = RasterIndex.from_transform(transform, width=100, height=100)
        """
        indexes: dict[WrappedIndexCoords, AxisAffineTransformIndex | CoordinateTransformIndex]

        # pixel centered coordinates
        affine = affine * Affine.translation(0.5, 0.5)

        if affine.is_rectilinear and affine.b == affine.d == 0:
            x_transform = AxisAffineTransform(affine, width, "x", x_dim, is_xaxis=True)
            y_transform = AxisAffineTransform(affine, height, "y", y_dim, is_xaxis=False)
            indexes = {
                "x": AxisAffineTransformIndex(x_transform),
                "y": AxisAffineTransformIndex(y_transform),
            }
        else:
            xy_transform = AffineTransform(affine, width, height, x_dim=x_dim, y_dim=y_dim)
            indexes = {("x", "y"): CoordinateTransformIndex(xy_transform)}

        return cls(indexes)

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> RasterIndex:
        # TODO: compute bounds, resolution and affine transform from explicit coordinates.
        raise NotImplementedError("Creating a RasterIndex from existing coordinates is not yet supported.")

    def create_variables(self, variables: Mapping[Any, Variable] | None = None) -> dict[Hashable, Variable]:
        new_variables: dict[Hashable, Variable] = {}

        for index in self._wrapped_indexes.values():
            new_variables.update(index.create_variables())

        return new_variables

    def isel(self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]) -> RasterIndex | None:
        new_indexes: dict[WrappedIndexCoords, WrappedIndex] = {}

        for coord_names, index in self._wrapped_indexes.items():
            index_indexers = _filter_dim_indexers(index, indexers)
            if not index_indexers:
                # no selection to perform: simply propagate the index
                # TODO: uncomment when https://github.com/pydata/xarray/issues/10063 is fixed
                # new_indexes[coord_names] = index
                ...
            else:
                new_index = index.isel(index_indexers)
                if new_index is not None:
                    new_indexes[coord_names] = new_index

        if new_indexes:
            # TODO: if there's only a single PandasIndex can we just return it?
            # (maybe better to keep it wrapped if we plan to later make RasterIndex CRS-aware)
            return RasterIndex(new_indexes)
        else:
            return None

    def sel(self, labels: dict[Any, Any], method=None, tolerance=None) -> IndexSelResult:
        results = []

        for coord_names, index in self._wrapped_indexes.items():
            if not isinstance(coord_names, tuple):
                coord_names = (coord_names,)
            index_labels = {k: v for k, v in labels.items() if k in coord_names}
            if index_labels:
                results.append(index.sel(index_labels, method=method, tolerance=tolerance))

        return merge_sel_results(results)

    def equals(self, other: Index, *, exclude=None) -> bool:
        if exclude is None:
            exclude = {}
        if not isinstance(other, RasterIndex):
            return False
        if set(self._wrapped_indexes) != set(other._wrapped_indexes):
            return False

        return all(
            index.equals(other._wrapped_indexes[k])  # type: ignore[arg-type]
            for k, index in self._wrapped_indexes.items()
            if k not in exclude
        )

    def to_pandas_index(self) -> pd.Index:
        # conversion is possible only if this raster index encapsulates
        # exactly one AxisAffineTransformIndex or a PandasIndex associated
        # to either the x or y axis (1-dimensional) coordinate.
        if len(self._wrapped_indexes) == 1:
            index = next(iter(self._wrapped_indexes.values()))
            if isinstance(index, AxisAffineTransformIndex | PandasIndex):
                return index.to_pandas_index()

        raise ValueError("Cannot convert RasterIndex to pandas.Index")

    def __repr__(self):
        items: list[str] = []

        for coord_names, index in self._wrapped_indexes.items():
            items += [repr(coord_names) + ":", textwrap.indent(repr(index), "    ")]

        return "RasterIndex\n" + "\n".join(items)

    def transform(self) -> Affine:
        """Affine transform for top-left corners."""
        return self.center_transform() * Affine.translation(-0.5, -0.5)

    def center_transform(self) -> Affine:
        """Affine transform for cell centers."""
        if len(self._wrapped_indexes) > 1:
            x = self._wrapped_indexes["x"].axis_transform.affine
            y = self._wrapped_indexes["y"].axis_transform.affine
            aff = Affine(x.a, x.b, x.c, y.d, y.e, y.f)
        else:
            index = next(iter(self._wrapped_indexes.values()))
            aff = index.affine
        return aff

    @property
    def bbox(self) -> BoundingBox:
        """Bounding Box for index.

        Returns
        -------
        BoundingBox
        """
        return BoundingBox.from_transform(
            shape=tuple(self._shape[k] for k in ("y", "x")),
            transform=self.transform(),
        )

    @classmethod
    def concat(
        cls,
        indexes: Sequence[Self],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> Self:
        if len(indexes) == 1:
            return next(iter(indexes))

        if positions is not None:
            raise NotImplementedError

        # Note: I am assuming that xarray has calling `align(..., exclude="x" [or 'y'])` already
        # and checked for equality along "y" [or 'x']
        new_bbox = bbox_union(as_compatible_bboxes(*indexes, concat_dim=dim))
        return indexes[0]._new_with_bbox(new_bbox)

    def _new_with_bbox(self, bbox: BoundingBox) -> Self:
        affine = self.transform()
        new_affine, Nx, Ny = bbox_to_affine(bbox, rx=affine.a, ry=affine.e)
        # TODO: set xdim, ydim explicitly
        new_index = self.from_transform(new_affine, width=Nx, height=Ny)
        assert new_index.bbox == bbox
        return new_index

    def join(self, other: RasterIndex, how) -> RasterIndex:
        if set(self._wrapped_indexes.keys()) != set(other._wrapped_indexes.keys()):
            # TODO: better error message
            raise ValueError(
                "Alignment is only supported between RasterIndexes, when both contain compatible transforms."
            )

        ours, theirs = as_compatible_bboxes(self, other, concat_dim=None)
        if how == "outer":
            new_bbox = ours | theirs
        elif how == "inner":
            new_bbox = ours & theirs
        else:
            raise NotImplementedError(f"{how=!r} not implemented yet for RasterIndex.")

        return self._new_with_bbox(new_bbox)

    def reindex_like(self, other: Self, method=None, tolerance=None) -> dict[Hashable, Any]:
        affine = self.transform()
        ours, theirs = as_compatible_bboxes(self, other, concat_dim=None)
        inter = bbox_intersection([ours, theirs])
        dx = affine.a
        dy = affine.e

        # Fraction of a pixel that can be ignored, defaults to 1/100. Bounding box of the output
        # geobox is allowed to be smaller than supplied bounding box by that amount.
        # FIXME: translate user-provided `tolerance` to `tol`
        tol: float = 0.01

        indexers = {}
        indexers["x"] = get_indexer(
            theirs.left, ours.left, inter.left, inter.right, spacing=dx, tol=tol, size=other._shape["x"]
        )
        indexers["y"] = get_indexer(
            theirs.top, ours.top, inter.top, inter.bottom, spacing=dy, tol=tol, size=other._shape["y"]
        )
        return indexers


def get_indexer(off, our_off, start, stop, spacing, tol, size) -> np.ndarray:
    istart = math.ceil(maybe_int((start - off) / spacing, tol))

    ours_istart = math.ceil(maybe_int((start - our_off) / spacing, tol))
    ours_istop = math.ceil(maybe_int((stop - our_off) / spacing, tol))

    idxr = np.concatenate(
        [
            np.full((istart,), fill_value=-1),
            np.arange(ours_istart, ours_istop),
            np.full((size - istart - (ours_istop - ours_istart),), fill_value=-1),
        ]
    )
    return idxr


def bbox_to_affine(bbox: BoundingBox, rx, ry) -> Affine:
    # Fraction of a pixel that can be ignored, defaults to 1/100. Bounding box of the output
    # geobox is allowed to be smaller than supplied bounding box by that amount.
    # FIXME: translate user-provided `tolerance` to `tol`
    tol: float = 0.01

    offx, nx = snap_grid(bbox.left, bbox.right, rx, 0, tol=tol)
    offy, ny = snap_grid(bbox.bottom, bbox.top, ry, 0, tol=tol)

    affine = Affine.translation(offx, offy) * Affine.scale(rx, ry)

    return affine, nx, ny


def as_compatible_bboxes(*indexes: RasterIndex, concat_dim: Hashable | None) -> tuple[BoundingBox, ...]:
    transforms = tuple(i.transform() for i in indexes)
    _assert_transforms_are_compatible(*transforms)

    expected_off_x = (transforms[0].c,) + tuple(
        t.c + i._shape["x"] * t.a for i, t in zip(indexes[:-1], transforms[:-1])
    )
    expected_off_y = (transforms[0].f,) + tuple(
        t.f + i._shape["y"] * t.e for i, t in zip(indexes[:-1], transforms[:-1])
    )

    off_x = tuple(t.c for t in transforms)
    off_y = tuple(t.f for t in transforms)

    if concat_dim is not None:
        if all(o == off_x[0] for o in off_x[1:]) and all(o == off_y[0] for o in off_y[1:]):
            raise ValueError("Attempting to concatenate arrays with same transform along X or Y.")

    if concat_dim == "x":
        if any(off_y[0] != o for o in off_y[1:]):
            raise ValueError("offsets must be identical in X when concatenating along Y")
        if any(a != b for a, b in zip(off_x, expected_off_x)):
            raise ValueError(
                f"X offsets are incompatible. Provided offsets {off_x}, expected offsets: {expected_off_x}"
            )
    elif concat_dim == "y":
        if any(off_x[0] != o for o in off_x[1:]):
            raise ValueError("offsets must be identical in X when concatenating along Y")

        if any(a != b for a, b in zip(off_y, expected_off_y)):
            raise ValueError(
                f"Y offsets are incompatible. Provided offsets {off_y}, expected offsets: {expected_off_y}"
            )

    return tuple(i.bbox for i in indexes)
