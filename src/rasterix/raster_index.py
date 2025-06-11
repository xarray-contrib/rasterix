from __future__ import annotations

import textwrap
from collections.abc import Hashable, Mapping
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import xproj
from affine import Affine
from pyproj import CRS
from xarray import Coordinates, DataArray, Dataset, Index, Variable, get_options
from xarray.core.coordinate_transform import CoordinateTransform

# TODO: import from public API once it is available
from xarray.core.indexes import CoordinateTransformIndex, PandasIndex
from xarray.core.indexing import IndexSelResult, merge_sel_results
from xproj.typing import CRSAwareIndex

from rasterix.rioxarray_compat import guess_dims

T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")


def assign_index(obj: T_Xarray, *, x_dim: str | None = None, y_dim: str | None = None) -> T_Xarray:
    if x_dim is None or y_dim is None:
        guessed_x, guessed_y = guess_dims(obj)
    x_dim = x_dim or guessed_x
    y_dim = y_dim or guessed_y

    index = RasterIndex.from_transform(
        obj.rio.transform(), obj.sizes[x_dim], obj.sizes[y_dim], x_dim=x_dim, y_dim=y_dim
    )
    coords = Coordinates.from_xindex(index)
    return obj.assign_coords(coords)


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

    def equals(self, other: CoordinateTransform, exclude: frozenset[Hashable] | None = None) -> bool:
        if not isinstance(other, AffineTransform):
            return False
        return self.affine == other.affine and self.dim_size == other.dim_size

    def __repr__(self) -> str:
        params = ", ".join(f"{pn}={getattr(self.affine, pn):.4g}" for pn in "abcdef")
        return f"{type(self).__name__}({params})"


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

    def equals(self, other: CoordinateTransform, exclude: frozenset[Hashable] | None = None) -> bool:
        if not isinstance(other, AxisAffineTransform):
            return False

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

    def __repr__(self) -> str:
        params = ", ".join(f"{pn}={getattr(self.affine, pn):.4g}" for pn in "abcdef")
        return f"{type(self).__name__}({params}, axis={'X' if self.is_xaxis else 'Y'}, dim={self.dim!r})"


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


class RasterIndex(Index, xproj.ProjIndexMixin):
    """Xarray index for raster coordinates.

    RasterIndex is itself a wrapper around one or more Xarray indexes associated
    with either the raster x or y axis coordinate or both, depending on the
    affine transformation and prior data selection (if any):

    - The affine transformation is not rectilinear or has rotation: this index
      encapsulates a single `CoordinateTransformIndex` object for both the x and
      y axis (2-dimensional) coordinates.

    - The affine transformation is rectilinear and has no rotation: this index
      encapsulates one or two index objects for either the x or y axis or both
      (1-dimensional) coordinates. The index type is either a subclass of
      `CoordinateTransformIndex` that supports slicing or `PandasIndex` (e.g.,
      after data selection at arbitrary locations).

    RasterIndex is CRS-aware, i.e., it has a ``crs`` property that is used for
    checking equality or compatibility with other RasterIndex instances. CRS is
    optional.

    """

    _wrapped_indexes: dict[WrappedIndexCoords, WrappedIndex]
    _crs: CRS | None

    def __init__(self, indexes: Mapping[WrappedIndexCoords, WrappedIndex], crs: CRS | Any | None = None):
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

        if crs is not None:
            crs = CRS.from_user_input(crs)

        self._crs = crs

    @classmethod
    def from_transform(
        cls,
        affine: Affine,
        width: int,
        height: int,
        x_dim: str = "x",
        y_dim: str = "y",
        crs: CRS | Any | None = None,
    ) -> RasterIndex:
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

        return cls(indexes, crs=crs)

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

    @property
    def crs(self) -> CRS | None:
        """Returns the coordinate reference system (CRS) of the index as a
        :class:`pyproj.crs.CRS` object, or ``None`` if CRS is undefined.
        """
        return self._crs

    def _proj_set_crs(self: RasterIndex, spatial_ref: Hashable, crs: CRS) -> RasterIndex:
        # Returns a raster index shallow copy with a replaced CRS
        # (XProj integration via xproj.ProjIndexMixin)
        # Note: XProj already handles the case of overriding any existing CRS
        return RasterIndex(self._wrapped_indexes, crs=crs)

    def isel(self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]) -> RasterIndex | None:
        new_indexes: dict[WrappedIndexCoords, WrappedIndex] = {}

        for coord_names, index in self._wrapped_indexes.items():
            index_indexers = _filter_dim_indexers(index, indexers)
            if not index_indexers:
                new_indexes[coord_names] = index
            else:
                new_index = index.isel(index_indexers)
                if new_index is not None:
                    new_indexes[coord_names] = cast(WrappedIndex, new_index)

        if new_indexes:
            # TODO: if there's only a single PandasIndex can we just return it?
            # (maybe better to keep it wrapped if we plan to later make RasterIndex CRS-aware)
            return RasterIndex(new_indexes, crs=self.crs)
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

    def equals(self, other: Index) -> bool:
        if not isinstance(other, RasterIndex):
            return False
        if not self._proj_crs_equals(cast(CRSAwareIndex, other), allow_none=True):
            return False
        if set(self._wrapped_indexes) != set(other._wrapped_indexes):
            return False

        return all(
            index.equals(other._wrapped_indexes[k])  # type: ignore[arg-type]
            for k, index in self._wrapped_indexes.items()
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

    def _repr_inline_(self, max_width: int) -> str:
        # TODO: remove when fixed in Xarray (https://github.com/pydata/xarray/pull/10415)
        if max_width is None:
            max_width = get_options()["display_width"]

        srs = xproj.format_crs(self.crs, max_width=max_width)
        return f"{self.__class__.__name__} (crs={srs})"

    def __repr__(self) -> str:
        srs = xproj.format_crs(self.crs)
        items: list[str] = []

        for coord_names, index in self._wrapped_indexes.items():
            # TODO: remove when CoordinateTransformIndex.__repr__ is implemented in Xarray
            if isinstance(index, CoordinateTransformIndex):
                index_repr = f"{type(index).__name__}({index.transform!r})"
            else:
                index_repr = repr(index)
            items += [repr(coord_names) + ":", textwrap.indent(index_repr, "    ")]

        return f"RasterIndex(crs={srs})\n" + "\n".join(items)

    def transform(self) -> Affine:
        """Returns Affine transform for top-left corners."""
        if len(self._wrapped_indexes) > 1:
            x = self._wrapped_indexes["x"].axis_transform.affine
            y = self._wrapped_indexes["y"].axis_transform.affine
            aff = Affine(x.a, x.b, x.c, y.d, y.e, y.f)
        else:
            index = next(iter(self._wrapped_indexes.values()))
            aff = index.affine
        return aff * Affine.translation(-0.5, -0.5)
