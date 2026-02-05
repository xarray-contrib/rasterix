from __future__ import annotations

import math
import textwrap
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Self, TypeVar, cast

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
from xarray.core.types import JoinOptions
from xproj.typing import CRSAwareIndex

from rasterix.odc_compat import BoundingBox, bbox_intersection, bbox_union, maybe_int, snap_grid
from rasterix.options import get_options as get_rasterix_options
from rasterix.rioxarray_compat import guess_dims
from rasterix.utils import get_affine, get_crs_from_proj_zarr_convention

T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")

__all__ = ["assign_index", "RasterIndex"]


# X/Y axis order conventions used for public and internal attributes
XAXIS = 0
YAXIS = 1


def assign_index(
    obj: T_Xarray, *, x_dim: str | None = None, y_dim: str | None = None, crs: bool = True
) -> T_Xarray:
    """Assign a RasterIndex to an Xarray DataArray or Dataset.

    By default, the affine transform is guessed by first looking for a ``GeoTransform`` attribute
    on a CF "grid mapping" variable (commonly ``"spatial_ref"``). If not present, then the affine is determined from 1D coordinate
    variables named ``x_dim`` and ``y_dim`` provided to this function.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The object to assign the index to.
    x_dim : str, optional
        Name of the x dimension. If None, will be automatically detected.
    y_dim : str, optional
        Name of the y dimension. If None, will be automatically detected.
    crs: bool, optional
       Auto-detect CRS using xproj?

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The input object with RasterIndex coordinates assigned.

    Notes
    -----
    The "grid mapping" variable is determined following the CF conventions:

    - If a DataArray is provided, we look for an attribute named ``"grid_mapping"``.
    - For a Dataset, we pull the first detected ``"grid_mapping"`` attribute when iterating over data variables.

    The value of this attribute is a variable name containing projection information (commonly ``"spatial_ref"``).
    We then look for a ``"GeoTransform"`` attribute on this variable (following GDAL convention).

    References
    ----------
    - `CF conventions document <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#grid-mappings-and-projections>`_.
    - `GDAL docs on GeoTransform <https://gdal.org/en/stable/tutorials/geotransforms_tut.html>`_.

    Examples
    --------
    >>> import xarray as xr
    >>> import rioxarray  # Required for reading TIFF
    >>> da = xr.open_dataset("path/to/raster.tif", engine="rasterio")
    >>> indexed_da = assign_index(da)
    """
    if x_dim is None or y_dim is None:
        guessed_x, guessed_y = guess_dims(obj)
    x_dim = x_dim or guessed_x
    y_dim = y_dim or guessed_y

    affine = get_affine(obj, x_dim=x_dim, y_dim=y_dim, clear_transform=True)

    detected_crs = obj.proj.crs if crs else None
    if detected_crs is None:
        detected_crs = get_crs_from_proj_zarr_convention(obj)

    index = RasterIndex.from_transform(
        affine,
        width=obj.sizes[x_dim],
        height=obj.sizes[y_dim],
        x_dim=x_dim,
        y_dim=y_dim,
        crs=detected_crs,
    )
    coords = Coordinates.from_xindex(index)
    return obj.assign_coords(coords)


def _isclose(a: float, b: float) -> bool:
    """Check if two floats are close using rasterix tolerance options."""
    opts = get_rasterix_options()
    return math.isclose(a, b, rel_tol=opts["transform_rtol"], abs_tol=opts["transform_atol"])


def _assert_transforms_are_compatible(*affines) -> None:
    A1 = affines[0]
    for index, A2 in enumerate(affines[1:]):
        if not (
            _isclose(A1.a, A2.a) and _isclose(A1.b, A2.b) and _isclose(A1.d, A2.d) and _isclose(A1.e, A2.e)
        ):
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
        self.xy_dims = self.dims[XAXIS], self.dims[YAXIS]
        self.dims = self.dims[YAXIS], self.dims[XAXIS]

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

    def equals(self, other: CoordinateTransform, *, exclude: frozenset[Hashable] | None = None) -> bool:
        if exclude is not None:
            raise NotImplementedError
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

    def equals(self, other: CoordinateTransform, *, exclude: frozenset[Hashable] | None = None) -> bool:
        if not isinstance(other, AxisAffineTransform):
            return False
        if exclude is not None:
            raise NotImplementedError

        # only compare the affine parameters of the relevant axis
        if self.is_xaxis:
            affine_match = _isclose(self.affine.a, other.affine.a) and _isclose(self.affine.c, other.affine.c)
        else:
            affine_match = _isclose(self.affine.e, other.affine.e) and _isclose(self.affine.f, other.affine.f)

        return affine_match and self.size == other.size

    def generate_coords(self, dims: tuple[str, ...] | None = None) -> dict[Hashable, Any]:
        assert dims is None or dims == self.dims
        return self.forward({self.dim: np.arange(self.size)})

    def slice(self, slice: slice) -> AxisAffineTransform:
        newrange = range(self.size)[slice]
        start = newrange.start
        step = newrange.step or 1
        size = len(newrange)
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
    ) -> AxisAffineTransformIndex | None:
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
        # otherwise drop the index
        # (TODO: return a PandasIndex with values computed by forward transformation when it
        #  will be possible to auto-convert RasterIndex to PandasIndex for x and/or y axis)
        else:
            return None
            # values = self.axis_transform.forward({self.dim: idxer})[self.axis_transform.coord_name]
            # if isinstance(idxer, Variable):
            #     new_dim = idxer.dims[0]
            # else:
            #     new_dim = self.dim
            # return PandasIndex(values, new_dim, coord_dtype=values.dtype)

    def sel(self, labels, method=None, tolerance=None):
        coord_name = self.axis_transform.coord_name
        label = labels[coord_name]
        transform = self.axis_transform
        if isinstance(label, slice):
            # Use 'is None' check instead of 'or' to correctly handle 0 values
            label = slice(
                transform.forward({coord_name: 0})[coord_name] if label.start is None else label.start,
                transform.forward({coord_name: transform.size})[coord_name]
                if label.stop is None
                else label.stop,
                label.step,
            )
            if label.step is None:
                # continuous interval slice indexing (preserves the index)
                pos = self.transform.reverse({coord_name: np.array([label.start, label.stop])})
                # np.round rounds to even, this way we round upwards
                pos = np.floor(pos[self.dim] + 0.5).astype("int")
                size = self.axis_transform.size
                # Clamp both start and stop to valid range [0, size]
                new_start = max(min(pos[0], size), 0)
                new_stop = max(min(pos[1] + 1, size), 0)
                # Ensure stop >= start (empty slice if selection is completely outside bounds)
                new_stop = max(new_stop, new_start)
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


WrappedIndex = tuple[AxisAffineTransformIndex, AxisAffineTransformIndex] | CoordinateTransformIndex


class RasterIndex(Index, xproj.ProjIndexMixin):
    """Xarray index for raster coordinate indexing and spatial operations.

    RasterIndex provides spatial indexing capabilities for raster data by wrapping
    one or more Xarray indexes that handle coordinate transformations. It supports
    both rectilinear and non-rectilinear (rotated/skewed) raster grids.

    The internal structure depends on the affine transformation:

    - **Non-rectilinear or rotated grids**: Uses a single 2D CoordinateTransformIndex
      for coupled x/y coordinates that handles rotation and skew.

    - **Rectilinear grids**: Uses separate 1D indexes for independent x/y axes,
      enabling more efficient slicing operations.

    RasterIndex is CRS-aware, i.e., it has a ``crs`` property that is used for
    checking equality or compatibility with other RasterIndex instances. CRS is
    optional.

    Do not use :py:meth:`~rasterix.RasterIndex.__init__` directly. Instead use
    :py:meth:`~rasterix.RasterIndex.from_transform` or
    :py:func:`~rasterix.assign_index`.

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

    _index: WrappedIndex
    _axis_independent: bool
    _crs: CRS | None
    _xy_shape: tuple[int, int]
    _xy_dims: tuple[str, str]
    _xy_coord_names: tuple[Hashable, Hashable]

    def __init__(self, index: WrappedIndex, crs: CRS | Any | None = None):
        if isinstance(index, CoordinateTransformIndex) and isinstance(index.transform, AffineTransform):
            self._axis_independent = False
            xtransform = cast(AffineTransform, index.transform)
            dim_size = xtransform.dim_size
            xy_dims = xtransform.xy_dims
            self._xy_shape = (dim_size[xy_dims[XAXIS]], dim_size[xy_dims[YAXIS]])
            self._xy_dims = xtransform.xy_dims
            self._xy_coord_names = (xtransform.coord_names[XAXIS], xtransform.coord_names[YAXIS])
        elif (
            isinstance(index, tuple)
            and len(index) == 2
            and isinstance(index[XAXIS], AxisAffineTransformIndex)
            and isinstance(index[YAXIS], AxisAffineTransformIndex)
        ):
            self._axis_independent = True
            self._xy_shape = (index[XAXIS].axis_transform.size, index[YAXIS].axis_transform.size)
            self._xy_dims = (index[XAXIS].axis_transform.dim, index[YAXIS].axis_transform.dim)
            self._xy_coord_names = (
                index[XAXIS].axis_transform.coord_name,
                index[YAXIS].axis_transform.coord_name,
            )
        else:
            raise ValueError(f"Could not create RasterIndex. Received invalid index {index!r}")

        self._index = index

        if crs is not None:
            crs = CRS.from_user_input(crs)
        self._crs = crs

    @property
    def _wrapped_indexes(self) -> tuple[CoordinateTransformIndex | AxisAffineTransformIndex, ...]:
        """Returns the wrapped index objects as a tuple."""
        if not isinstance(self._index, tuple):
            return (self._index,)
        else:
            return self._index

    @property
    def _xy_indexes(self) -> tuple[AxisAffineTransformIndex, AxisAffineTransformIndex]:
        assert self._axis_independent
        return cast(tuple[AxisAffineTransformIndex, AxisAffineTransformIndex], self._index)

    @property
    def _xxyy_index(self) -> CoordinateTransformIndex:
        assert not self._axis_independent
        return cast(CoordinateTransformIndex, self._index)

    @property
    def xy_shape(self) -> tuple[int, int]:
        """Return the dimension size of the X and Y axis, respectively."""
        return self._xy_shape

    @property
    def xy_dims(self) -> tuple[str, str]:
        """Return the dimension name of the X and Y axis, respectively."""
        return self._xy_dims

    @property
    def xy_coord_names(self) -> tuple[Hashable, Hashable]:
        """Return the name of the coordinate variables representing labels on
        the X and Y axis, respectively.

        """
        return self._xy_coord_names

    @classmethod
    def from_transform(
        cls,
        affine: Affine,
        *,
        width: int,
        height: int,
        x_dim: str = "x",
        y_dim: str = "y",
        x_coord_name: str = "xc",
        y_coord_name: str = "yc",
        crs: CRS | Any | None = None,
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
        x_dim : str, optional
            Name for the x dimension.
        y_dim : str, optional
            Name for the y dimension.
        x_coord_name : str, optional
            Name for the x dimension. For non-rectilinear transforms only.
        y_coord_name : str, optional
            Name for the y dimension. For non-rectilinear transforms only.
        crs : :class:`pyproj.crs.CRS` or any, optional
            The coordinate reference system. Any value accepted by
            :meth:`pyproj.crs.CRS.from_user_input`.

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

        Create a non-rectilinear index:

        >>> index = RasterIndex.from_transform(
        ...     Affine.rotation(45), width=100, height=100, x_coord_name="x1", y_coord_name="x2"
        ... )
        """
        index: WrappedIndex

        # pixel centered coordinates
        affine = affine * Affine.translation(0.5, 0.5)

        if affine.is_rectilinear and affine.b == affine.d == 0:
            x_transform = AxisAffineTransform(affine, width, x_dim, x_dim, is_xaxis=True)
            y_transform = AxisAffineTransform(affine, height, y_dim, y_dim, is_xaxis=False)
            index = (
                AxisAffineTransformIndex(x_transform),
                AxisAffineTransformIndex(y_transform),
            )
        else:
            xy_transform = AffineTransform(
                affine,
                width,
                height,
                x_dim=x_dim,
                y_dim=y_dim,
                x_coord_name=x_coord_name,
                y_coord_name=y_coord_name,
            )
            index = CoordinateTransformIndex(xy_transform)

        return cls(index, crs=crs)

    @classmethod
    def from_tiepoint_and_scale(
        cls,
        *,
        tiepoint: list[float] | tuple[float, ...],
        scale: list[float] | tuple[float, ...],
        width: int,
        height: int,
        x_dim: str = "x",
        y_dim: str = "y",
        x_coord_name: str = "xc",
        y_coord_name: str = "yc",
        crs: CRS | Any | None = None,
    ) -> RasterIndex:
        """Create a RasterIndex from GeoTIFF tiepoint and pixel scale metadata.

        Parameters
        ----------
        tiepoint : list or tuple
            GeoTIFF model tiepoint in format [I, J, K, X, Y, Z]
            where (I, J, K) are pixel coords and (X, Y, Z) are world coords.
        scale : list or tuple
            GeoTIFF model pixel scale in format [ScaleX, ScaleY, ScaleZ].
        width : int
            Number of pixels in the x direction.
        height : int
            Number of pixels in the y direction.
        x_dim : str, optional
            Name for the x dimension.
        y_dim : str, optional
            Name for the y dimension.
        x_coord_name : str, optional
            Name for the x coordinate. For non-rectilinear transforms only.
        y_coord_name : str, optional
            Name for the y coordinate. For non-rectilinear transforms only.
        crs : :class:`pyproj.crs.CRS` or any, optional
            The coordinate reference system. Any value accepted by
            :meth:`pyproj.crs.CRS.from_user_input`.

        Returns
        -------
        RasterIndex
            A new RasterIndex object with appropriate internal structure.

        Raises
        ------
        AssertionError
            If ScaleZ is not 0 (only 2D rasters are supported).

        Examples
        --------
        Create an index from GeoTIFF metadata:

        >>> tiepoint = [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0]
        >>> scale = [30.0, 30.0, 0.0]
        >>> index = RasterIndex.from_tiepoint_and_scale(tiepoint=tiepoint, scale=scale, width=100, height=100)
        """
        from rasterix.lib import affine_from_tiepoint_and_scale

        affine = affine_from_tiepoint_and_scale(tiepoint, scale)
        return cls.from_transform(
            affine,
            width=width,
            height=height,
            x_dim=x_dim,
            y_dim=y_dim,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            crs=crs,
        )

    @classmethod
    def from_stac_proj_metadata(
        cls,
        metadata: dict,
        *,
        width: int,
        height: int,
        x_dim: str = "x",
        y_dim: str = "y",
        x_coord_name: str = "xc",
        y_coord_name: str = "yc",
        crs: CRS | Any | None = None,
    ) -> RasterIndex:
        """Create a RasterIndex from STAC projection metadata.

        Parameters
        ----------
        metadata : dict
            Dictionary containing STAC metadata. Must contain a 'proj:transform' key
            with the affine transformation as a flat array.
        width : int
            Number of pixels in the x direction.
        height : int
            Number of pixels in the y direction.
        x_dim : str, optional
            Name for the x dimension.
        y_dim : str, optional
            Name for the y dimension.
        x_coord_name : str, optional
            Name for the x coordinate. For non-rectilinear transforms only.
        y_coord_name : str, optional
            Name for the y coordinate. For non-rectilinear transforms only.
        crs : :class:`pyproj.crs.CRS` or any, optional
            The coordinate reference system. Any value accepted by
            :meth:`pyproj.crs.CRS.from_user_input`.

        Returns
        -------
        RasterIndex
            A new RasterIndex object with appropriate internal structure.

        Raises
        ------
        ValueError
            If 'proj:transform' is not found in metadata or has invalid format.

        Examples
        --------
        Create an index from STAC metadata:

        >>> metadata = {"proj:transform": [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0]}
        >>> index = RasterIndex.from_stac_proj_metadata(metadata, width=100, height=100)
        """
        from rasterix.lib import affine_from_stac_proj_metadata

        affine = affine_from_stac_proj_metadata(metadata)
        if affine is None:
            raise ValueError("metadata must contain 'proj:transform' key")

        return cls.from_transform(
            affine,
            width=width,
            height=height,
            x_dim=x_dim,
            y_dim=y_dim,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            crs=crs,
        )

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

        for index in self._wrapped_indexes:
            new_variables.update(index.create_variables())

        if self.crs is not None:
            xname, yname = self.xy_coord_names
            xattrs, yattrs = self.crs.cs_to_cf()
            if "axis" in xattrs and "axis" in yattrs:
                # The axis order is defined by the projection
                # So we have to figure which is which.
                # This is an ugly hack that works for common cases.
                if xattrs["axis"] == "Y":
                    xattrs, yattrs = yattrs, xattrs
                new_variables[xname].attrs = xattrs
                new_variables[yname].attrs = yattrs

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
        if not self._axis_independent:
            # preserve RasterIndex is not supported in the case of coupled x/y 2D coordinates
            return None
        else:
            new_indexes = []

            for index in self._xy_indexes:
                dim = index.axis_transform.dim

                if dim not in indexers:
                    # simply propagate the index
                    new_indexes.append(index)
                else:
                    new_index = index.isel({dim: indexers[dim]})
                    if new_index is not None:
                        new_indexes.append(new_index)

            # TODO: if/when supported in Xarray, return PandasIndex instances for either the
            # x or the y axis (or both) instead of returning None (drop the index)
            if len(new_indexes) == 2:
                return RasterIndex(tuple(new_indexes), crs=self.crs)
            else:
                return None

    def sel(self, labels: dict[Any, Any], method=None, tolerance=None) -> IndexSelResult:
        if not self._axis_independent:
            return self._xxyy_index.sel(labels, method=method, tolerance=tolerance)
        else:
            results: list[IndexSelResult] = []

            for index in self._xy_indexes:
                coord_name = index.axis_transform.coord_name
                if coord_name in labels:
                    res = index.sel({coord_name: labels[coord_name]}, method=method, tolerance=tolerance)
                    results.append(res)

            return merge_sel_results(results)

    def equals(self, other: Index, *, exclude=None) -> bool:
        if exclude is None:
            exclude = {}

        if not isinstance(other, RasterIndex):
            return False
        if not self._proj_crs_equals(cast(CRSAwareIndex, other), allow_none=True):
            return False
        if self._axis_independent != other._axis_independent:
            return False

        if self._axis_independent:
            return all(
                idx.equals(other_idx)
                for idx, other_idx in zip(self._xy_indexes, other._xy_indexes)
                if idx.axis_transform.dim not in exclude
            )
        else:
            return self._xxyy_index.equals(other._xxyy_index)

    def transform(self) -> Affine:
        """Affine transform for top-left corners."""
        return self.center_transform() * Affine.translation(-0.5, -0.5)

    def as_geotransform(self, *, decimals: int | None = None) -> str:
        """Convert the affine transform to a string suitable for saving as the GeoTransform attribute."""
        gt = self.transform().to_gdal()
        if decimals is not None:
            fmt = f".{decimals}f"
            return " ".join(f"{num:{fmt}}" for num in gt)
        else:
            return " ".join(map(str, gt))

    def center_transform(self) -> Affine:
        """Affine transform for cell centers."""
        if not self._axis_independent:
            return cast(AffineTransform, self._xxyy_index.transform).affine
        else:
            x = self._xy_indexes[XAXIS].axis_transform.affine
            y = self._xy_indexes[YAXIS].axis_transform.affine
            return Affine(x.a, x.b, x.c, y.d, y.e, y.f)

    def _as_pandas_index(self) -> dict[str, PandasIndex]:
        """Convert RasterIndex to equivalent PandasIndex objects.

        Returns a dict mapping dimension names to PandasIndex instances.
        For internal/testing use.
        """
        result = {}
        if self._axis_independent:
            for idx in self._xy_indexes:
                dim = idx.axis_transform.dim
                values = idx.to_pandas_index()
                result[dim] = PandasIndex(values, dim)
        else:
            raise NotImplementedError("_as_pandas_index not supported for non-rectilinear grids")

        return result

    @property
    def bbox(self) -> BoundingBox:
        """Bounding Box for index.

        Returns
        -------
        BoundingBox
        """
        yx_shape = (self.xy_shape[YAXIS], self.xy_shape[XAXIS])
        return BoundingBox.from_transform(shape=yx_shape, transform=self.transform())

    @classmethod
    def concat(
        cls,
        indexes: Sequence[Self],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> RasterIndex:
        if len(indexes) == 1:
            return next(iter(indexes))

        if positions is not None:
            raise NotImplementedError

        # Note: I am assuming that xarray has calling `align(..., exclude="x" [or 'y'])` already
        # and checked for equality along "y" [or 'x']
        new_bbox = bbox_union(as_compatible_bboxes(*indexes, concat_dim=dim))
        return indexes[0]._new_with_bbox(new_bbox)

    def _new_with_bbox(self, bbox: BoundingBox) -> RasterIndex:
        affine = self.transform()
        new_affine, Nx, Ny = bbox_to_affine(bbox, rx=affine.a, ry=affine.e)
        # TODO: set xdim, ydim explicitly
        new_index = self.from_transform(new_affine, width=Nx, height=Ny)
        opts = get_rasterix_options()
        assert new_index.bbox.isclose(bbox, rtol=opts["transform_rtol"], atol=opts["transform_atol"])
        return new_index

    def join(self, other: RasterIndex, how: JoinOptions = "inner") -> RasterIndex:
        """Join two RasterIndexes by computing the union or intersection of their bounding boxes.

        Transform compatibility is checked using the tolerance configured via
        :py:func:`rasterix.set_options` (``transform_rtol`` and ``transform_atol``).
        """
        if not self._proj_crs_equals(cast(CRSAwareIndex, other), allow_none=True):
            raise ValueError(
                "raster indexes on objects to align do not have the same CRS\n"
                f"first index:\n{self!r}\n\nsecond index:\n{other!r}"
            )

        if len(self._wrapped_indexes) != len(other._wrapped_indexes):
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
        x_dim, y_dim = self.xy_dims
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
        indexers[x_dim] = get_indexer(
            theirs.left, ours.left, inter.left, inter.right, spacing=dx, tol=tol, size=other.xy_shape[XAXIS]
        )
        indexers[y_dim] = get_indexer(
            theirs.top, ours.top, inter.top, inter.bottom, spacing=dy, tol=tol, size=other.xy_shape[YAXIS]
        )
        return indexers

    def _repr_inline_(self, max_width: int) -> str:
        # TODO: remove when fixed in Xarray (https://github.com/pydata/xarray/pull/10415)
        if max_width is None:
            max_width = get_options()["display_width"]

        srs = xproj.format_crs(self.crs, max_width=max_width)
        return f"{self.__class__.__name__} (crs={srs})"

    def __repr__(self) -> str:
        srs = xproj.format_crs(self.crs)

        def index_repr(idx) -> str:
            return textwrap.indent(f"{type(idx).__name__}({idx.transform!r})", "    ")

        if self._axis_independent:
            idx_repr = "\n".join(index_repr(idx) for idx in self._xy_indexes)
        else:
            idx_repr = index_repr(self._xxyy_index)

        return f"RasterIndex(crs={srs})\n{idx_repr}"


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


def bbox_to_affine(bbox: BoundingBox, rx, ry) -> tuple[Affine, int, int]:
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
        t.c + i.xy_shape[XAXIS] * t.a for i, t in zip(indexes[:-1], transforms[:-1])
    )
    expected_off_y = (transforms[0].f,) + tuple(
        t.f + i.xy_shape[YAXIS] * t.e for i, t in zip(indexes[:-1], transforms[:-1])
    )

    off_x = tuple(t.c for t in transforms)
    off_y = tuple(t.f for t in transforms)

    if concat_dim is not None:
        if all(_isclose(o, off_x[0]) for o in off_x[1:]) and all(_isclose(o, off_y[0]) for o in off_y[1:]):
            raise ValueError("Attempting to concatenate arrays with same transform along X or Y.")

    # note: Xarray alignment already ensures that the indexes dimensions are compatible.
    x_dim, y_dim = indexes[0].xy_dims

    if concat_dim == x_dim:
        if any(not _isclose(off_y[0], o) for o in off_y[1:]):
            raise ValueError("offsets must be identical in Y when concatenating along X")
        if any(not _isclose(a, b) for a, b in zip(off_x, expected_off_x)):
            raise ValueError(
                f"X offsets are incompatible. Provided offsets {off_x}, expected offsets: {expected_off_x}"
            )
    elif concat_dim == y_dim:
        if any(not _isclose(off_x[0], o) for o in off_x[1:]):
            raise ValueError("offsets must be identical in X when concatenating along Y")

        if any(not _isclose(a, b) for a, b in zip(off_y, expected_off_y)):
            raise ValueError(
                f"Y offsets are incompatible. Provided offsets {off_y}, expected offsets: {expected_off_y}"
            )

    return tuple(i.bbox for i in indexes)
