```{eval-rst}
.. currentmodule:: rasterix
```

# Design Choices

In designing {py:class}`RasterIndex`, we faced a few thorny questions. Below we discuss these considerations, and the approach we've taken.
Ultimately, there are no easy answers and tradeoffs to be made.

## CRS handling

{py:class}`xproj.CRSIndex` is an attempt at providing a building block for CRS handling in the Xarray ecosystem.

How might `RasterIndex` integrate with `xproj.CRSIndex`? Our options are:

1. fully encapsulate {py:class}`xproj.CRSIndex`, or
1. satisfy the ["CRS-aware" protocol](https://xproj.readthedocs.io/en/latest/integration.html) provided by `xproj`, or
1. simply handle the affine transform and ignore the CRS altogether.

Why do we want CRS handling? We want Xarray to disallow alignment of two Xarray objects with different CRS e.g. `da1 + da2` should fail if `da1`, and `da2` have different CRS. This is enabled by assigning {py:class}`xproj.CRSIndex` to the `spatial_ref` variable.

### Why should `RasterIndex` be aware of the CRS?

RasterIndex handles indexing and the creation of coordinate variables. With CRS information handy, this would allow us to

1. Support wraparound indexing along the `longitude` dimension ({issue}`26`)
1. Assign appropriate attributes to the created coordinate variables ({issue}`22`). (e.g. choose between `standard_name: latitude` and `standard_name: projection_y_coordinate`)
1. more?

### Why not encapsulate CRSIndex?

If RasterIndex must track CRS in some form, one way to do that would be to have RasterIndex internally build a `CRSIndex` for the `spatial_ref` variable.
Thus, `RasterIndex` would be associated with 3 variables instead of 2: `x`, `y`, and `spatial_ref`, for example.

The downside of this approach is that it doesn't compose well with any other Index that would also like to handle the CRS (e.g. {py:class}`xvec.GeometryIndex`).
For example, `xr.merge([geometries, raster])` where `geometries` has `xvec.GeometryIndex[geometry, spatial_ref]` (square brackets list associated coordinate variable names) and `raster` has `RasterIndex[x, y, spatial_ref]`, would fail because the variable `spatial_ref` is associated with two Indexes of different types.
This fails because the Xarray model enforces that *one Variable is only associated with only one Index*, in order to prevent different Indexes modifying the same Variable.

### CRS-aware Index

Therefore, we have chosen to experiment with the "CRS-aware" approach described in the [xproj docs](https://xproj.readthedocs.io/en/latest/integration.html).
Here `RasterIndex` tracks it's own _optional_ copy of a CRS object (not an Index) and defines the hooks needed for `CRSIndex` to communicate with `RasterIndex`.
The downside here is that CRS information is duplicated in two places explicitly, and requires explicit handling to ensure consistency

### Don't like it?

We chose this approach to enable experimentation. It is entirely possible to experiment with other approaches. Please reach out if you have opinions on this topic.
