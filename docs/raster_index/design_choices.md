```{eval-rst}
.. currentmodule:: rasterix
```

# Design Choices

In designing {py:class}`RasterIndex`, we faced a few thorny questions. Below we discuss these considerations, and the approach we've taken.
Ultimately, there are no easy answers and tradeoffs to be made.

## Handling the `GeoTransform` attribute

GDAL _chooses_ to track the affine transform using a `GeoTransform` attribute on a `spatial_ref` variable. The `"spatial_ref"` is a
"grid mapping" variable (as termed by the CF-conventions). It also records CRS information. Currently, our design is that
{py:class}`xproj.CRSIndex` controls the CRS information and handles the creation of the `"spatial_ref"` variable, or more generally,
the grid mapping variable. Thus, {py:class}`RasterIndex` _cannot_ keep the `"GeoTransform"` attribute on `"spatial_ref"` up-to-date
because it does not control it.

Thus, {py:func}`assign_index` will delete the `"GeoTransform"` attribute on the grid mapping variable if it is detected, after using it
to construct the affine matrix.
