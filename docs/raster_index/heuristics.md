# Heuristics

{py:func}`~rasterix.assign_index` automatically detects the affine transform by trying each source in order, using the first one that succeeds:

| Priority | Source                                 | Location                                      | Reference                                                                                               |
| -------- | -------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 0        | Existing `RasterIndex`                 | Index already on the x/y dims                 | Built at open time by some backends, e.g. gdalxarray                                                    |
| 1        | `GeoTransform`                         | CF grid mapping variable (e.g. `spatial_ref`) | [GDAL GeoTransform](https://gdal.org/en/stable/tutorials/geotransforms_tut.html)                        |
| 2        | `proj:transform`                       | DataArray `.attrs`                            | [STAC Projection Extension](https://github.com/stac-extensions/projection)                              |
| 3        | `model_tiepoint` + `model_pixel_scale` | DataArray `.attrs`                            | [GeoTIFF spec](https://docs.ogc.org/is/19-008r4/19-008r4.html)                                          |
| 4        | `spatial:transform`                    | DataArray, data variable, or Dataset `.attrs` | [Zarr Spatial Convention](https://zarr-specs.readthedocs.io/en/latest/v3/conventions/spatial/v1.0.html) |
| 5        | 1D coordinate arrays                   | Coordinate variables for x/y dims             | Common in NetCDF                                                                                        |

## Existing RasterIndex

For priority 0, if the object already has a single {py:class}`~rasterix.RasterIndex` (or one over the requested `x_dim`/`y_dim`), its affine transform and periods are reused and only the CRS is (re)detected. Backends such as [gdalxarray](https://github.com/hypertidy/gdalxarray) build the RasterIndex directly from GDAL's geotransform in classic-raster mode, so calling `assign_index` on their output keeps rotated and sheared transforms, whose 2D coordinates the coordinate fallback cannot read.

## Dimension detection

When `x_dim`/`y_dim` are not given, they are taken from an existing RasterIndex, then from `spatial:dimensions` (see below), then from these dimension-name pairs in order: `x`/`y`, `longitude`/`latitude`, `X`/`Y` (the names GDAL's multidimensional API gives classic rasters), `lon`/`lat`. Failing those, 1D dimension coordinates are matched on the CF `axis`, `standard_name` or `units` (`degrees_east`/`degrees_north`) attributes. If neither dimension can be determined a `ValueError` asks for them explicitly.

## Grid mapping variable lookup

For priority 1, the grid mapping variable is found following CF conventions:

- **DataArray**: looks for a `grid_mapping` attribute on the DataArray
- **Dataset**: uses the first `grid_mapping` attribute found across data variables
- **Fallback**: a coordinate variable named `spatial_ref`

## Zarr spatial convention lookup

For priority 4, the convention must be registered in the `zarr_conventions` attribute (by name `spatial:` or by UUID); bare `spatial:` attributes are ignored otherwise. Attributes are looked up on the DataArray for DataArrays, and on each data variable then the Dataset itself for Datasets — array-level attributes take precedence over group-level ones, per the convention.

If a `spatial:dimensions` attribute is present, it is also used to auto-detect the x/y dimension names when `x_dim`/`y_dim` are not passed to {py:func}`~rasterix.assign_index`. The listed names are interpreted as `[y, x]`, following the convention's examples; `spatial:transform` uses standard [Affine](https://affine.readthedocs.io/en/latest/) coefficient ordering `[a, b, c, d, e, f]`, mapping `(column, row) -> (x, y)`.

## Coordinate array fallback

For priority 5, coordinate variables must be 1D with at least 2 values. Pixel spacing is computed as `x[1] - x[0]` and `y[1] - y[0]`, and coordinates are assumed to be pixel-centered.

## Logging

Enable trace-level logging to see which source is used:

```python
import logging
from rasterix.lib import TRACE

logging.basicConfig(level=TRACE)
logging.getLogger("rasterix").setLevel(TRACE)
```
