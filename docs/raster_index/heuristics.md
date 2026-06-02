# Heuristics

{py:func}`~rasterix.assign_index` automatically detects the affine transform by trying each source in order, using the first one that succeeds:

| Priority | Source                                 | Location                                      | Reference                                                                                               |
| -------- | -------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1        | `GeoTransform`                         | CF grid mapping variable (e.g. `spatial_ref`) | [GDAL GeoTransform](https://gdal.org/en/stable/tutorials/geotransforms_tut.html)                        |
| 2        | `proj:transform`                       | DataArray `.attrs`                            | [STAC Projection Extension](https://github.com/stac-extensions/projection)                              |
| 3        | `model_tiepoint` + `model_pixel_scale` | DataArray `.attrs`                            | [GeoTIFF spec](https://docs.ogc.org/is/19-008r4/19-008r4.html)                                          |
| 4        | `spatial:transform`                    | DataArray, data variable, or Dataset `.attrs` | [Zarr Spatial Convention](https://zarr-specs.readthedocs.io/en/latest/v3/conventions/spatial/v1.0.html) |
| 5        | 1D coordinate arrays                   | Coordinate variables for x/y dims             | Common in NetCDF                                                                                        |

## Grid mapping variable lookup

For priority 1, the grid mapping variable is found following CF conventions:

- **DataArray**: looks for a `grid_mapping` attribute on the DataArray
- **Dataset**: uses the first `grid_mapping` attribute found across data variables
- **Fallback**: a coordinate variable named `spatial_ref`

## Zarr spatial convention lookup

For priority 4, the convention must be registered in the `zarr_conventions` attribute (by name `spatial:` or by UUID); bare `spatial:` attributes are ignored otherwise. Attributes are looked up on the DataArray for DataArrays, and on each data variable then the Dataset itself for Datasets — array-level attributes take precedence over group-level ones, per the convention.

If a `spatial:dimensions` attribute is present, it is also used to auto-detect the x/y dimension names when `x_dim`/`y_dim` are not passed to {py:func}`~rasterix.assign_index`. The order of the listed names is not significant: X is whichever named dimension comes last in the array's dimension order, since the transform maps `(column, row) -> (x, y)` and columns vary along the trailing spatial dimension.

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
