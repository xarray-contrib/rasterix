# `assign_index` Heuristics

This page documents the heuristics and decision-making logic used by rasterix when working with spatial data.

## Affine Transform Detection in `assign_index`

When calling {py:func}`assign_index()` to create a `RasterIndex` for an Xarray object, rasterix needs to determine the affine transformation that maps pixel coordinates to spatial coordinates. The function follows a specific priority order when searching for this information.

### Priority Order

Rasterix checks for transform information in the following order:

1. `GeoTransform` attribute on a CF 'grid mapping' variable (commonly `spatial_ref`).
1. STAC `proj:transform` attribute
1. GeoTIFF metadata: (`model_tiepoint` and `model_pixel_scale` for now)

Each method is tried in sequence, and the first successful match is used.

If these fail, an index is constructed from explicit coordinate arrays if present.

### 1. `GeoTransform`

```{seealso}
[GDAL GeoTransform tutorial](https://gdal.org/en/stable/tutorials/geotransforms_tut.html)
```

The function first looks for a CF conventions "grid mapping" variable (commonly named `spatial_ref`). The grid mapping variable is identified by:

- For DataArrays: the `grid_mapping` attribute on the DataArray itself
- For Datasets: the `grid_mapping` attribute on the first data variable
- As a fallback: a coordinate variable named `spatial_ref`

If found, rasterix checks for a `GeoTransform` attribute on this variable, which should be a GDAL-format geotransform string with 6 space-separated numbers.

**Format**: `"c a b f d e"` (GDAL format)

**Example**:

```python
da.coords["spatial_ref"].attrs["GeoTransform"] = "323400.0 30.0 0.0 4265400.0 0.0 30.0"
```

### 2. STAC `proj:transform`

```{seealso}
[STAC Projection Extension](https://github.com/stac-extensions/projection)
```

If no GeoTransform is found, rasterix checks the DataArray's attributes for a STAC projection extension `proj:transform` field. This represents the affine transformation as a flat array.

**Format**: `[a, b, c, d, e, f]` or `[a, b, c, d, e, f, 0, 0, 1]`

The transformation follows this matrix formula:

```
[Xp]   [a  b  c]   [Pixel]
[Yp] = [d  e  f] * [Line ]
[1 ]   [0  0  1]   [1    ]
```

Where:

- `a` = pixel width (x-scale)
- `b` = row rotation (typically 0)
- `c` = x-coordinate of upper-left pixel corner
- `d` = column rotation (typically 0)
- `e` = pixel height (y-scale, negative if y decreases)
- `f` = y-coordinate of upper-left pixel corner

**Example**:

```python
da.attrs["proj:transform"] = [30.0, 0.0, 323400.0, 0.0, 30.0, 4268400.0]
```

### 3. GeoTIFF Metadata

```{seealso}
[GeoTIFF specification on coordinate transformations](https://docs.ogc.org/is/19-008r4/19-008r4.html#_geotiff_tags_for_coordinate_transformations)
```

If no STAC transform is found, rasterix checks for GeoTIFF-style metadata using model tiepoints and pixel scale.

**Format**:

- `model_tiepoint`: `[I, J, K, X, Y, Z]` - Maps pixel `(I, J, K)` to world coordinates `(X, Y, Z)`
- `model_pixel_scale`: `[ScaleX, ScaleY, ScaleZ]` - Pixel dimensions in world coordinates

**Constraints**:

- `ScaleZ` must be 0 (only 2D rasters are supported)
- If the tiepoint is at pixel `(I, J)`, the affine is computed as:
  - `c = X - I * ScaleX`
  - `f = Y - J * ScaleY`

**Example**:

```python
da.attrs["model_tiepoint"] = [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0]
da.attrs["model_pixel_scale"] = [30.0, 30.0, 0.0]
```

**Trace log**: `"Creating affine from GeoTIFF model_tiepoint and model_pixel_scale attributes"`

### 4. Coordinate Arrays

If no metadata is found, rasterix falls back to computing the affine transformation from 1D coordinate arrays.

**Requirements**:

- Coordinate variables must exist for both x and y dimensions
- Coordinates must be 1D arrays
- Coordinates must have at least 2 values to compute spacing

**Computation**:

- Pixel spacing: `dx = x[1] - x[0]`, `dy = y[1] - y[0]`
- Origin calculation accounts for whether y increases or decreases
- Assumes pixel-centered coordinates, adjusts to pixel corners

**Example**:

```python
da = xr.DataArray(
    data,
    coords={
        "x": np.arange(0.5, 100.5),  # Pixel-centered
        "y": np.arange(0.5, 100.5),
    },
)
```

### Logging

To see which method is being used, enable trace-level logging:

```python
import logging

logging.basicConfig(level=logging.TRACE)
logging.getLogger("rasterix").setLevel(logging.TRACE)
```

This will output messages like:

```
TRACE:rasterix:Creating affine from STAC proj:transform attribute
```
