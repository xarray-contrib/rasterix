# Rasterization Overview

Rasterix provides dask-aware tools for converting vector geometries to raster data. There are two main types of operations:

## Operations

### Rasterization (Burning)

"Burning" assigns integer indices to raster pixels based on which geometry they fall within. This is useful for:

- Creating categorical rasters from polygons
- Generating zone maps for zonal statistics
- Creating masks for spatial selection

**Functions:** `rasterize()`, `geometry_mask()`, `geometry_clip()`

### Coverage Calculation

Coverage calculation computes the precise fractional area of each pixel covered by each geometry. This is useful for:

- Area-weighted aggregations
- Accurate zonal statistics
- Sub-pixel analysis

**Function:** `coverage()` (exactextract only)

```{note}
The `coverage()` function returns a `sparse.COO` array, which efficiently stores the sparse coverage matrix. This is important because most pixels have zero coverage for any given geometry.
```

## Engines

Three rasterization engines are available, each with different tradeoffs:

| Engine           | GDAL Required | `all_touched` | Coverage | Notes                              |
| ---------------- | ------------- | ------------- | -------- | ---------------------------------- |
| **rasterio**     | Yes           | Yes           | No       | Most compatible, requires GDAL     |
| **rusterize**    | No            | No            | No       | Fast, easy to install              |
| **exactextract** | No            | No\*          | Yes      | Sub-pixel precision, sparse output |

\* exactextract naturally includes any pixel with non-zero coverage, similar to `all_touched=True`

### Choosing an Engine

- **rasterio**: Best when you need `all_touched=True` or already have GDAL installed
- **rusterize**: Best for simple rasterization without GDAL dependency (default when available)
- **exactextract**: Best when you need precise coverage fractions or area calculations

## Quick Examples

### Rasterization with different engines

```python
from rasterix.rasterize import rasterize

# Auto-select engine (prefers rusterize, falls back to rasterio)
result = rasterize(ds, geometries, xdim="x", ydim="y")

# Explicitly choose an engine
result = rasterize(ds, geometries, xdim="x", ydim="y", engine="rasterio")
result = rasterize(ds, geometries, xdim="x", ydim="y", engine="rusterize")
result = rasterize(ds, geometries, xdim="x", ydim="y", engine="exactextract")
```

### Coverage calculation

```python
from rasterix.rasterize.exact import coverage

# Fractional coverage (0-1)
cov = coverage(ds, geometries, xdim="x", ydim="y", coverage_weight="fraction")

# Area in square meters
area = coverage(ds, geometries, xdim="x", ydim="y", coverage_weight="area_spherical_m2")

# Binary coverage (0 or 1)
binary = coverage(ds, geometries, xdim="x", ydim="y", coverage_weight="none")
```

### Masking and clipping

```python
from rasterix.rasterize import geometry_mask, geometry_clip

# Create a boolean mask
mask = geometry_mask(ds, geometries, xdim="x", ydim="y")

# Clip data to geometries
clipped = geometry_clip(ds, geometries, xdim="x", ydim="y")
```

## Out-of-Core Support

All operations support dask arrays and dask-geopandas for out-of-core computation:

```python
import dask_geopandas as dgpd

# Chunked raster + dask geometries
result = rasterize(
    ds.chunk({"y": 100, "x": 100}),
    dgpd.from_geopandas(geometries, npartitions=4),
    xdim="x",
    ydim="y",
)
```
