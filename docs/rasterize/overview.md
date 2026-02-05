# Overview

Rasterix provides dask-aware tools for converting vector geometries to raster data.

## Functions

### rasterize

{py:func}`~rasterix.rasterize.rasterize` burns geometry indices into a raster. Each pixel is assigned an integer corresponding to the geometry that covers it.

```python
from rasterix.rasterize import rasterize

result = rasterize(ds, geometries, xdim="x", ydim="y")
```

Use cases:

- Creating categorical rasters from polygons
- Generating zone maps for zonal statistics
- Labeling pixels by region

### geometry_mask

{py:func}`~rasterix.rasterize.geometry_mask` creates a boolean mask indicating which pixels fall within any of the provided geometries.

```python
from rasterix.rasterize import geometry_mask

mask = geometry_mask(ds, geometries, xdim="x", ydim="y")
```

Use cases:

- Creating masks for spatial selection
- Identifying pixels inside/outside regions

### geometry_clip

{py:func}`~rasterix.rasterize.geometry_clip` clips raster data to geometries, setting pixels outside the geometries to NaN.

```python
from rasterix.rasterize import geometry_clip

clipped = geometry_clip(ds, geometries, xdim="x", ydim="y")
```

Use cases:

- Extracting data for a specific region
- Removing data outside an area of interest

## Engines

Three rasterization engines are available for the functions above:

| Engine           | GDAL Required | `all_touched` | Notes                          |
| ---------------- | ------------- | ------------- | ------------------------------ |
| **rasterio**     | Yes           | Yes           | Most compatible, requires GDAL |
| **rusterize**    | No            | No            | Fast, easy to install, no GDAL |
| **exactextract** | No            | No\*          | Sub-pixel precision, no GDAL   |

\* exactextract naturally includes any pixel with non-zero coverage, similar to `all_touched=True`

### Choosing an Engine

- **rasterio**: Best when you need `all_touched=True` or already have GDAL installed
- **rusterize**: Best for simple rasterization without GDAL dependency (default when available)
- **exactextract**: Best when you need sub-pixel precision without GDAL

```python
# Auto-select engine (prefers rusterize, falls back to rasterio)
result = rasterize(ds, geometries, xdim="x", ydim="y")

# Explicitly choose an engine
result = rasterize(ds, geometries, xdim="x", ydim="y", engine="rasterio")
result = rasterize(ds, geometries, xdim="x", ydim="y", engine="rusterize")
result = rasterize(ds, geometries, xdim="x", ydim="y", engine="exactextract")
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

## Coverage Calculation

In addition to the rasterization functions above, {py:func}`~rasterix.rasterize.exact.coverage` computes the precise fractional area of each pixel covered by each geometry. This is only available with the exactextract engine.

```python
from rasterix.rasterize.exact import coverage

# Fractional coverage (0-1)
cov = coverage(ds, geometries, xdim="x", ydim="y", coverage_weight="fraction")

# Area in square meters
area = coverage(ds, geometries, xdim="x", ydim="y", coverage_weight="area_spherical_m2")

# Binary coverage (0 or 1)
binary = coverage(ds, geometries, xdim="x", ydim="y", coverage_weight="none")
```

Use cases:

- Area-weighted aggregations
- Accurate zonal statistics
- Sub-pixel analysis

```{note}
The `coverage()` function returns a `sparse.COO` array, which efficiently stores the sparse coverage matrix. This is important because most pixels have zero coverage for any given geometry.
```
