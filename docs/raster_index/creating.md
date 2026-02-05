---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python
---

```{eval-rst}
.. currentmodule:: rasterix
```

```{code-cell} python
---
tags: [remove-cell]
---
%xmode minimal
import xarray as xr
xr.set_options(display_expand_indexes=True);
```

# Creating a RasterIndex

There are several ways to create a {py:class}`RasterIndex`, depending on the source of your data and the metadata conventions it uses.

## From existing Xarray objects: `assign_index`

The most common way to create a RasterIndex is using {py:func}`assign_index`. This function automatically detects the affine transform from various metadata conventions:

```{code-cell}
import numpy as np
import xarray as xr
import rasterix
```

### From GeoTIFF files (GDAL GeoTransform)

When loading data with rioxarray, the `GeoTransform` attribute is stored on the `spatial_ref` coordinate:

```{code-cell}
import pyproj

# Create a dataset with GDAL-style GeoTransform metadata
ds = xr.Dataset(
    {"temperature": (("y", "x"), np.random.rand(100, 100))},
    coords={
        "spatial_ref": (
            (),
            0,
            pyproj.CRS.from_epsg(32610).to_cf() | {"GeoTransform": "400000.0 10.0 0.0 5000000.0 0.0 -10.0"},
        )
    },
    attrs={"grid_mapping": "spatial_ref"},
)

# Assign a RasterIndex
ds = rasterix.assign_index(ds)
ds
```

### From 1D coordinate arrays

If your data has 1D coordinate arrays (common with NetCDF files), `assign_index` can infer the transform:

```{code-cell}
# Create dataset with 1D coordinates
ds_coords = xr.Dataset(
    {"temperature": (("y", "x"), np.random.rand(100, 100))},
    coords={
        "x": np.arange(400000, 401000, 10) + 5,  # pixel centers
        "y": np.arange(5000000, 4999000, -10) - 5,
    },
)

ds_coords = rasterix.assign_index(ds_coords)
ds_coords
```

### From GeoTIFF tiepoint/scale metadata

Some GeoTIFF files use `model_tiepoint` and `model_pixel_scale` attributes instead of GeoTransform:

```{code-cell}
ds_tiepoint = xr.Dataset(
    {"elevation": (("y", "x"), np.random.rand(100, 100))},
    attrs={
        "model_tiepoint": [0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0],
        "model_pixel_scale": [30.0, 30.0, 0.0],
    },
)

ds_tiepoint = rasterix.assign_index(ds_tiepoint)
ds_tiepoint
```

### From STAC proj:transform

Data loaded from STAC catalogs often includes `proj:transform` metadata:

```{code-cell}
ds_stac = xr.Dataset(
    {"reflectance": (("y", "x"), np.random.rand(100, 100))},
    attrs={
        "proj:transform": [30.0, 0.0, 323400.0, 0.0, -30.0, 4268400.0],
    },
)

ds_stac = rasterix.assign_index(ds_stac)
ds_stac
```

### From Zarr Spatial Convention

The [Zarr Spatial Convention](https://zarr-specs.readthedocs.io/en/latest/v3/conventions/spatial/v1.0.html) uses `spatial:transform` along with a `zarr_conventions` attribute to indicate that the data follows this convention:

```{code-cell}
ds_zarr_spatial = xr.Dataset(
    {"temperature": (("y", "x"), np.random.rand(100, 100))},
    attrs={
        "zarr_conventions": [{"name": "spatial:"}],
        "spatial:transform": [30.0, 0.0, 323400.0, 0.0, -30.0, 4268400.0],
    },
)

ds_zarr_spatial = rasterix.assign_index(ds_zarr_spatial)
ds_zarr_spatial
```

The Zarr Spatial Convention can also be combined with the [Zarr Proj Convention](https://zarr-specs.readthedocs.io/en/latest/v3/conventions/proj/v1.0.html) for CRS information using `proj:code`, `proj:wkt2`, or `proj:projjson`:

```{code-cell}
ds_zarr_proj = xr.Dataset(
    {"temperature": (("y", "x"), np.random.rand(100, 100))},
    attrs={
        "zarr_conventions": [{"name": "spatial:"}, {"name": "proj:"}],
        "spatial:transform": [30.0, 0.0, 323400.0, 0.0, -30.0, 4268400.0],
        "proj:code": "EPSG:32610",
    },
)

ds_zarr_proj = rasterix.assign_index(ds_zarr_proj)
ds_zarr_proj.xindexes["x"].crs
```

## Direct construction with class methods

For more control, you can create a {py:class}`RasterIndex` directly using class methods and then assign it to your data.

### `RasterIndex.from_transform`

Create from an affine transform directly:

```{code-cell}
from affine import Affine

transform = Affine(10.0, 0.0, 400000.0, 0.0, -10.0, 5000000.0)
index = rasterix.RasterIndex.from_transform(
    transform,
    width=100,
    height=100,
    crs="EPSG:32610",
)
index
```

```{code-cell}
# Assign to data
ds_manual = xr.Dataset(
    {"data": (("y", "x"), np.random.rand(100, 100))},
    coords=xr.Coordinates.from_xindex(index),
)
ds_manual
```

### `RasterIndex.from_tiepoint_and_scale`

Create from GeoTIFF-style tiepoint and pixel scale:

```{code-cell}
index = rasterix.RasterIndex.from_tiepoint_and_scale(
    tiepoint=[0.0, 0.0, 0.0, 323400.0, 4265400.0, 0.0],
    scale=[30.0, 30.0, 0.0],
    width=100,
    height=100,
    crs="EPSG:32610",
)
index
```

### `RasterIndex.from_stac_proj_metadata`

Create from STAC projection metadata:

```{code-cell}
metadata = {"proj:transform": [30.0, 0.0, 323400.0, 0.0, -30.0, 4268400.0]}
index = rasterix.RasterIndex.from_stac_proj_metadata(
    metadata,
    width=100,
    height=100,
    crs="EPSG:32610",
)
index
```

## Accessing transform information

Once created, you can access the affine transform:

```{code-cell}
ds = rasterix.assign_index(
    xr.Dataset(
        {"data": (("y", "x"), np.random.rand(100, 100))},
        coords={
            "spatial_ref": ((), 0, {"GeoTransform": "400000.0 10.0 0.0 5000000.0 0.0 -10.0"}),
        },
    )
)

# Top-left corner transform (GDAL convention)
ds.xindexes["x"].transform()
```

```{code-cell}
# Pixel center transform
ds.xindexes["x"].center_transform()
```

```{code-cell}
# Bounding box
ds.xindexes["x"].bbox
```

```{code-cell}
# As GeoTransform string (for saving)
ds.xindexes["x"].as_geotransform()
```
