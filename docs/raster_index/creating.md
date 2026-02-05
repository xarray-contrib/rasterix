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

# Creating

There are several ways to create a {py:class}`RasterIndex`, depending on the source of your data and the metadata conventions it uses.

```{code-cell}
import numpy as np
import xarray as xr
import rasterix
```

## Using `assign_index`

The easiest way to create a RasterIndex is using {py:func}`assign_index`. This function automatically detects the affine transform from various metadata conventions, including:

- [GDAL GeoTransform](https://gdal.org/en/stable/tutorials/geotransforms_tut.html) (from rioxarray/GeoTIFF files)
- [GeoTIFF](https://docs.ogc.org/is/19-008r4/19-008r4.html) tiepoint and pixel scale attributes
- [STAC projection extension](https://github.com/stac-extensions/projection) `proj:transform` metadata
- [Zarr Spatial Convention](https://zarr-specs.readthedocs.io/en/latest/v3/conventions/spatial/v1.0.html) (`spatial:transform`)
- 1D coordinate arrays (common in NetCDF files)

```{code-cell}
import pyproj

# Example: dataset with GDAL-style GeoTransform metadata
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

# assign_index auto-detects the convention and creates the RasterIndex
ds = rasterix.assign_index(ds)
ds
```

## Direct construction with class methods

For more control, you can create a {py:class}`RasterIndex` directly using class methods. This is useful when you have the transform parameters available directly, or when working with data that doesn't have embedded metadata.

### From an Affine transform

Use {py:meth}`RasterIndex.from_transform` to create from an [Affine](https://github.com/rasterio/affine) transform directly (corresponds to [GDAL GeoTransform](https://gdal.org/en/stable/tutorials/geotransforms_tut.html) convention):

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

### From a GeoTransform

Use {py:meth}`RasterIndex.from_geotransform` to create from a [GDAL GeoTransform](https://gdal.org/en/stable/tutorials/geotransforms_tut.html), either as a sequence or a space-separated string:

```{code-cell}
# From a tuple
geotransform = (400000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0)
index = rasterix.RasterIndex.from_geotransform(
    geotransform,
    width=100,
    height=100,
    crs="EPSG:32610",
)
index
```

```{code-cell}
# From a string (as commonly stored in netCDF attributes)
geotransform = "400000.0 10.0 0.0 5000000.0 0.0 -10.0"
index = rasterix.RasterIndex.from_geotransform(
    geotransform,
    width=100,
    height=100,
    crs="EPSG:32610",
)
index
```

### From GeoTIFF tiepoint and scale

Use {py:meth}`RasterIndex.from_tiepoint_and_scale` to create from [GeoTIFF](https://docs.ogc.org/is/19-008r4/19-008r4.html)-style tiepoint and pixel scale attributes:

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

### From STAC projection metadata

Use {py:meth}`RasterIndex.from_stac_proj_metadata` to create from [STAC projection extension](https://github.com/stac-extensions/projection) metadata:

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

Once created, you can access the affine transform using methods on {py:class}`RasterIndex`:

```{code-cell}
ds = rasterix.assign_index(
    xr.Dataset(
        {"data": (("y", "x"), np.random.rand(100, 100))},
        coords={
            "spatial_ref": ((), 0, {"GeoTransform": "400000.0 10.0 0.0 5000000.0 0.0 -10.0"}),
        },
    )
)

# Top-left corner transform (GDAL convention) via transform()
ds.xindexes["x"].transform()
```

```{code-cell}
# Pixel center transform via center_transform()
ds.xindexes["x"].center_transform()
```

```{code-cell}
# Bounding box via the bbox property
ds.xindexes["x"].bbox
```

```{code-cell}
# As GeoTransform string (for saving) via as_geotransform()
ds.xindexes["x"].as_geotransform()
```
