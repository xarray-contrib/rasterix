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

# Tolerance

Raster tiles from different sources may have tiny floating-point differences in their transform parameters (resolution, origin), causing alignment or concatenation to fail even though the tiles are intended to be compatible.

Rasterix applies a small default relative tolerance (`transform_rtol=1e-12`) when comparing transform parameters. Use {py:func}`set_options` to adjust this. Both `transform_rtol` (relative) and `transform_atol` (absolute) are supported, matching {py:func}`math.isclose` semantics.

```{code-cell}
import rasterix

rasterix.get_options()
```

## Combining

Consider two tiles that should concatenate along X, but have a tiny difference in their X resolution (`926.625433054...` vs `926.625433055...`):

```{note}
These transforms come from a real dataset [MODIS/Terra Vegetation Indices Monthly L3 Global 1km SIN Grid V061](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13a3-061#variables)!
```

```{code-cell}
import pyproj
import numpy as np

transforms = [
    "-8895604.157333 926.6254330549995 0.0 3335851.559 0.0 -926.6254330558334",
    "-7783653.637667 926.6254330558338 0.0 3335851.559 0.0 -926.6254330558334",
]

dsets = [
    xr.Dataset(
        {"temp": (("y", "x"), np.ones((10, 1200)), {"grid_mapping": "spatial_ref"})},
        coords={
            "spatial_ref": (
                (),
                0,
                pyproj.CRS.from_epsg(3857).to_cf() | {"GeoTransform": transform},
            )
        },
    )
    for transform in transforms
]
dsets = list(map(rasterix.assign_index, dsets))
```

The relative difference here is ~9e-13, which is handled by the default tolerance:

```{code-cell}
xr.concat(dsets, dim="x")
```

### Larger differences

With larger differences (~1e-8 relative), the default tolerance is not enough:

```{code-cell}
transforms_noisy = [
    "0.0 1.0 0.0 1.0 0.0 -1.0",
    "10.0 1.00000001 0.0 1.0 0.0 -1.0",
]

dsets_noisy = [
    xr.Dataset(
        {"temp": (("y", "x"), np.ones((10, 10)), {"grid_mapping": "spatial_ref"})},
        coords={
            "spatial_ref": (
                (),
                0,
                pyproj.CRS.from_epsg(4326).to_cf() | {"GeoTransform": transform},
            )
        },
    )
    for transform in transforms_noisy
]
dsets_noisy = list(map(rasterix.assign_index, dsets_noisy))
```

```{code-cell}
---
tags: [raises-exception]
---
xr.concat(dsets_noisy, dim="x")
```

Increase the tolerance using {py:func}`set_options` as a context manager:

```{code-cell}
with rasterix.set_options(transform_rtol=1e-7):
    result = xr.concat(dsets_noisy, dim="x")
result
```

## Alignment

The same tolerance applies to {py:func}`xarray.align`:

```{code-cell}
with rasterix.set_options(transform_rtol=1e-7):
    aligned = xr.align(*dsets_noisy, join="outer")
aligned[0]
```

For `join="exact"`, tolerance helps when two datasets cover the same area but differ only in their resolution. Here both datasets have origin `0.0` and 10 pixels, but slightly different spacing:

```{code-cell}
transforms_exact = [
    "0.0 1.0 0.0 1.0 0.0 -1.0",
    "0.0 1.00000001 0.0 1.0 0.0 -1.0",
]

dsets_exact = [
    xr.Dataset(
        {"temp": (("y", "x"), np.ones((10, 10)), {"grid_mapping": "spatial_ref"})},
        coords={
            "spatial_ref": (
                (),
                0,
                pyproj.CRS.from_epsg(4326).to_cf() | {"GeoTransform": transform},
            )
        },
    )
    for transform in transforms_exact
]
dsets_exact = list(map(rasterix.assign_index, dsets_exact))
```

```{code-cell}
with rasterix.set_options(transform_rtol=1e-7):
    exact = xr.align(*dsets_exact, join="exact")
exact[0]
```
