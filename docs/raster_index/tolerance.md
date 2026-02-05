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

This works with the default tolerance:

```{code-cell}
xr.concat(dsets, dim="x")
```

For larger differences, increase the tolerance using {py:func}`set_options` as a context manager:

```{code-cell}
with rasterix.set_options(transform_rtol=1e-9):
    result = xr.concat(dsets, dim="x")
result
```

## Alignment

The same tolerance applies to {py:func}`xarray.align`:

```{code-cell}
aligned = xr.align(*dsets, join="outer")
aligned[0]
```
