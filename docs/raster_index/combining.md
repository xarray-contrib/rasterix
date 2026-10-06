---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python
---

```{eval-rst}
.. currentmodule:: rasterix.raster_index
```

```{code-cell} python
---
tags: [remove-cell]
---
import xarray as xr
xr.set_options(display_expand_indexes=True);
```

# Combining

{py:class}`RasterIndex` supports concatenation along a single axis through either {py:func}`xarray.concat` or across multiple axes using {py:func}`xarray.combine_nested`.
In all cases, a new {py:class}`RasterIndex` is created.

Cases (a) and (b) in the following image are supported, case (c) is not.

```{image} images/combining-schematic.png
---
alt: Schematic of different combining options
width: 70%
align: center
---
```

## `concat`

Here are [GeoTransform](https://gdal.org/en/stable/tutorials/geotransforms_tut.html) attributes for three tiles separated by 2 pixels in the X direction:

```{code-cell}
transforms = [
    "-50.0 5 0.0 0.0 0.0 -0.25",
    "-40.0 5 0.0 0.0 0.0 -0.25",
    "-30.0 5 0.0 0.0 0.0 -0.25",
]
```

To illustrate we'll create 3 datasets

```{code-cell}
import pyproj
import numpy as np
import xarray as xr

dsets = [
    (i+1) * xr.Dataset(
        {"foo": (("y", "x"), np.ones((4, 2)), {"grid_mapping": "spatial_ref"})},
        coords={
            "spatial_ref": (
                (),
                0,
                pyproj.CRS.from_epsg(4326).to_cf() | {"GeoTransform": transform},
            )
        },
    )
    for i, transform in enumerate(transforms)
]
dsets[0]
```

Now we assign RasterIndex to all three datasets using {py:func}`assign_index`

```{code-cell}
import rasterix

dsets = tuple(map(rasterix.assign_index, dsets))
dsets[0]
```

And... concatenate! 🪄

```{code-cell}
xr.concat(dsets, dim="x")
```

(concat-periodic)=

### Periodic axes

Concatenation along a periodic axis is supported (see {doc}`periodic`).
The offsets of the concatenated datasets can then differ by a multiple of the period.
{py:class}`RasterIndex` moves each piece by that multiple, so the result is contiguous
and its coordinates are _unwrapped_.

```{code-cell}
raw = xr.Dataset(
    {"foo": (("y", "x"), np.ones((2, 12)), {"grid_mapping": "spatial_ref"})},
    coords={
        "spatial_ref": (
            (),
            0,
            pyproj.CRS.from_epsg(4326).to_cf() | {"GeoTransform": "-180 30 0 90 0 -90"},
        )
    },
)
# copy: assign_index removes GeoTransform from its input
global_ds = rasterix.assign_index(raw.copy(), x_period=360)
tail, head = global_ds.isel(x=slice(-2, None)), global_ds.isel(x=slice(0, 2))
xr.concat([tail, head], dim="x")
```

Without a period, the same concatenation raises an error:

```{code-cell}
---
tags: [raises-exception]
---
no_period = rasterix.assign_index(raw)
xr.concat([no_period.isel(x=slice(-2, None)), no_period.isel(x=slice(0, 2))], dim="x")
```

<!-- Concatenation is supported both for increasing and decreasing y-axis coordinates -->

<!-- ```{code-cell} -->

<!-- reversed_datasets = [ds.isel(y=slice(None, None, -1)) for ds in dsets] -->

<!-- xr.concat(reversed_datasets, dim="x") -->

<!-- ``` -->

## `combine_nested`

Xarray supports n-dimensional concatenation through the {py:func}`xarray.combine_nested` API.
Here we use that to do two dimension combination:

We define a 2x3 tiling grid with these GeoTransforms:

```{code-cell}
transforms = [
    # row 1
    "-50.0 5 0.0 0.0 0.0 -0.25",
    "-40.0 5 0.0 0.0 0.0 -0.25",
    "-30.0 5 0.0 0.0 0.0 -0.25",
    # row 2
    "-50.0 5 0.0 -1 0.0 -0.25",
    "-40.0 5 0.0 -1 0.0 -0.25",
    "-30.0 5 0.0 -1 0.0 -0.25",
]
```

Again, construct datasets

```{code-cell}
dsets = [
    (i+1) * xr.Dataset(
        {"foo": (("y", "x"), np.ones((4, 2)), {"grid_mapping": "spatial_ref"})},
        coords={
            "spatial_ref": (
                (),
                0,
                pyproj.CRS.from_epsg(4326).to_cf() | {"GeoTransform": transform},
            )
        },
    )
    for i, transform in enumerate(transforms)
]
dsets = tuple(map(rasterix.assign_index, dsets))
```

And.. combine 🪄!

```{code-cell}
xr.combine_nested([dsets[:3], dsets[3:]], concat_dim=["y", "x"])
```
