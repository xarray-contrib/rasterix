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

# Periodic axes

{py:class}`RasterIndex` accepts `x_period` (or `y_period`), in
{py:func}`assign_index` or {py:meth}`RasterIndex.from_transform` to indicate periodic axes:

```python
ds = rasterix.assign_index(ds, x_period=360)
```

rasterix does _not_ infer the period from the CRS (see {doc}`design_choices`).

```{code-cell}
---
tags: [hide-cell]
---
import numpy as np
import xarray as xr
from affine import Affine
from rasterix import RasterIndex
```

```{code-cell}
index = RasterIndex.from_transform(
    Affine(30, 0, -180, 0, -30, 90), width=12, height=2, x_period=360
)
da = xr.DataArray(
    np.broadcast_to(np.arange(12), (2, 12)),
    dims=("y", "x"),
    coords=xr.Coordinates.from_xindex(index),
    name="column",
)
da
```

Read the period back with {py:attr}`RasterIndex.x_period` and {py:attr}`RasterIndex.y_period`:

```{code-cell}
index.x_period, index.y_period
```

## Requirements

`period / |dx|` must be an integer number of cells, to within 1e-3 cell by default
(set `period_atol` / `period_rtol` with {py:func}`rasterix.set_options`).
For example, 360 / 0.7 is not:

```{code-cell}
---
tags: [raises-exception]
---
RasterIndex.from_transform(
    Affine(0.7, 0, -180, 0, -1, 90), width=514, height=2, x_period=360
)
```

The axis must not have more cells than one period. A grid made with
`np.linspace(-180, 180, n)` has a copy of the first column at the end; drop it first with
`ds.isel(x=slice(0, -1))`:

```{code-cell}
---
tags: [raises-exception]
---
RasterIndex.from_transform(
    Affine(30, 0, -195, 0, -30, 90), width=13, height=2, x_period=360
)
```

## Behavior

All rules work in cell positions, so they also apply to a decreasing axis (`dx < 0`).
On a decreasing axis, write slices in decreasing label order, as usual in Xarray.

### Labels outside the extent wrap

If a label is in the extent i.e. ≤ `period`, it is used as-is.
If not, the label is wrapped by the period.
Thus −100° and 260° give the same column:

```{code-cell}
da.sel(x=-100, method="nearest").values, da.sel(x=260, method="nearest").values
```

### Slices across the period seam

#### `start` < `stop`

When slicing across the period seam (e.g. the anti-meridian),
`start` is wrapped and the width `stop - start` is kept. The cells come from both ends of the
array, and the coordinates of the result are _unwrapped and contiguous_.

For example, on a global longitude axis, a slice that starts left of the extent, say -180,
returns coordianates in that frame.
Below, on a 1° grid from −180° to 180°, `slice(-190, -170)` gives coordinates
−189.5 … −169.5.

```{code-cell}
global_index = RasterIndex.from_transform(
    Affine(1, 0, -180, 0, -1, 90), width=360, height=1, x_period=360
)
global_da = xr.DataArray(
    np.arange(360)[np.newaxis],
    dims=("y", "x"),
    coords=xr.Coordinates.from_xindex(global_index),
)
global_da.sel(x=slice(-190, -170))
```

#### `stop < start`

Behaviour is similar at the other end of the axis:

```{code-cell}
da.sel(x=slice(150, -150))
```

### A slice that contains the full extent is not rolled

We never duplicate data of the array by _slicing_.

```{code-cell}
da.sel(x=slice(-200, 200)).x.values
```

### Period is preserved

`da.sel(x=slice(120, 210))` has coordinates 135° … 225° and preserves `x_period`.

```{code-cell}
subset = da.sel(x=slice(120, 210))
subset
```

So −160° (the same as 200°) still selects column 0:

```{code-cell}
subset.sel(x=-160, method="nearest")
```

### Alignment and arithmetic need the same frame

Alignment compares labels exactly; it does not wrap them.
`subset` has labels 195 and 225, which are the same cells as −165 and −135 in `da`.
Exact matching would drop these cells (inner join) or duplicate them (outer join),
so alignment raises an error. Use the same frame on both sides before you combine.

```{code-cell}
---
tags: [raises-exception]
---
da + subset
```

### Concatenation across the seam

The end and the start of a periodic axis concatenate to a contiguous result.
See {ref}`concat-periodic` for the rule.

```{code-cell}
xr.concat(
    [da.isel(x=slice(-2, None)), da.isel(x=slice(0, 2))], dim="x"
)
```

### Indexes with different periods do not combine

```{code-cell}
---
tags: [raises-exception]
---
no_period = da.copy().assign_coords(
    xr.Coordinates.from_xindex(
        RasterIndex.from_transform(
            Affine(30, 0, -180, 0, -30, 90), width=12, height=2
        )
    )
)
xr.align(da, no_period, join="outer")
```
