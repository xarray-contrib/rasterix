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
%xmode minimal
import xarray as xr
xr.set_options(display_expand_indexes=True);
```

# Alignment

```{seealso}
See the Xarray tutorial on [Alignment](https://tutorial.xarray.dev/fundamentals/02.3_aligning_data_objects.html#alignment-putting-data-on-the-same-grid).
```

Simple forms of alignment are supported.

Consider this non-overlapping pair of tiles with two transform:

```{code-cell}
transforms = [
    "-50.0 5 0.0 0.0 0.0 -0.25",
    "-40.0 5 0.0 0.0 0.0 -0.25",
]
```

```{code-cell}
---
tags: [hide-input]
---
import pyproj
import numpy as np
import xarray as xr
import rasterix

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
datasets = list(map(rasterix.assign_index, dsets))
datasets[0]
```

```{code-cell}
datasets[1]
```

## `join="outer"`

For outer joins, we reindex so that both datasets are expanded to the _union_ of the input bounding boxes:

```{code-cell}
outer = xr.align(*datasets, join="outer")
outer[0]
```

```{code-cell}
outer[1]
```

## `join="inner"`

For outer joins, we reindex so that both datasets are restricted to the _intersection_ of the input bounding boxes:

There is no overlap in `x` for this set of non-overlapping tiles

```{code-cell}
inner = xr.align(*datasets, join="inner")
inner[0]
```

## `join="exact"`

For exact joins, we compare transforms. Since these are not identical, we see an {py:class}`AlignmentError`.

```{code-cell}
---
raises:
tags: [raises-exception]
---
xr.align(*datasets, join="exact")
```
