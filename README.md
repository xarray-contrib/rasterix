# rasterix: Raster tricks for Xarray

<img src="rasterix.png" width="300">

This WIP project contains tools to make it easier to analyze raster data with Xarray.

The intent is to provide reusable building blocks for the many sub-ecosystems around: e.g. rioxarray, odc-geo, etc.

It currently has two pieces.

## RasterIndex

See `rasterix/raster_index.py` and `notebooks/test_raster_index.ipynb` for a brief demo.

## Dask-aware rasterization wrappers

See `rasterize.py` for dask-aware wrappers around [`exactextract`](https://github.com/dcherian/rasterix/blob/ec3f51e60e25aa312e6f48c4b22f91bec70413ed/rasterize.py#L165), [`rasterio.features.rasterize`](https://github.com/dcherian/rasterix/blob/ec3f51e60e25aa312e6f48c4b22f91bec70413ed/rasterize.py#L307), and [`rasterio.features.geometry_mask`](https://github.com/dcherian/rasterix/blob/ec3f51e60e25aa312e6f48c4b22f91bec70413ed/rasterize.py#L472).

This code is likely to move elsewhere!
