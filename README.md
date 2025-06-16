# rasterix: Raster tricks for Xarray

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/dcherian/rasterix/test.yml?branch=main&logo=github&style=flat)](https://github.com/dcherian/rasterix/actions)
[![Documentation Status](https://readthedocs.org/projects/rasterix/badge/?version=latest)](https://rasterix.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/rasterix.svg?style=flat)](https://pypi.org/project/rasterix/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/rasterix.svg?style=flat)](https://anaconda.org/conda-forge/rasterix)

<img src="rasterix.png" width="300">

This WIP project contains tools to make it easier to analyze raster data with Xarray.

The intent is to provide reusable building blocks for the many sub-ecosystems around: e.g. rioxarray, odc-geo, etc.

## Contents

It currently has two pieces.

### 1. RasterIndex

See `src/ rasterix/raster_index.py` and `notebooks/raster_index.ipynb` for a brief demo.

### 2. Dask-aware rasterization wrappers

See `src/rasterix/rasterize.py` for dask-aware wrappers around [`exactextract`](https://github.com/dcherian/rasterix/blob/ec3f51e60e25aa312e6f48c4b22f91bec70413ed/rasterize.py#L165), [`rasterio.features.rasterize`](https://github.com/dcherian/rasterix/blob/ec3f51e60e25aa312e6f48c4b22f91bec70413ed/rasterize.py#L307), and [`rasterio.features.geometry_mask`](https://github.com/dcherian/rasterix/blob/ec3f51e60e25aa312e6f48c4b22f91bec70413ed/rasterize.py#L472).

This code is likely to move elsewhere!

## Installing

### PyPI

`rasterix` alpha releases are available on pypi

```
pip install rasterix
```

## Developing

1. Clone the repo
   ```
   git remote add upstream git@github.com:dcherian/rasterix.git
   cd rasterix
   ```
1. [Install hatch](https://hatch.pypa.io/1.12/install/)
1. Run the tests
   ```
   hatch env run --env test.py3.13 run-pytest  # Run the tests without coverage reports
   hatch env run --env test.py3.13 run-coverage-html   # Run the tests with an html coverage report
   ```
