# rasterix: Raster tricks for Xarray

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/xarray-contrib/rasterix/test.yml?branch=main&logo=github&style=flat)](https://github.com/xarray-contrib/rasterix/actions)
[![Documentation Status](https://readthedocs.org/projects/rasterix/badge/?version=latest)](https://rasterix.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/rasterix.svg?style=flat)](https://pypi.org/project/rasterix/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/rasterix.svg?style=flat)](https://anaconda.org/conda-forge/rasterix)

<img src="_static/rasterix.png" width="300">

This project contains tools to make it easier to analyze raster data with Xarray.
It currently has two pieces.

1. `RasterIndex` for indexing using the affine transform recorded in GeoTIFFs.
1. Dask-aware rasterization wrappers around `exactextract`, `rasterio.features.rasterize`, and `rasterio.features.geometry_mask`.

Our intent is to provide reusable building blocks for the many sub-ecosystems around: e.g. `rioxarray`, `odc.geo`, etc.

## Installing

`rasterix` releases are available on pypi

```
pip install rasterix
```

## Building Documentation Locally

Install the package with documentation dependencies:

```
pip install -e ".[docs]"
```

Build the HTML docs:

```
cd docs
make html
```

Open `docs/_build/html/index.html` in your browser to view the result.

For a live-reload development server that rebuilds on changes:

```
cd docs
make livehtml
```

To also open the docs in your browser automatically:

```
cd docs
make livehtml SPHINXOPTS="--open-browser"
```
