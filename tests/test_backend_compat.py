"""assign_index on objects as produced by different xarray backends.

Pure-xarray tests mimic the shapes backends return (no GDAL needed); the
gdalxarray round trips at the end only run when gdalxarray is installed.
"""

import numpy as np
import pyproj
import pytest
import xarray as xr
import xproj  # noqa: F401
from affine import Affine

from rasterix import RasterIndex, assign_index
from rasterix.rioxarray_compat import guess_dims


def _regular(dims: tuple[str, str], attrs: tuple[dict, dict] = ({}, {})) -> xr.Dataset:
    ydim, xdim = dims
    return xr.Dataset(
        {"foo": (dims, np.zeros((3, 4)))},
        coords={
            xdim: (xdim, 100.5 + np.arange(4), attrs[1]),
            ydim: (ydim, -40.5 - np.arange(3), attrs[0]),
        },
    )


@pytest.mark.parametrize(
    "dims",
    [("y", "x"), ("latitude", "longitude"), ("Y", "X"), ("lat", "lon")],
)
def test_guess_dims_names(dims):
    assert guess_dims(_regular(dims)) == (dims[1], dims[0])


@pytest.mark.parametrize(
    "yattrs, xattrs",
    [
        ({"axis": "Y"}, {"axis": "X"}),
        ({"standard_name": "projection_y_coordinate"}, {"standard_name": "projection_x_coordinate"}),
        ({"units": "degrees_north"}, {"units": "degrees_east"}),
    ],
)
def test_guess_dims_cf_attrs(yattrs, xattrs):
    assert guess_dims(_regular(("rows", "cols"), (yattrs, xattrs))) == ("cols", "rows")


def test_guess_dims_unknown_raises_valueerror():
    with pytest.raises(ValueError, match="Pass x_dim and y_dim explicitly"):
        guess_dims(_regular(("rows", "cols")))


def test_assign_index_gdal_multidim_names():
    # GDAL's multidim API names a classic raster's dims X/Y, with no CF attrs
    ds = assign_index(_regular(("Y", "X")))
    index = ds.xindexes["X"]
    assert isinstance(index, RasterIndex)
    assert index.transform() == Affine(1, 0, 100, 0, -1, -40)


def _with_index(affine: Affine, crs=None) -> xr.Dataset:
    # The shape a backend returns when it builds the RasterIndex itself
    index = RasterIndex.from_transform(affine, width=4, height=3)
    ds = xr.Dataset({"foo": (("y", "x"), np.zeros((3, 4)))})
    ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
    if crs is not None:
        ds = ds.proj.assign_crs(crs=crs)
    return ds


def test_assign_index_reuses_existing_index_and_attaches_crs():
    affine = Affine(30, 0, 500000, 0, -30, 5300000)
    ds = assign_index(_with_index(affine, crs="EPSG:32755"))
    index = ds.xindexes["x"]
    assert index.transform() == affine
    assert index.crs == pyproj.CRS.from_epsg(32755)


def test_assign_index_reuses_existing_rotated_index():
    # rotated transforms have only 2D coords, which the 1D-coordinate fallback cannot read
    affine = Affine(30, 5, 500000, 4, -30, 5300000)
    ds = assign_index(_with_index(affine, crs="EPSG:32755"))
    raster_indexes = [i for i in ds.xindexes.get_unique() if isinstance(i, RasterIndex)]
    assert len(raster_indexes) == 1
    assert raster_indexes[0].transform() == affine
    assert raster_indexes[0].crs == pyproj.CRS.from_epsg(32755)


# ---------------------------------------------------------------------------
# Round trips through gdalxarray, when available
# ---------------------------------------------------------------------------


@pytest.fixture
def gdal():
    pytest.importorskip("gdalxarray")
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


def _write_tiff(gdal, path, geotransform, epsg=32755):
    from osgeo import osr

    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 3, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.arange(12, dtype="float32").reshape(3, 4))
    ds = None


@pytest.mark.parametrize("geotransform", [(500000, 30, 0, 5300000, 0, -30), (500000, 30, 5, 5300000, 4, -30)])
def test_gdalxarray_classic(gdal, tmp_path, geotransform):
    path = tmp_path / "r.tif"
    _write_tiff(gdal, path, geotransform)
    ds = assign_index(xr.open_dataset(path, engine="gdalxarray", multidim=False))
    (index,) = [i for i in ds.xindexes.get_unique() if isinstance(i, RasterIndex)]
    assert index.transform() == Affine.from_gdal(*geotransform)
    assert index.crs == pyproj.CRS.from_epsg(32755)


def test_gdalxarray_multidim_zarr(gdal, tmp_path):
    geotransform = (500000, 30, 0, 5300000, 0, -30)
    _write_tiff(gdal, tmp_path / "r.tif", geotransform)
    gdal.Translate(str(tmp_path / "r.zarr"), str(tmp_path / "r.tif"), format="Zarr")
    ds = assign_index(xr.open_dataset(tmp_path / "r.zarr", engine="gdalxarray"))
    assert ds.xindexes["X"].transform() == Affine.from_gdal(*geotransform)
