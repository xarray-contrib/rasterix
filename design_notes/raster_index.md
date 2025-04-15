# Design for a RasterIndex

## TL;DR

1. We propose designing a RasterIndex that can handle many ways of expressing a raster → model space transformation.
1. We propose that this RasterIndex _record_ the information for this transformation (e.g. `GeoTransform`) _internally_ and remove that information from the dataset when constructed.
1. Since the information is recorded internally, a user must intentionally write the transformation back to the dataset, destroying the index while doing so, and then write to disk.

## Goals:

### UX goals:

1. Make it easy to read a GeoTIFF with CRS and raster -> model space transformation information in to Xarray with appropriate indexes. There are at least two indexes: one associated with the CRS; and one with the transformation.
1. The raster ↔ model transformation information can be ambiguous, so an explicit API should be provided.
   1. [RPCs](http://geotiff.maptools.org/rpc_prop.html):
      > The RPC model in a GeoTIFF file is supplementary to all other GeoTIFF tags and not directly related. That is, it is possible to have a conventional set of GeoTIFF tags (such as a tiepoint + pixel scale + projected coordinate system description) along with the RPCCoefficientTag. The RPCCoefficientTag is always describing a transformation to WGS84, regardless of what geographic coordinate system might be described in the coordinate system description tags of the GeoTIFF file. It is also possible to have only the RPCCoefficientTag tag and no other GeoTIFF tags.
   1. [GeoTransform is not in the GeoTIFF standard](https://docs.ogc.org/is/19-008r4/19-008r4.html). Instead that uses ModelTiepointTag, ModelPixelScaleTag, ModelTransformationTag.
   1. [GCPs are ambiguous](https://gdal.org/en/stable/user/raster_data_model.html#gcps):
      > The GDAL data model does not imply a transformation mechanism that must be generated from the GCPs … this is left to the application. However 1st to 5th order polynomials are common.
   1. [And for extra fun](https://gdal.org/en/stable/user/raster_data_model.html#gcps):
      > Normally a dataset will contain either an affine geotransform, GCPs or neither. It is uncommon to have both, and it is undefined which is authoritative.

### Index design goals:

1. The CRS Index allows us to assert CRS compliance during alignment. This is provided by XProj.
1. The transform index should allow us to:
   1. Do accurate alignment in pixel space unaffected by floating-point inaccuracies in model-space;
   1. All possible transforms have **offsets** which means they need to be kept up-to-date during slicing.
   1. Allow extracting metadata necessary to accurately represent the information on disk.

## Some complications

### Handling projections

1. There are at least 5 ways to record a raster ⇒ model space transformation.
1. Information for these transforms may be stored in many places depending on the reading library:
   1. `ds.spatial_ref.attrs`(rioxarray stores GeoTransform, gcps here)
   1. `ds.band_data.attrs` (rioxarray stores TIEPOINTS here)
   1. `ds.attrs` possibly since this would be the most direct way to map TIFF tags
   1. It seems possible to store RPCs as either arrays or as attrs.

We'd like a design that is extensible to handle all 5 (or more) cases; again suggesting an explicit lower-level API.

### Composing with CRSIndex

[One unanswered question so far is](https://github.com/benbovy/xproj/issues/22#issuecomment-2789459387)

> I think the major decision is "who handles the spatial_ref / grid mapping variable". Should it be exclusively handled by xproj.CRSIndex? Or should it be bound to a geospatial index such as rasterix.RasterIndex, xvec.GeometryIndex, xoak.S2PointIndex, etc.?

i.e. does the Index wrap CRSIndex too, or compose with it.
Some points:

1. reprojection requires handling both the transform and the CRS.
1. the EDR-like selection API is similar.

Importantly, GDAL chooses to write GeoTransform to the `grid_mapping` variable in netCDF which _also_ records CRS information.

This gives us two options:

1. RasterIndex wraps CRSIndex too and handles everything.
1. RasterIndex extracts projection information, however it is stored (e.g. GeoTransform), and tracks it internally. Any functionality that relies on both the transformation and CRS will need to be built as an accessor layer.

Below is a proposal for (2).

## Proposal for transform index

We design RasterIndex as a wrapper around **one** of many transform based indexes:

1. AffineTransformIndex ↔ GeoTransform
1. ModelTransformationIndex ↔ ModelTransformationTag
1. ModelTiepointScaleIndex ↔ ModelTiepointTag + ModelPixelScaleTag
1. GCPIndex ↔ Ground Control Points
1. RPCIndex ↔ Rational Polynomial Coefficients
1. Subsampled auxiliary coordinates, detailed in [CF section 8.3](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#compression-by-coordinate-subsampling) and equivalent to GDAL's [geolocation arrays](https://gdal.org/en/stable/development/rfc/rfc4_geolocate.html) with `PIXEL_STEP` and/or LINE_STEP\` > 1.

Each of the wrapped index has an associated transform:

```python
@dataclass
class RasterTransform:
    rpc: RPC | None # rpcs
    tiepoint_scale : ModelTiepointAndScale | None  # ModelTiepointTag, ModelPixelScaleTag
    gcps: GroundControlPoints | None
    transformation : ModelTransformation | None
    geotransform : Affine | None # GeoTransform

    def from_geotransform(attrs: dict) -> Self:
        ...

    def from_tiepoints(attrs: dict) -> Self:
        ...

    def from_gcps(gcps: ?) -> Self:
        ...

    def from_rpcs(?) -> Self:
        ...

    def from_geolocation_arrays(?) -> Self:
        ...
```

### Read-time

These transforms are constructed by **popping** the relevant information from a user-provided source.
This is analogous to an "encode/decode" workflow we currently have in Xarray.

```python
transform = rasterix.RasterTransform.from_geotransform(ds.spatial_ref.attrs)
# transform = rasterix.RasterTransform.from_tiepoints(ds.band_data.attrs)
```

By **popping** the information, the transformation is stored _only_ on the index, and must be rewritten back to the dataset by the user when writing to the disk. This is the RioXarray pattern.

Once a transform is constructed, we could do

```python
    index = RasterIndex.from_transform(transform, dims_and_sizes=...)
    ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
    ds = ds.set_xindex("spatial_ref", xproj.CRSIndex)
```

### Write-time

Before write, we must write the transform (similar to rioxarray) and _destroy_ the RasterIndex instance.
This seems almost required since there are many ways of recording the transformation information on the dataset; and many of these approaches might be used in the same dataset.

```python
    # encodes the internal RasterTransform in attributes of the `to` variable
    # destroys the RasterIndex
    ds = ds.rasterix.write_transform(to: Hashable | rasterix.SELF, formats=["geotransform", "tiepoint"])
    ds.rio.to_raster()
```

Here:

1. `SELF` could mean write to the object (Dataset | DataArray) attrs
1. `formats` allow you to record the same information in multiple ways
1. `.rio.write_transform` could just dispatch to this method.

## Appendix

### Encode/decode workflow for subsampled coordinates

Taking the example 8.3 from [CF section 8.3](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#compression-by-coordinate-subsampling), the decode step may consist in:

1. turn the tie point coordinate variables `lat(tp_yc, tp_xc)` and `lon(tp_yc, tp_xc)` into `lat(yc, xc)` and `lon(yc, xc)` in to Xarray coordinates associated with a custom transformation index that stores only the tie points. In other words, uncompress the dimensions of the `lat` & `lon` coordinates without uncompressing their data.
1. also remove the tie point index variables and the interpolation variable, and track their data / metadata internally in the index

The encode step would then consist in restoring the compressed tie point coordinate & index variables as well as the interpolation variable.
