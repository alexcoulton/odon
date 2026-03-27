# Data Sources Overview

`odon` can open several kinds of spatial imaging and annotation data, but not every source is equally central to the product.

## Primary Data Sources

These are the most mature user workflows:

- OME-Zarr / OME-NGFF imagery
- GeoParquet, Parquet, GeoJSON, and CSV overlays
- Project JSON workspaces
- samplesheet-driven mosaic input

## Secondary Or Compatibility Workflows

These are supported, but should be approached as compatibility workflows rather than the core product center:

- SpatialData containers
- Xenium Explorer datasets
- TIFF / OME-TIFF data

## Choosing The Right Entry Point

- start with [OME-Zarr](ome-zarr.md) for image-first spatial proteomics review
- use [Projects and Samplesheets](projects-and-samplesheets.md) when you need saved workspace state or multi-ROI workflows
- use [SpatialData and Xenium](spatialdata-and-xenium.md) for those specific ecosystem formats
