# Object And Overlay Data

Odon can display object and overlay data alongside image channels. This page
describes the file types, path conventions, and coordinate assumptions used by
object-centric workflows.

For the GUI workflow after data is loaded, see
[Objects and Overlays](../workflows/objects-and-overlays.md).

## Supported Inputs

| Input | Typical extension | Geometry | Properties | Best for |
| --- | --- | --- | --- | --- |
| GeoParquet | `.parquet`, `.geoparquet` | Polygons or points | Yes | Rich cell/object review, filtering, measurement, and enriched export. |
| Parquet | `.parquet` | Odon-supported object tables | Yes | Project-linked segmentation/object tables. |
| GeoJSON | `.geojson`, `.json` | Polygons or lines | Limited | Lightweight outlines, simple interchange, and mask-like overlays. |
| CSV points | `.csv` | Points from `x`/`y` columns | Yes | Centroids, landmarks, or point-like QC overlays. |
| OME-Zarr labels | `labels/...` | Label masks rendered as outlines | No tabular object properties | Quick bundled segmentation QC. |
| Mask GeoJSON | `.geojson`, `.json` | Editable polygons | Layer metadata | Artefact, exclusion, and review regions. |

For spatial proteomics projects, prefer GeoParquet or Parquet object files when
you need object properties, cell categories, object filtering, measurements, or
enriched export.

## Project And Samplesheet Linking

The most scalable way to connect object files to ROIs is the samplesheet
`segpath` column:

```csv
id,path,segpath
ROI_001,rois/ROI_001.ome.zarr,objects/ROI_001.cells.parquet
ROI_002,rois/ROI_002.ome.zarr,objects/ROI_002.cells.parquet
```

`path` points to the ROI image dataset. `segpath` points to the segmentation or
object file for that ROI.

Relative `path` and `segpath` values are resolved relative to the samplesheet CSV
file. This makes a project folder portable as long as the relative folder layout
stays the same.

Absolute paths are also accepted, but they are less portable across machines.

For the full samplesheet layout, see
[Projects and Samplesheets](projects-and-samplesheets.md).

## Segmentation Search Roots

If object files are not listed directly in `segpath`, use project search roots:

1. Open or build a project.
2. Click `Add Seg Search Root...`.
3. Choose a folder containing segmentation/object files.
4. Click `Auto-match Seg`.

Odon uses segmentation search roots to find compatible object files for project
ROIs. This is useful when image data and object data live in separate folders.

For large review sessions, use `Object Cache` after object files are matched.
The cache can preload compatible GeoParquet or Parquet files so object-backed
review opens faster across many ROIs.

## Coordinate Assumptions

Object and overlay geometry should be in the same level-0 world coordinate space
as the image being viewed.

Practical expectations:

- polygon and point coordinates should align with the full-resolution image
  plane
- image and object files should refer to the same ROI
- if objects appear offset, check whether a transform, downsample, or coordinate
  conversion has already been applied upstream
- layer move/transform state in Odon affects viewer/project alignment state; it
  does not rewrite the source object file

If object geometry was created from a downsampled image, convert it back to
level-0 coordinates before using it as the canonical object file.

## GeoParquet And Parquet Objects

Use GeoParquet or Parquet when object tables contain many cells or rich
properties.

These formats support the most complete object workflows:

- object outlines and fills
- point-like display when appropriate
- property-driven colouring through `Color by`
- legend visibility controls
- object filters
- object selection
- `Analysis` tab workflows
- `Measurements` tab workflows
- enriched GeoParquet and CSV export

Recommended property columns include:

| Column type | Examples | Use |
| --- | --- | --- |
| Stable object id | `cell_id`, `object_id`, `id` | Tracking objects across exports and reports. |
| Categorical annotations | `cell_type`, `broad_cell_type`, `cluster`, `phenotype` | `Color by`, legend visibility, filtering, review. |
| Numeric measurements | `area`, `eccentricity`, marker intensities | Analysis, threshold/call review, plotting, filtering. |
| QC fields | `quality`, `is_tissue`, `is_border`, `artefact_flag` | Filtering and review subsets. |
| ROI/sample fields | `sample_id`, `roi_id`, `patient_id` | Context after export or concatenation. |

Wide object tables can be loaded lazily. In that case, some property names may
appear with `(load)` in `Color by` or filter controls. Selecting them loads the
column when needed.

## GeoJSON Objects And Masks

GeoJSON is useful for smaller polygon overlays, simple interchange, and masks.

Use GeoJSON for:

- manually authored masks
- artefact regions
- exclusion polygons
- simple segmentation outlines
- exchange with scripts or GIS-like tooling

For masks, use the mask polygon workflow and export mask layers to GeoJSON when
another tool needs the geometry. See
[Mask Polygons](../workflows/mask-polygons.md).

For rich cell tables with many properties, prefer GeoParquet or Parquet over
GeoJSON.

## CSV Point Data

CSV point overlays are useful for centroids, landmarks, and transcript-like
points when polygon geometry is not needed.

A CSV point file should include coordinate columns such as:

```csv
x,y,label
120.5,241.0,point_a
220.0,310.2,point_b
```

Choose the X and Y columns in the load dialog. Other selected columns can be used
as point properties.

Point-only layers are useful for display and selection, but polygon measurements
require polygon object geometry.

## OME-Zarr Labels

OME-Zarr label groups live inside the image dataset under `labels/...`.

Use bundled labels for quick segmentation QC when the labels are part of the
OME-Zarr image package. Odon renders these labels as outlines.

Use project object segmentation instead when you need:

- object properties
- cell category colouring
- filters
- measurements
- enriched export
- deep-link object display state

## Exported Object Data

The `Measurements` tab can attach image-derived measurements back onto loaded
objects. Export options include:

- `Export Enriched GeoParquet...`
- `Export Enriched CSV...`

Enriched exports include measured properties and derived call/selection columns.
GeoParquet keeps geometry. CSV exports tabular columns and is easier to inspect
in spreadsheet-like tools, but it does not preserve polygon geometry.

## Troubleshooting

### Objects Do Not Align With The Image

Check that the object file belongs to the current ROI and uses level-0 image
coordinates. Also check whether a layer transform or object offset is active in
Odon.

### `segpath` Does Not Resolve

For samplesheets, relative `segpath` values are resolved relative to the
samplesheet CSV file. Check the path from that directory, not from the current
terminal directory.

### `Color by` Does Not Show A Column

The column may not be loaded yet. Select the property marked with `(load)`, use
it in a filter row, or verify that the property exists in the object table.

### Measurements Are Unavailable

Measurements require loaded polygon objects. Point-only CSV or point-like
GeoParquet layers cannot be summarized over polygon regions.

## Related Pages

- [Objects and Overlays](../workflows/objects-and-overlays.md)
- [Projects and Samplesheets](projects-and-samplesheets.md)
- [Mosaic Mode](../workflows/mosaic.md)
- [Deep Links](../reference/deep-links.md)
