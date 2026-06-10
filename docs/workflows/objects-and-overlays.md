# Objects and Overlays

Odon can draw more than image channels. You can layer segmentation labels,
polygon objects, point objects, masks, and annotation data over the image canvas.
These overlays are used for visual QC, cell review, object filtering,
measurement, and downstream export.

## Overlay Types

| Overlay type | Typical source | Best for |
| --- | --- | --- |
| NGFF labels | OME-Zarr `labels/...` groups | Quick segmentation-outline review when labels are bundled with the image. |
| Segmentation objects | GeoParquet or Parquet | Cell/object review, property colouring, filtering, measurements, and enriched export. |
| Segmentation GeoJSON | GeoJSON or JSON | Lightweight outline overlays and simple interchange. |
| Masks | GeoJSON or project mask state | Exclusion regions, artefact marking, and manual review regions. |
| Points | CSV, GeoParquet points, SpatialData, Xenium adapters | Centroids, transcripts, landmarks, and other point-like data. |

For spatial proteomics review, the most useful object workflow is usually
GeoParquet or Parquet segmentation objects linked to each ROI through a project
or samplesheet `segpath`. For file-format details, see
[Object and Overlay Data](../data-sources/object-and-overlay-data.md).

## Open Objects From The GUI

Start from a single image or project ROI:

1. Open Odon and open one ROI.
2. In the left layer list, select `Segmentation Objects`.
3. In the right-panel `Properties` tab, click `Load Seg Objects...`.
4. Choose a GeoParquet or Parquet object file.
5. In the load dialog, choose whether to load polygon geometry or point-like
   objects when that choice is available.
6. Confirm the load and wait for the object count/status to update.

After loading, keep `Segmentation Objects` selected in the left layer list. The
right `Properties`, `Analysis`, and `Measurements` tabs then operate on that
object source.

## Project-Linked Objects

For repeated review, link each ROI to its object file through project metadata
instead of loading files one at a time.

The recommended samplesheet pattern is:

```csv
id,path,segpath
ROI_001,rois/ROI_001.ome.zarr,objects/ROI_001.cells.parquet
ROI_002,rois/ROI_002.ome.zarr,objects/ROI_002.cells.parquet
```

When a project ROI has `segpath`, Odon can find the matching object file during
single-ROI review, project navigation, mosaic review, object cache preloading,
and deep-link workflows.

Useful project tools:

- `Import Samplesheet CSV...` imports `segpath` metadata.
- `Add Seg Search Root...` adds folders where Odon should look for object files.
- `Auto-match Seg` attempts to match project ROIs to segmentation/object files.
- `Object Cache` can preload compatible GeoParquet or Parquet object files for
  faster review across many ROIs.

For the samplesheet format, see
[Projects and Samplesheets](../data-sources/projects-and-samplesheets.md). For
object-file conventions, see
[Object and Overlay Data](../data-sources/object-and-overlay-data.md).

## Segmentation GeoJSON

Use `Load Seg GeoJSON...` for simpler GeoJSON or JSON segmentation outlines:

1. Select the segmentation GeoJSON layer in the layer list.
2. In the right-panel `Properties` tab, click `Load Seg GeoJSON...`.
3. Choose the GeoJSON file.
4. Adjust `Visible`, `Opacity`, and `Width`.

GeoJSON is useful for lightweight outline display. For richer object properties,
colouring, filtering, measurement, and enriched export, prefer GeoParquet or
Parquet segmentation objects.

## Bundled NGFF Labels

If an OME-Zarr dataset includes label groups under `labels/`, Odon can render
those label layers as outlines.

Useful notes:

- label groups are discovered from the dataset rather than assumed to be called
  `cells`
- bundled labels are good for quick segmentation QC
- project object segmentation is usually better when you need object properties,
  cell categories, filtering, and measurements
- label rendering currently focuses on outlines rather than filled masks

## Display Controls

Select the object or overlay layer in the left panel, then use the right-panel
`Properties` tab.

For `Segmentation Objects`, common controls include:

- `Visible`
- `Opacity`
- `Width`
- `Fast rendering`
- `Fill cells`
- `Fill opacity`
- `Selected fill`
- `Color`
- `Color by`
- `Reload`
- `Clear`

Use `Color by` to colour objects by a categorical property such as cell type,
cluster, phenotype, or call state. If the object file uses lazy property loading,
some properties may appear with `(load)` and are loaded when selected.

When colouring by property, the legend appears below the filter controls. Legend
checkboxes show or hide specific category values without changing the underlying
object file.

## Filtering Objects

The `Filter` section in `Segmentation Objects` properties changes the active
object subset.

Typical workflow:

1. Select `Segmentation Objects`.
2. Open the right-panel `Properties` tab.
3. In `Filter`, choose a property column.
4. Choose or type a query value.
5. Enable the filter row.
6. Add more rows with `+` if needed.

Filters are combined with AND. The `Visible after filter` line shows how many
objects remain active.

Use filters when you want analysis, selection, rendering, or measurements to
operate on a subset of objects. Use legend checkboxes when you only want to hide
or show categories for display.

## Selecting Objects

Object selection is available when an object-backed layer is active.

Useful tools:

- `Pan`: click objects to select them while still using empty canvas for panning.
- Rect select: drag a rectangle to select cells by centroid.
- Lasso select: draw a freehand polygon to select cells by centroid.

Selections feed object review and analysis workflows. They do not edit source
segmentation geometry.

## Measurements

The `Measurements` tab summarizes image signal over loaded polygon objects.

Typical workflow:

1. Load polygon segmentation objects.
2. Open the right `Measurements` tab.
3. Choose `Mean` or `Median (exact)`.
4. Choose the image pyramid `Level`.
5. Optionally enable `Filtered cells only`.
6. Set a `Column prefix`.
7. Click `Measure mean intensities` or `Measure median intensities`.

Measurement results are attached back onto the loaded objects as numeric
properties, so they become available in `Analysis` immediately.

Use `Export Enriched GeoParquet...` or `Export Enriched CSV...` to save measured
properties, derived calls, and selection columns for use outside Odon.

Median measurements are exact but can use substantially more memory on large
datasets because they need per-channel pixel buffers. Prefer coarser levels or
mean measurements when doing a fast QC pass.

## Analysis

The `Analysis` tab is available for loaded segmentation objects and object-backed
SpatialData shape layers.

Use it to:

- review numeric object-property histograms
- create threshold/call review state
- compare object properties with image channels
- map object properties to channel names
- zoom to selected objects

Analysis needs object geometry plus object properties. Image channels alone are
not enough.

## Masks

Masks are editable review regions, not cell/object segmentations.

Use masks for:

- artefact regions
- exclusion regions
- manually drawn review areas
- threshold-created region cleanup

Mask polygon creation and editing are covered in
[Mask Polygons](mask-polygons.md).

## Recommended Review Workflow

1. Open a project ROI.
2. Load or confirm the project-linked segmentation objects.
3. Select `Segmentation Objects` in the layer list.
4. Use `Visible`, `Opacity`, `Width`, and `Fill cells` to make objects readable.
5. Use `Color by` to review a categorical annotation.
6. Use `Filter` to focus on a biological or QC subset.
7. Use `Analysis` for property review and threshold/call checks.
8. Use `Measurements` when you need image-intensity summaries over objects.
9. Export enriched GeoParquet or CSV when results need to leave Odon.

## Troubleshooting

### Object Controls Are Missing

Click `Segmentation Objects` or another object-backed layer in the left layer
list. The right-panel `Properties` tab changes based on the active layer.

### No Objects Load

Check that the file exists and is readable. For project workflows, check that
`segpath` is present, relative paths resolve from the samplesheet or project
location, and segmentation search roots point to the right folders.

### A Property Is Missing From `Color by` Or `Filter`

If the object file uses lazy property loading, select the property marked with
`(load)` or choose it in a filter row so Odon loads that column. If it still does
not appear, verify that the property exists in the object file.

### Measurements Are Disabled

Measurements require loaded polygon objects. Point-only object layers and
unloaded object sources cannot be measured over polygon regions.

### Results Look Offset

Check image/object alignment, layer move/transform state, object offsets, and
whether the object file belongs to the current ROI.

## Related Pages

- [Projects and Samplesheets](../data-sources/projects-and-samplesheets.md)
- [Object and Overlay Data](../data-sources/object-and-overlay-data.md)
- [Mask Polygons](mask-polygons.md)
- [Deep Links](../reference/deep-links.md)
- [Odon MCP](../reference/codex-mcp-odon.md)
