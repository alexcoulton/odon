# Thresholding

Odon has two threshold-oriented workflows:

- object-property thresholds for marker calls and review
- image-channel threshold regions for creating editable mask polygons

Both are viewer-side review tools. They help you inspect and author calls or
masks in image context, but they are not a replacement for a full downstream
statistics or phenotype logic environment.

## Choose The Right Workflow

| Goal | Use | Requires |
| --- | --- | --- |
| Turn measured marker intensities into positive/negative or custom calls | `Analysis` tab `Calls` and `Thresholds` | Loaded object geometry with numeric object properties. |
| Review object-property distributions and select cells above or below a value | `Analysis` tab histogram/scatter tools | Loaded object geometry with numeric object properties. |
| Create a mask from bright pixels in the active image channel | Image-channel `Properties` > `Threshold Regions` | An image channel in XY view. |
| Export measured marker values or call columns | `Measurements` export workflows | Loaded polygon objects and selected image channels. |

For object formats and `segpath` conventions, see
[Object and Overlay Data](../data-sources/object-and-overlay-data.md).

## Object-Property Thresholds

Use object-property thresholds when you already have object data, such as cell
segmentation polygons with marker intensity columns.

This workflow is best for:

- reviewing marker distributions
- setting marker-specific calls
- creating reusable call presets
- selecting objects that meet one or more threshold rules
- checking calls against the image signal

### Prepare The Data

1. Open one ROI.
2. Load segmentation objects from GeoParquet, Parquet, or another supported
   object source.
3. Make sure the object table contains numeric marker or measurement columns.
4. Select the relevant image channel in the channel list.
5. Open the right-panel `Analysis` tab.

If the object file does not already contain marker intensities, use the
`Measurements` tab to measure image channels over polygon objects first, then
export or continue with the enriched object data.

## Calls And Marker Binding

In the `Analysis` tab, the `Calls` section turns measured marker intensities
into reusable boolean call definitions.

Useful controls:

| Control | Purpose |
| --- | --- |
| `Preset name` | Name for the current set of call definitions. |
| `Save Preset...` | Save call definitions to a JSON preset. |
| `Load Preset...` | Load an existing call preset. |
| `Bind edits to active marker` | Keep threshold edits associated with the currently selected image channel/marker. |
| `Mapping settings...` | Review or override how image channel names map to object-property columns. |
| `New call for active marker` | Create a marker-specific call for the selected channel. |
| `New composite call` | Create a call that can combine multiple threshold rules. |
| `Mark failed` | Mark a marker call as failed for export/review purposes. |

Call presets save call definitions only. Measurement results are exported from
the `Measurements` tab.

## Threshold Rules

The `Thresholds` controls define the rules used by the selected call.

A threshold rule has:

- a numeric object-property column
- an operator, either `>=` or `<=`
- a threshold value

Use `Add threshold` when a call needs more than one rule.

When `Bind edits to active marker` is enabled, Odon tries to align the active
image channel with the most relevant numeric object column. If the automatic
mapping is wrong, open `Mapping settings...` and choose the correct column.

## Histogram Review

The `Analysis` tab can show object-property histograms for numeric columns.

Use the histogram workflow to:

- inspect the distribution of marker values
- compare raw and `arcsinh` views
- drag or edit threshold values
- snap thresholds to suggested levels
- use `Quantiles` or `K-means` level suggestions
- update the object selection from threshold rules

Plots use the currently filtered object set when object filters are active.
Otherwise, plots use all loaded objects.

The selection overlay can show which objects currently match the active
threshold rules. For large object sets, live selection may be disabled or less
interactive; apply selection deliberately after adjusting rules.

## Image-Channel Threshold Regions

`Threshold Regions` is a separate workflow under an active image channel's
right-panel `Properties` tab.

Use it to create editable mask polygons from pixels above a chosen image
threshold.

This workflow is useful for:

- rapid review masks
- artefact regions
- exclusion regions
- rough tissue or signal-positive regions

### Create A Threshold Region Mask

1. Select an image channel.
2. Make sure the viewer is in XY view.
3. Open the right-panel `Properties` tab.
4. Expand `Threshold Regions` > `Controls`.
5. Choose `Visible region` or `Entire image`.
6. For `Entire image`, choose a safe pyramid level.
7. Click `Start threshold preview from visible region` or
   `Start threshold preview from entire image`.
8. Adjust `Threshold` and `Min component pixels`.
9. Click `Refresh preview` after changing the viewed area, level, or channel.
10. Click `Apply mask from preview`.

The preview is a raster overlay on the canvas. Applying the preview converts it
into editable mask polygons.

For mask editing after applying the preview, see
[Mask Polygons](mask-polygons.md).

## Visible Region Versus Entire Image

Use `Visible region` for interactive review. It thresholds the current canvas
area at the current viewer level.

Use `Entire image` only when you need a mask across a selected pyramid level.
Odon shows the preview size and disables levels that are too large. Lower-numbered
levels are usually higher resolution and more expensive; higher-numbered levels
are coarser and safer for broad masks.

For large images, start with a coarse level and inspect the result before trying
finer levels.

## Saving And Exporting Results

Different threshold workflows produce different outputs:

| Workflow | Output |
| --- | --- |
| Object-property calls | Call definitions in Odon state or a call preset JSON. |
| Object-property threshold selection | Active object selection/review state. |
| Measurements | Enriched GeoParquet or CSV with measured values and derived columns. |
| Threshold regions | Editable mask polygons that can be saved with the project or exported through mask workflows. |

If you need a durable table for downstream analysis, use the measurement or
object export workflows rather than relying only on transient viewer selection.

## Practical Review Loop

A typical marker-call review session is:

1. Open an ROI from a project or samplesheet.
2. Load object segmentation for that ROI.
3. Measure marker intensities if the object table does not already contain them.
4. Select the marker image channel.
5. Open `Analysis`.
6. Use `Bind edits to active marker`.
7. Confirm the channel-to-column mapping.
8. Adjust the histogram threshold and inspect selected objects over the image.
9. Save a call preset or export enriched object data as needed.
10. Move to the next marker or ROI.

## Troubleshooting

### The Analysis Tab Does Not Show Threshold Controls

Load a compatible object layer first. Object-property thresholding needs object
geometry plus numeric object columns.

### The Wrong Object Column Is Used For A Marker

Open `Mapping settings...` in the `Calls` section and manually map the image
channel to the correct numeric object-property column.

### A Column Is Missing

Check that the object table contains the property and that it is numeric. If the
object source uses lazy property loading, select or use the column so Odon can
load it.

### Threshold Region Preview Is Unavailable

Select an image channel and switch to XY view. Threshold-region preview is only
available for image channels in XY view.

### Entire-Image Thresholding Is Disabled

The selected level is too large for interactive thresholding. Choose a coarser
pyramid level.

### Preview Looks Stale

Click `Refresh preview` after panning, zooming, changing channel, changing
threshold scope, or changing the selected pyramid level.

### Generated Mask Polygons Need Cleanup

Apply the preview, then use the mask polygon tools to edit, delete, or refine
the generated polygons.

## Related Pages

- [Viewing Channels](viewing.md)
- [Objects and Overlays](objects-and-overlays.md)
- [Object and Overlay Data](../data-sources/object-and-overlay-data.md)
- [Mask Polygons](mask-polygons.md)
- [Projects and Samplesheets](../data-sources/projects-and-samplesheets.md)
- [Current Limitations](../advanced/current-limitations.md)
