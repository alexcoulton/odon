# Deep Links

Odon deep links let external reports and analysis notebooks open a specific
project ROI with a specific viewer state. They are useful when an HTML report,
Quarto document, RMarkdown output, spreadsheet, or dashboard should link back to
the raw image context behind a result.

A deep link uses the `odon://open` scheme:

```text
odon://open?project=file:///path/to/odon.project.json&roi=ROI_001
```

Deep links can set the active ROI, visible channels, channel colours, contrast
windows, object segmentation source, object colouring, object filters, legend
visibility, and camera position.

## Quick Start

In reports and dashboards, use normal links whose `href` is an `odon://open`
URL:

```html
<a href="odon://open?project=file:///path/to/odon.project.json&roi=ROI_001">
  Open ROI_001 in Odon
</a>
```

Open a project ROI, activate CD45, and show only CD45 and DAPI:

```text
odon://open?project=file:///path/to/odon.project.json&roi=ROI_001&channel=CD45&visible_channels=CD45%7CDAPI
```

Open a project ROI, load project object segmentation, colour cells by an
annotation column, and fill cells:

```text
odon://open?project=file:///path/to/odon.project.json&roi=ROI_001&segmentation_source=geoparquet&load_labels=0&cell_color_by=broad_cell_type&fill_cells=1
```

Clickable `odon://` links require operating-system URL scheme registration; see
[Clickable Links](#clickable-links). Command-line deep links are useful for
testing and development, but they are not the primary report workflow.

!!! tip "Test Deep Links"

    Use the [Odon deep-link test page](../assets/odon-deep-link-test.html) to
    check whether `odon://` links are registered on your system. The page
    includes a smoke-test link, an installed synthetic example link, and a
    small form for generating project-specific links from a local project JSON
    path and ROI identifier.

    [Open the deep-link test page](../assets/odon-deep-link-test.html){ .md-button }

Packaged builds also include an installed synthetic example. To open it without
knowing the install path, use:

```text
odon://open?example=synthetic_5ch
```

## Link Format

Use the `odon://open` action with URL query parameters:

```text
odon://open?project=file:///path/to/project.json&roi=ROI_001&channel=CD45
```

Values should be URL-encoded when they contain spaces, slashes, pipes, commas,
hashes, equals signs, or other reserved URL characters.

Common encodings:

| Character | Encoded form |
| --- | --- |
| space | `%20` |
| `/` | `%2F` |
| `|` | `%7C` |
| `#` | `%23` |
| `=` | `%3D` |

For example, `Sample A/ROI 2` becomes `Sample%20A%2FROI%202`.

List-like parameters accept values separated by commas, pipes, or semicolons.
Pipes are often easiest to read after URL encoding:

```text
visible_channels=CD45%7CCD3%7CDAPI
```

## Opening Methods

### Clickable Links

Clickable `odon://` links are the intended user-facing workflow for reports and
dashboards. They require the operating system to know which application handles
the `odon` URL scheme.

For packaged macOS and Windows builds, the installer or app bundle registers
the `odon://` scheme with the operating system. Open the app once after
installation if the operating system has not yet associated the scheme with
Odon.

When Odon is already running, a clicked deep link starts a short-lived Odon
process that forwards the URL to the existing window over a local loopback
listener on `127.0.0.1:17871`, then exits. This listener is separate from the
MCP control bridge on `127.0.0.1:17870`.

For macOS development builds, register the helper app:

```bash
./scripts/register-macos-url-handler.sh
```

The helper registers `odon://`, logs received URLs, forwards URLs to a running
Odon process through the local development socket, and falls back to:

```bash
cargo run -- '<clicked odon URL>'
```

Test a clicked link on macOS with:

```bash
open 'odon://open?project=file:///path/to/odon.project.json&roi=ROI_001'
```

Linux can use deep links as startup arguments. Clickable URL scheme
registration on Linux depends on desktop integration for the target environment.

### Command Line Testing

Command-line deep links are useful for development, automated tests, and
debugging:

```bash
cargo run -- 'odon://open?project=file:///path/to/project.json&roi=ROI_001'
```

If Odon is already running, the new process attempts to forward the deep link to
the existing Odon window through local single-instance IPC.

## Project and ROI Parameters

| Parameter | Meaning |
| --- | --- |
| `example`, `demo`, `example_dataset` | Installed example dataset alias. Use `synthetic_5ch` to open the packaged synthetic 5-channel example. |
| `project`, `project_path` | Project JSON path. `file:///...` URLs and plain local paths are accepted. |
| `roi`, `roi_id` | ROI id, display name, path fragment, or source-key fragment. |
| `sample`, `case`, `dataset_id` | Optional disambiguation value when multiple ROIs match the same `roi`. |

When `example=synthetic_5ch` is used without a `project` value, Odon searches
its installed examples folder and applies sensible defaults: ROI
`synthetic_5ch.ome.zarr`, active channel `DAPI`, and visible channels `DAPI`,
`CD3`, and `PanCK`.

If the requested project is not already loaded, Odon loads it before resolving
the ROI. If the project is already open, Odon reuses it.

ROI matching is intentionally flexible so report generators can link by stable
IDs, display names, or path fragments. If a link matches multiple ROIs, add a
`sample` value or use a more specific `roi` value.

## Channel Parameters

| Parameter | Meaning |
| --- | --- |
| `channel`, `marker` | Active image channel. Matching is case-insensitive and marker-like names are supported. |
| `visible_channels`, `show_channels`, `only_channels` | Show only these channels and hide all others. |
| `hidden_channels`, `hide_channels` | Hide these channels while leaving other channel visibility unchanged. |
| `channel_order`, `channels_order`, `channel_sort` | Use `listed` to move resolved visible channels to the top of the channel list. |
| `order_visible_channels`, `order_listed_channels` | Boolean aliases for `channel_order=listed`. |
| `group_visible_channels`, `group_channels`, `group_visible` | If true, create or reuse a channel group for the visible channels. |
| `visible_channel_group`, `channel_group`, `group_name` | Name for the visible-channel group. Providing a name also enables grouping. |
| `visible_channel_group_color`, `channel_group_color`, `group_color` | Colour for the visible-channel group. |
| `channel_color`, `channel_colors`, `channel_colour`, `channel_colours` | Per-channel colours as `channel:colour`, separated by `|` or `;`. |

Examples:

```text
channel=CD45
visible_channels=CD45%7CCD3%7CDAPI
hidden_channels=Autofluorescence
visible_channels=CD3%7CCD4%7CCD8&channel_order=listed
visible_channels=CD3%7CCD4%7CCD8&visible_channel_group=T%20cells
channel_color=CD3:red%7CCD4:green%7CCD8:blue
```

Colour values accept hex colours such as `%23ff3366`, `ff3366`, or `f36`, and
aliases such as `red`, `green`, `blue`, `cyan`, `magenta`, `yellow`, `orange`,
`purple`, `pink`, `lime`, `grey`, `gray`, `white`, and `black`.

## Contrast Parameters

| Parameter | Meaning |
| --- | --- |
| `contrast_min`, `channel_min`, `window_min` | Lower contrast limit for the active channel. |
| `contrast_max`, `channel_max`, `window_max` | Upper contrast limit for the active channel. |
| `channel_contrast`, `channel_contrasts`, `channel_window`, `channel_windows` | Per-channel contrast windows as `Channel:min:max`, separated by `|` or `;`. |

Examples:

```text
channel=CD45&contrast_min=0&contrast_max=1200
channel_contrast=CD45:0:1200%7CCD3:20:900
```

Contrast limits are applied as manual contrast windows, equivalent to editing
contrast in the UI.

## Segmentation and Object Parameters

| Parameter | Meaning |
| --- | --- |
| `segmentation`, `label`, `labels` | Bundled OME-Zarr label group name, for example `cells`. |
| `segmentation_source`, `segmentation_layer`, `segmentation_kind` | Segmentation source. Use `geoparquet`, `parquet`, `objects`, or `project` for project object segmentation. Use `labels` or omit the value for bundled OME-Zarr labels. |
| `load_labels`, `load_segmentation_labels`, `load_ome_zarr_labels`, `load_bundled_labels` | Whether to load bundled OME-Zarr labels. Use `0` or `false` to skip them. |
| `cell_color_by`, `color_by`, `object_color_by` | Object property used to colour loaded segmentation objects. |
| `fill_cells` | Fill object polygons. Accepts `1`, `true`, `0`, or `false`. |
| `show_selection_overlay`, `selection_overlay` | Show or hide the object selection overlay. |

For report links that should colour cells by an annotation column, prefer project
object segmentation:

```text
segmentation_source=geoparquet&load_labels=0&cell_color_by=broad_cell_type
```

This opens the image, loads the project GeoParquet or Parquet object
segmentation from the ROI's `segpath`, avoids loading bundled OME-Zarr labels,
and applies the requested object colouring.

For supported object formats and `segpath` conventions, see
[Object and Overlay Data](../data-sources/object-and-overlay-data.md).

## Legend Visibility And Colours

These parameters apply after `cell_color_by` has selected an object property.

| Parameter | Meaning |
| --- | --- |
| `visible_cell_types`, `show_cell_types`, `only_cell_types`, `cell_types` | Show only these category values and hide the rest. |
| `hidden_cell_types`, `hide_cell_types` | Hide these category values while leaving the rest visible. |
| `object_level_colors`, `object_level_colours`, `level_colors`, `level_colours`, `cell_type_colors`, `cell_type_colours`, `category_colors`, `category_colours` | Set colours for category values as `value:colour`, separated by `|` or `;`. |

Examples:

```text
cell_color_by=broad_cell_type&visible_cell_types=tumor%7Cimmune
cell_color_by=broad_cell_type&hidden_cell_types=unknown%7Cambiguous
cell_color_by=broad_cell_type&object_level_colors=tumor:%23ff4f8b%7Cimmune:cyan
```

Category matching is case-insensitive after trimming and normalizing
whitespace-like differences.

## Object Filters

Object filters control the same filter rows shown in the segmentation-object
right-panel `Properties` tab. Filters change the active object subset used for
rendering, counts, analysis, and export.

| Parameter | Meaning |
| --- | --- |
| `filter`, `filters`, `object_filter`, `object_filters` | One or more object filter clauses. |
| `filter_property`, `filter_key`, `object_filter_property`, `object_filter_key` | Property name for an explicit single filter. |
| `filter_query`, `filter_value`, `object_filter_query`, `object_filter_value` | Query value for an explicit single filter. |

Filter clauses may use `column:value`, `column=value`, `column==value`, or
`column~value`. Separate multiple clauses with `|` or `;`. Multiple clauses are
enabled and combined with AND.

Examples:

```text
filter=broad_cell_type:immune
filter=zz_mask_galectin_3%3D%3DTRUE
filter=broad_cell_type:immune%7Czz_mask_galectin_3%3D%3DTRUE
filter_property=sample_id&filter_query=Sample%2001
```

## Camera Parameters

| Parameter | Meaning |
| --- | --- |
| `center`, `center_world` | World-coordinate center as `x,y`. |
| `zoom` | Camera zoom in screen pixels per level-0 pixel. Must be positive. |

Example:

```text
center=1000,2500&zoom=0.25
```

## Full Examples

Open an ROI and show a T-cell marker panel:

```text
odon://open?project=file:///path/to/project.json&roi=ROI_001&visible_channels=CD3%7CCD4%7CCD8%7CDAPI&visible_channel_group=T%20cell%20panel&channel_order=listed
```

Open an ROI with project cell objects, colour by broad cell type, show only
immune categories, and hide the selection overlay:

```text
odon://open?project=file:///path/to/project.json&roi=ROI_001&segmentation_source=geoparquet&load_labels=0&cell_color_by=broad_cell_type&fill_cells=1&show_selection_overlay=0&visible_cell_types=immune_lymphoid%7Cimmune_myeloid
```

Open an ROI, apply object filters, and set channel contrast:

```text
odon://open?project=file:///path/to/project.json&roi=ROI_001&segmentation_source=geoparquet&load_labels=0&cell_color_by=broad_cell_type&filter=broad_cell_type:immune%7Cmarker_positive%3D%3DTRUE&channel=CD45&contrast_min=0&contrast_max=1200
```

## Troubleshooting

### The Link Opens Odon But Not The ROI

Check that the `project` path exists and that the ROI value matches an ROI id,
display name, source path, or source-key fragment in the project.

If several ROIs match, add `sample=...` or use a more specific `roi=...`.

### A Channel Or Object Category Is Not Applied

Deep links use forgiving matching, but they still need enough text to resolve a
channel or category. Try the exact channel or category label shown in Odon, and
URL-encode reserved characters.

### A Clicked `odon://` Link Does Nothing

The operating system probably has no URL handler registered for the `odon`
scheme. Open the packaged macOS app once, register the intended app bundle, or
run the macOS development handler registration script. Command-line deep links
remain available for testing.

### The Wrong Odon Build Handles The Link

On macOS, LaunchServices may route `odon://` to an older registered app bundle.
Register the intended bundle again or rerun `scripts/register-macos-url-handler.sh`
for development.

## Safety Notes

Deep links are intended to open and configure local viewer state. They should not
perform destructive actions. Treat deep links from untrusted sources with the
same caution as any link that can open local files.
