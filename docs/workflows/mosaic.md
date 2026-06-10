# Mosaic Mode

Mosaic mode lets you review many ROIs on one canvas with shared channel display
settings. It is useful for TMA review, cohort-level quality control, comparing
marker patterns across samples, and quickly finding ROIs that need closer
inspection in the single-image viewer.

## When To Use Mosaic Mode

Use mosaic mode when you want to:

- compare many ROIs without opening them one by one
- review a TMA-style collection of cores
- sort or group ROIs by metadata such as patient, cohort, batch, response, or
  tissue region
- use the same channel visibility, colour, and contrast settings across every
  ROI
- perform fast image-first review before choosing individual ROIs for detailed
  object or mask work

Mosaic mode is strongest for multi-ROI image review. Some object, mask, and
analysis workflows are still more complete in single-view mode.

## Launching A Mosaic

### From A Samplesheet

Samplesheets are the easiest way to build a mosaic from a folder of ROIs and
tabular metadata:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv"
```

To set the initial number of mosaic columns:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv" --mosaic-cols 10
```

### From A Project

Project-based mosaics open ROIs that already belong to a saved project dataset:

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2"
```

To set the initial number of columns:

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2" --mosaic-cols 10
```

Use project-based mosaics when you already have a curated project with saved
ROIs, masks, annotation layers, or view state. Use samplesheet-based mosaics
when you want a lightweight, tabular way to define a multi-ROI workspace.

## Samplesheet Basics

A samplesheet is a CSV file with at least two columns:

| Column | Meaning |
| --- | --- |
| `id` | Stable ROI identifier shown in Odon. |
| `path` | Path to the ROI image dataset. |

Every column after `id` and `path` becomes ROI metadata. Those metadata columns
are available in mosaic mode for sorting, grouping, labels, and review context.

Example:

```csv
id,path,dataset,segpath,tma_row,tma_col,patient_id,cohort,response,batch
core_0001,cores/core_0001/image.ome.zarr,Synthetic TMA,cores/core_0001/objects/cells.parquet,A,1,SYN-P001,Discovery,PR,B1
core_0002,cores/core_0002/image.ome.zarr,Synthetic TMA,cores/core_0002/objects/cells.parquet,A,2,SYN-P001,Discovery,SD,B1
core_0003,cores/core_0003/image.ome.zarr,Synthetic TMA,cores/core_0003/objects/cells.parquet,A,3,SYN-P002,Validation,PD,B1
```

Relative paths are resolved relative to the CSV file. If the samplesheet is
stored next to a `cores/` folder, `cores/core_0001/image.ome.zarr` resolves to
that folder. This makes a synthetic or demonstration TMA portable as long as the
folder structure stays the same.

Common metadata columns for TMA review include:

| Column | Use |
| --- | --- |
| `tma_row`, `tma_col`, `well` | Restore or inspect plate/core layout. |
| `patient_id`, `case_id`, `core_replicate` | Keep replicate cores together. |
| `cohort`, `diagnosis`, `tumor_site` | Group biological or clinical categories. |
| `stage`, `grade`, `response`, `treatment_arm` | Demonstrate clinical review layouts. |
| `batch`, `slide`, `scanner_run` | Review technical effects. |
| `core_quality` | Separate good and low-quality cores. |
| `segpath` | Optional object/segmentation file for each ROI. |

For the full CSV contract, see
[Projects And Samplesheets](../data-sources/projects-and-samplesheets.md).

## What You See In Mosaic Mode

The mosaic window has the same general structure as the single viewer:

- the centre canvas shows all ROI images
- the left panel contains layers and project controls
- the right panel contains `Properties`, `Views`, `Layout`, and `Memory`
- the top bar contains fit/navigation controls and compact channel controls

Each ROI is drawn into a shared mosaic coordinate system. Odon streams visible
tiles first, draws coarse levels early, and refines to finer levels as tiles
arrive.

Channel settings are global in a mosaic. When you hide `CD45`, change the colour
of `DAPI`, or adjust contrast for a marker, that setting applies across every
ROI in the mosaic.

## Navigation

Use these controls while reviewing a mosaic:

| Action | Control |
| --- | --- |
| Fit the full mosaic | Press `F` or click `Fit Mosaic (F)`. |
| Zoom | Mouse wheel or trackpad pinch. |
| Pan | Drag on the canvas. |
| Focus one ROI | Double-click the ROI. |
| Step through ROIs | Use `Prev. Core` and `Next. Core`. |
| Return from mosaic to project/single view | Use `Back` when available. |

Double-clicking an ROI is useful when scanning a cohort: it fits that ROI to the
viewport without leaving mosaic mode.

## Arranging ROIs

Open the right panel and choose the `Layout` tab to arrange the mosaic.

### Group By

`Group by` splits ROIs into blocks based on a metadata column. For example:

- `cohort` creates one block per cohort
- `response` creates one block per response category
- `batch` creates one block per acquisition batch
- `tma_row` creates one block per TMA row

When grouping is enabled, `Show group labels` draws the group name on the canvas.
`Gap` controls the spacing between grouped blocks.

If a group is labelled `(missing)`, at least one ROI does not have a value for
the selected metadata column. This usually means one of the following:

- the column is blank for that row in the samplesheet
- the mosaic includes an ROI that did not come from the samplesheet
- a saved project or view restored a `Group by` column that is not present in the
  current samplesheet

### Layout Mode

The `Layout` selector has two modes:

| Mode | Best For | Behaviour |
| --- | --- | --- |
| `Fit Cells` | TMA review and mixed-size ROIs | Each ROI is scaled to fit a regular grid cell while preserving aspect ratio. |
| `Native Pixels` | Comparing true pixel sizes | ROIs are placed at native pixel scale, so larger ROIs take more canvas space. |

`Fit Cells` is usually the best starting point for a TMA or demo mosaic because
each core gets comparable visual space.

### Sort By And Then By

`Sort by` orders ROIs using `id` or any metadata column. Enable `Then by` for a
secondary sort. Common combinations are:

- `Group by: cohort`, `Sort by: patient_id`, `Then by: core_replicate`
- `Group by: response`, `Sort by: stage`, `Then by: patient_id`
- `Group by: batch`, `Sort by: tma_row`, `Then by: tma_col`

The layout updates when you change these controls or click `Apply sort`.

## Text Labels

The left panel includes a `Text Labels` layer. Select that layer, then use the
right-panel `Properties` tab to choose which values are drawn over each ROI.

You can add multiple label lines. Useful label columns include:

- `id`
- `patient_id`
- `cohort`
- `response`
- `core_quality`
- `well`

Text labels are most useful after arranging the mosaic by metadata. For example,
group by `cohort`, sort by `patient_id`, and label each ROI with `id` and
`response`.

## Channel Display

Select a channel layer in the left panel to edit global channel display settings
in the `Properties` tab.

The channel properties include:

- channel selection
- visibility through the layer list
- colour
- contrast window
- channel grouping
- notes

Because the settings are shared, a contrast adjustment made for one channel
applies to every ROI. This is intentional: mosaic mode is designed for comparing
marker patterns under a consistent display setup.

The top bar also provides quick channel stepping and compact contrast controls
for rapid review.

## Segmentation And Object Paths

Samplesheets can include a `segpath` column. In mosaic mode, Odon uses that
column to discover per-ROI segmentation/object files.

Example:

```csv
id,path,segpath
core_0001,cores/core_0001/image.ome.zarr,cores/core_0001/objects/cells.parquet
core_0002,cores/core_0002/image.ome.zarr,cores/core_0002/objects/cells.parquet
```

When the `Segmentation` layer is available, select it in the left panel to see
loading and display controls in the `Properties` tab. Mosaic object workflows are
intended for broad review across ROIs; detailed object measurement and editing
workflows are still better in single-view mode.

## Memory Tab

The `Memory` tab controls optional RAM pinning for mosaic rendering.

By default, Odon streams visible tiles on demand. This is usually the safest
choice. Pinning is manual and should be used only when you want to keep selected
channels and levels resident in CPU RAM for repeated review.

The Memory tab lets you:

- choose which channels are eligible for pinning
- load a selected level for the focused ROI
- load a selected level for all eligible ROIs
- unload pinned levels
- see estimated RAM use and warning/danger labels

Lower-numbered levels are higher resolution and usually require more RAM.
Higher-numbered levels are coarser and cheaper to pin. For large mosaics, avoid
pinning level `0` across all ROIs unless the estimate is clearly safe.

## Recommended TMA Workflow

For a TMA-style demonstration:

1. Create a samplesheet with one row per core.
2. Use relative `path` and `segpath` values so the folder can be moved.
3. Include layout columns such as `tma_row`, `tma_col`, and `well`.
4. Include review columns such as `cohort`, `response`, `batch`, and
   `core_quality`.
5. Launch with `--mosaic-samplesheet`.
6. Open the `Layout` tab.
7. Start with `Fit Cells`.
8. Group by `cohort` or `response`.
9. Sort by `patient_id`, then by `core_replicate`.
10. Select the `Text Labels` layer and add labels such as `id`, `patient_id`, and
    `response`.
11. Use channel visibility and contrast to compare marker patterns.
12. Double-click interesting ROIs for closer review.

## Troubleshooting

### A Group Is Labelled `(missing)`

The selected `Group by` column is empty for at least one ROI. Check that:

- the column exists in the samplesheet header
- every row has a value for that column
- the mosaic does not contain extra ROIs from another project or previous import
- the project view has not restored a grouping column from a different
  samplesheet

### Some Rows Did Not Open

Rows with empty `id` or `path` values are skipped. Rows whose image paths cannot
be opened are also skipped. Check that relative paths are correct relative to the
samplesheet CSV location.

### Metadata Columns Are Not Available

Only columns after `id` and `path` are treated as metadata. If a column does not
appear in `Group by`, `Sort by`, or label controls, check that it is present in
the CSV header and not duplicated with different spelling.

### The Mosaic Is Slow

Try these steps:

- reduce the number of visible channels
- use `Fit Cells` rather than `Native Pixels` for overview review
- avoid pinning high-resolution levels across all ROIs
- group or sort only after the initial mosaic has opened
- use local storage for large demos when possible

### The Viewer Cannot Open Mosaic Mode

Mosaic mode currently depends on the GPU rendering path. If the app cannot create
the mosaic view, check the terminal output and confirm that the system supports
the required OpenGL backend.

## Current Limitations

- mosaic mode is image-first; some overlay, mask, and analysis workflows are more
  complete in single-view mode
- object and segmentation display in mosaics is intended for review rather than
  full per-object analysis
- the mosaic renderer depends on the GPU path
- very large mosaics can still be limited by storage bandwidth, visible channel
  count, and memory pressure

For broader constraints, see
[Current Limitations](../advanced/current-limitations.md).
