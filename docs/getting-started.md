# Getting Started

This guide is for users who want to open Odon from the desktop application and
load image data quickly.

## Open Odon

Open the Odon desktop application from your operating system's application
launcher.

Odon opens to the project/start screen. The Project panel is the main place to
load data, create a workspace, and open single-ROI or mosaic views.

## Choose How To Load Data

Use the Project panel for normal data loading.

| What you have | What to click | Result |
| --- | --- | --- |
| A saved Odon project | `Load Project...` | Restores a saved workspace with ROIs, paths, and project metadata. |
| A samplesheet CSV | `Import Samplesheet CSV...` | Creates a multi-ROI project from tabular metadata and image paths. |
| A folder containing OME-Zarr datasets | `Add OME-Zarr Root...` | Scans the folder and lists compatible image datasets as project ROIs. |

After data appears in the Project panel:

1. Select one ROI and click `Open` for single-image viewing.
2. Select multiple compatible ROIs and click `Open mosaic (N)` for mosaic
   viewing.
3. Use `Select all` before `Open mosaic (N)` when you want to review every
   listed ROI.

For samplesheet details, see
[Projects and Samplesheets](data-sources/projects-and-samplesheets.md). For
multi-ROI review, see [Mosaic Mode](workflows/mosaic.md).

## First Single-Image Session

After opening one ROI:

1. Use the left panel to inspect image, object, mask, and annotation layers.
2. Select an image layer or channel group.
3. Use the right panel to adjust channel visibility, colour, and contrast.
4. Pan with left-drag and zoom with the mouse wheel or trackpad.
5. Press `F` to fit the image to the current view.

The viewer streams pyramid tiles as needed. On large images, Odon should show a
coarse view first and refine visible tiles as they arrive.

## First Mosaic Session

After opening a mosaic:

1. Use the shared channel controls to choose the channels shown across all ROIs.
2. Open the right-panel `Layout` tab.
3. Group, sort, or label ROIs using samplesheet/project metadata.
4. Use `Fit Mosaic (F)` to fit the full mosaic.
5. Use `Fit Cells` when reviewing regular TMA-like layouts.

Mosaic mode is designed for comparing many ROIs with the same channel settings.
Detailed object measurement and editing workflows are usually better in
single-image mode.

## Interface Layout

| Area | Purpose |
| --- | --- |
| Left panel | Layers, project actions, object layers, masks, and tools. |
| Center canvas | Image data, overlays, masks, object outlines, and mosaic tiles. |
| Right panel | Properties for the active layer, channel controls, layout tools, analysis tabs, and memory controls. |
| Top bar | Compact viewer controls and quick access actions. |

## Recommended First Data Types

For the smoothest first experience, start with:

- a local OME-Zarr image pyramid
- a samplesheet CSV if you have multiple ROIs
- optional GeoParquet, Parquet, or GeoJSON object data linked with `segpath`
- a saved Odon project JSON if you already have one

## Development And CLI Startup

Command-line startup is available for development builds, scripted demos,
debugging, and automated checks. It is not the primary user workflow.

From the repository root:

```bash
cargo run
```

See [CLI](reference/cli.md) for command-line options.

## What To Read Next

- [Viewing Channels](workflows/viewing.md)
- [Mosaic Mode](workflows/mosaic.md)
- [Objects and Overlays](workflows/objects-and-overlays.md)
- [Mask Polygons](workflows/mask-polygons.md)
- [Projects and Samplesheets](data-sources/projects-and-samplesheets.md)
- [Object and Overlay Data](data-sources/object-and-overlay-data.md)
