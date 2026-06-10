# Getting Started

This guide is for users who want to open Odon and load data as quickly as
possible.

## Open Odon

For normal use, open the Odon desktop application from your operating system's
application launcher.

For development builds, start the app from the repository root:

```bash
cargo run
```

This opens the app landing screen.

## Common Opening Paths

Use the Project panel as the primary way to open data:

1. Open Odon.
2. Click `Load Project...`, `Import Samplesheet CSV...`, or
   `Add OME-Zarr Root...`.
3. Select one ROI and click `Open`, or select multiple OME-Zarr ROIs and click
   `Open mosaic (N)`.

Command-line opening is available for scripted demos, debugging, and automated
checks, but it is not the primary user workflow. See [CLI](reference/cli.md) for
those commands.

## First Session Checklist

1. Launch the app.
2. Use the Project panel to open a local OME-Zarr dataset, samplesheet, or saved
   project.
3. In the left panel, inspect the layer list and select the active layer.
4. In the right panel, adjust channel contrast, color, and overlay settings.
5. Use `F` to fit the current view.

## Interface Layout

- Left panel: layers, tools, and project actions
- Right panel: properties for the active layer and analysis-oriented tabs
- Center canvas: imagery and overlays
- Top bar: quick controls, including compact contrast controls when side panels are hidden

## Recommended First Data Types

For the smoothest first experience, start with:

- a local OME-Zarr image pyramid
- an optional GeoJSON segmentation or mask overlay
- a project JSON if you already have a workspace prepared

## What To Read Next

- [Viewing Channels](workflows/viewing.md)
- [Mosaic Mode](workflows/mosaic.md)
- [Objects and Overlays](workflows/objects-and-overlays.md)
- [Projects and Samplesheets](data-sources/projects-and-samplesheets.md)
