# Getting Started

This guide is for users who want to run the viewer locally and open data as quickly as possible.

## Build And Launch

From the repository root:

```bash
cargo run
```

This opens the app landing screen.

## Common Launch Paths

Open a saved project:

```bash
cargo run -- --project "/path/to/project.json"
```

Open a single local dataset directly:

```bash
cargo run -- "/path/to/ROI1.ome.zarr"
```

Open a mosaic from a project:

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2"
```

Open a mosaic from a samplesheet:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv"
```

Optionally fix the mosaic column count:

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2" --mosaic-cols 10
```

## First Session Checklist

1. Launch the app.
2. Open a local OME-Zarr dataset or a saved project.
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
- [Objects and Overlays](workflows/objects-and-overlays.md)
- [Projects and Samplesheets](data-sources/projects-and-samplesheets.md)
