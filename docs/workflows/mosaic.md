# Mosaic Mode

Mosaic mode lets you inspect many ROIs on one canvas with shared display settings.

## When To Use It

Use mosaic mode when you want to:

- compare many ROIs quickly
- review marker patterns across a TMA-style layout
- browse a project selection without opening each ROI one by one

## Launching A Mosaic

From a project:

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2"
```

From a samplesheet:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv"
```

Set a fixed number of columns if needed:

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2" --mosaic-cols 10
```

## How It Behaves

- ROIs are placed into a grid
- each ROI is scaled to fit its grid cell while preserving aspect ratio
- channel visibility, color, and contrast are shared across the mosaic

## Navigation

- `F` fits the full mosaic bounds
- double click fits the ROI under the cursor
- `Prev. Core` and `Next. Core` jump between ROIs so one fills the viewport

## Current Strengths

- designed for image-first comparison across many ROIs
- coarse-to-fine rendering keeps the canvas responsive
- works well with project files and samplesheet-driven organization

## Current Limitations

- mosaic mode is currently image-first; not every overlay workflow is available there
- the current implementation depends on the GPU rendering path
- some advanced overlay and analysis workflows are better in single-view mode

For the full constraints, see [Current Limitations](../advanced/current-limitations.md).
