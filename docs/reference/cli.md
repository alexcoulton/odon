# CLI

The CLI is a secondary interface for development, scripted demos, debugging, and
automated checks. For routine use, open Odon as a desktop application and use the
Project panel to load projects, samplesheets, OME-Zarr roots, and mosaics.

## Launch A Development Build

```bash
cargo run
```

## Open A Project Directly

```bash
cargo run -- --project "/path/to/project.json"
```

## Open A Single Dataset Directly

```bash
cargo run -- "/path/to/ROI1.ome.zarr"
```

## Open A Mosaic From A Project Directly

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2"
```

## Open A Mosaic From A Samplesheet Directly

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv"
```

The samplesheet CSV must have a header row. The first two columns are `id` and
`path`; any later columns are imported as metadata for sorting, grouping, and
labels. See
[Projects And Samplesheets](../data-sources/projects-and-samplesheets.md#csv-format)
for the full CSV layout.

Set the initial number of columns for a samplesheet mosaic:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv" --mosaic-cols 10
```

## Set Mosaic Columns

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2" --mosaic-cols 10
```

## Run A Coarse-Level Sanity Check

```bash
cargo run -- --check "/path/to/ROI1.ome.zarr"
```
