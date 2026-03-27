# odon

`odon` is a native Rust desktop viewer for multiplex imaging and spatial proteomics data.

It is designed for fast interactive inspection of large image pyramids, channel overlays, segmentation objects, masks, and point-based data without requiring a browser-based stack.

## Current Scope

Primary workflows:

- OME-Zarr / OME-NGFF imagery
- Project JSON workspaces
- mosaic viewing across many ROIs
- GeoParquet, GeoJSON, CSV, and Parquet object overlays
- SpatialData and selected Xenium workflows

## Quick Start

Build and run the viewer:

```bash
cargo run
```

Open a dataset directly:

```bash
cargo run -- "/path/to/dataset.ome.zarr"
```

Open a project:

```bash
cargo run -- --project "/path/to/project.json"
```

Open a mosaic from a project:

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2"
```

Open a mosaic from a samplesheet:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv"
```

Show CLI help:

```bash
cargo run -- --help
```

## Documentation

The user documentation source lives in [`docs/`](docs/), and the MkDocs config lives in [`mkdocs.yml`](mkdocs.yml).

To build the documentation locally:

```bash
uv venv .venv-docs
source .venv-docs/bin/activate
uv pip install -r requirements-docs.txt
mkdocs serve
```

## Repository Layout

- `src/`: Rust application source
- `assets/`: runtime assets bundled with the app
- `docs/`: MkDocs user documentation
- `scripts/`: utility and packaging scripts
- `vendor/`: vendored crate patches

## License

This project is licensed under `GPL-3.0-only`. See [`LICENSE`](LICENSE).

## Status

This repository is being prepared for its first public release. Expect some active cleanup around packaging, documentation, and CI workflows.
