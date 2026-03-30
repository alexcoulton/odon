# odon

<p align="center">
  <img src="docs/assets/images/logo.upscaled.white.cropped.png" alt="odon logo" width="520">
</p>

`odon` is an ultra-fast native Rust desktop viewer for multiplexed spatial proteomics and spatial transcriptomics data, specifically for OME-Zarr files.

`odon` derives its name from the dragonfly (Odonata), which are reported to exhibit ultra-fast sensitivity to flickering light.

## Getting odon

Most users should download a precompiled binary from the GitHub releases page:

- [Latest releases](https://github.com/alexcoulton/odon/releases)
- [Full documentation](https://alexcoulton.github.io/odon/)

Release builds are intended to be the main distribution path for:

- macOS
- Windows
- Linux

## What odon is for

`odon` is intended for rapid visual exploration of spatial omics data, especially when you need to:

- inspect large OME-Zarr image pyramids interactively
- compare many markers quickly on the same tissue section
- review segmentation objects, masks, points, and polygon annotations
- browse many ROIs at once in mosaic mode
- work with project files or samplesheets that define a spatial proteomics workspace

Primary workflows:

- OME-Zarr / OME-NGFF imagery
- Project JSON workspaces
- mosaic viewing across many ROIs
- GeoParquet, GeoJSON, CSV, and Parquet object overlays
- SpatialData and selected Xenium workflows

## Highlights

- GPU-backed image compositing for high-plex channel panels
- viewport-driven tile loading for local and remote datasets
- interactive overlays for objects, masks, labels, and points
- integrated thresholding and live cell-selection workflows
- mosaic mode for simultaneous viewing of hundreds of ROIs
- native Rust implementation designed for responsive performance on standard hardware

## Documentation

Detailed user documentation is available at:

- [https://alexcoulton.github.io/odon/](https://alexcoulton.github.io/odon/)

The documentation source lives in [`docs/`](docs/), with MkDocs configured by [`mkdocs.yml`](mkdocs.yml).

## Compiling odon from source

If you want to build `odon` yourself instead of using the release binaries:

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

To build the documentation locally:

```bash
uv venv .venv-docs
source .venv-docs/bin/activate
uv pip install -r requirements-docs.txt
mkdocs serve
```

## Repository layout

- `src/`: Rust application source
- `assets/`: runtime assets bundled with the app
- `docs/`: MkDocs user documentation
- `scripts/`: utility and packaging scripts
- `vendor/`: vendored crate patches

## License

This project is licensed under `GPL-3.0-only`. See [`LICENSE`](LICENSE).
