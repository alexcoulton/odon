# odon

<p align="center">
  <img src="docs/assets/images/logo.upscaled.white.cropped.png" alt="odon logo" width="520">
</p>

`odon` is a native desktop viewer for multiplexed spatial proteomics and spatial transcriptomics data.

Multiplex imaging datasets are large, high-dimensional, and often awkward to inspect efficiently, especially at whole-slide or cohort scale. Visual review is still essential for rapid detection of staining artefacts, protein aggregates, non-specific signal, segmentation errors, and other acquisition issues that are often easier to detect by eye than by downstream analysis alone. `odon` is designed to make that inspection fast on standard laptops rather than requiring a heavily provisioned workstation.

The viewer is built primarily around the OME-Zarr imaging format, with annotations and object overlays via GeoJSON and GeoParquet, plus secondary support for SpatialData, Xenium containers, and TIFF / OME-TIFF. Data can be loaded locally or streamed directly from HTTP and S3-compatible object storage using viewport-driven tile loading. The rendering engine is optimized for rapid interaction with large high-plex datasets, including whole-slide imagery, large segmentation overlays, and mosaic views spanning many regions of interest.

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
