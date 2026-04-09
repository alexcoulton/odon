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
- [Biorxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.30.715233v1) (please cite us if you benefit from Odon!)

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

## General Conversion With bioformats2raw

For broader conversion from existing microscopy formats into pyramidal OME-Zarr, we recommend [`bioformats2raw`](https://github.com/glencoesoftware/bioformats2raw).

Install with conda:

```bash
conda install -c ome bioformats2raw
```

A validated command pattern for odon is:

```bash
bioformats2raw input_file output.ome.zarr \
  --series 0 \
  --scale-format-string '%2$d/' \
  --resolutions 5 \
  -c zlib
```

This writes a single multiscale image pyramid with resolution levels at the dataset root, which is a layout odon opens directly.

Key options:

- `--series 0`: convert a single image series
- `--scale-format-string '%2$d/'`: place pyramid levels at the OME-Zarr root
- `--resolutions 5`: generate five pyramid levels
- `-c zlib`: use zlib compression

For files with multiple image series, convert each required series separately using the appropriate `--series` index.

After conversion, verify:

- channel count and order
- channel names
- pyramid loading across zoom levels

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

## Synthetic OME-Zarr Fixture

The repository includes a small checked-in 5-channel OME-Zarr fixture at [`fixtures/synthetic_5ch.ome.zarr`](fixtures/synthetic_5ch.ome.zarr) for loader tests and manual viewer checks.

Regenerate it with:

```bash
python3 scripts/generate_ome_zarr_fixture.py --overwrite
```


## TIFF To OME-Zarr Conversion

[`scripts/tif_to_omezarr.py`](scripts/tif_to_omezarr.py) converts TIFF imagery into OME-Zarr using `tifffile`, `zarr`, and `ome-zarr`.

It was used to convert `.tif` files from the Synapse repository into the OME-Zarr datasets used for testing in the manuscript.

The script infers axes from the TIFF dimensionality:

- `YX` for 2D images
- `CYX` for 3D channel-first images
- `CZYX` for 4D channel-first volumes
- `TCZYX` for 5D time-series channel-first volumes

It then writes an OME-Zarr hierarchy and generates a multiscale pyramid with `ome-zarr`'s `Scaler`.

Example:

```bash
python3 scripts/tif_to_omezarr.py input.tif output.ome.zarr
```

## Repository layout

- `src/`: Rust application source
- `assets/`: runtime assets bundled with the app
- `docs/`: MkDocs user documentation
- `fixtures/`: small checked-in synthetic datasets for tests and manual checks
- `scripts/`: utility and packaging scripts
- `vendor/`: vendored crate patches

## License

This project is licensed under `GPL-3.0-only`. See [`LICENSE`](LICENSE).
