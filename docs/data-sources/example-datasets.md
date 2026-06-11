# Example Datasets

The repository includes two synthetic datasets that are useful for learning the
GUI workflows, checking an installation, and preparing demonstrations. Both are
synthetic and contain no real patient data.

## Synthetic 5-Channel OME-Zarr

Use this dataset for a quick single-image test.

Repository path:
[`fixtures/synthetic_5ch.ome.zarr`](https://github.com/alexcoulton/odon/tree/main/fixtures/synthetic_5ch.ome.zarr)

It is a small OME-Zarr pyramid with:

- 5 channels: `DAPI`, `CD3`, `PanCK`, `Ki67`, and `Collagen`
- 4 pyramid levels
- level 0 shape `5 x 512 x 512`, with `CYX` axes
- `uint16` image data
- 256 x 256 XY chunks at level 0

This fixture is best for:

- confirming that Odon can open an OME-Zarr image
- checking channel names, colours, and contrast controls
- testing tile loading without needing a large dataset
- reproducing simple viewer behaviour in bug reports

To open it from the GUI:

1. Launch Odon.
2. In the Project panel, click `Add OME-Zarr Root...`.
3. Choose the repository `fixtures/` folder, or choose the
   `fixtures/synthetic_5ch.ome.zarr` folder directly.
4. Select the discovered ROI.
5. Click `Open`.

After the image opens, use the channel list and the right-panel `Properties` tab
to show or hide markers and adjust contrast.

## Synthetic TMA 100x1MB

Use this dataset for mosaic mode, samplesheets, layout controls, and
segmentation/object overlay demonstrations.

Repository path:
[`fixtures/tma_100x1mb.zip`](https://github.com/alexcoulton/odon/blob/main/fixtures/tma_100x1mb.zip)

The archive expands to a `tma_100x1mb/` folder containing:

- `synthetic_tma_samplesheet.csv`
- `manifest.json`
- `cores/core_0001/` through `cores/core_0100/`
- one `image.ome.zarr` per core
- one `objects/cells.parquet` file per core
- one `objects/cells.geojson` file per core
- one `preview.png` per core

Each core image is a small synthetic OME-Zarr pyramid with:

- 24 channels
- 3 pyramid levels
- 112 x 112 pixels at level 0
- approximately 1 MiB allocated on disk per core folder
- synthetic object data for object/segmentation display

The samplesheet uses relative paths, for example:

```csv
id,path,dataset,segpath,preview_path,tma_row,tma_col,well,patient_id,case_id,core_replicate,cohort,response,core_quality
core_0001,cores/core_0001/image.ome.zarr,Synthetic TMA 100x1MB,cores/core_0001/objects/cells.parquet,cores/core_0001/preview.png,A,1,A01,SYN-P001,SYN-C0001,1,Discovery,PD,Good
```

Relative paths are resolved relative to the samplesheet CSV file. This means the
unzipped `tma_100x1mb/` folder can be moved as a unit without editing the CSV.

To open the TMA from the GUI:

1. Download or locate `fixtures/tma_100x1mb.zip`.
2. Unzip it.
3. Launch Odon.
4. In the Project panel, click `Import Samplesheet CSV...`.
5. Choose `tma_100x1mb/synthetic_tma_samplesheet.csv`.
6. Click `Select all`.
7. Click `Open mosaic (100)`.

After the mosaic opens:

1. Open the right-panel `Layout` tab.
2. Set the layout mode to `Fit Cells`.
3. Try `Group by: cohort` or `Group by: response`.
4. Try `Sort by: patient_id`, then `Then by: core_replicate`.
5. Select the `Text Labels` layer and add labels such as `id`, `patient_id`,
   `response`, or `core_quality`.

Useful samplesheet columns include:

| Column | Use |
| --- | --- |
| `path` | Relative path to the OME-Zarr image for each core. |
| `segpath` | Relative path to the per-core object file. |
| `preview_path` | Relative path to the preview PNG. |
| `tma_row`, `tma_col`, `well` | TMA-style layout coordinates. |
| `patient_id`, `case_id`, `core_replicate` | Demonstration grouping for repeated cores. |
| `cohort`, `response`, `batch`, `slide`, `scanner_run` | Mosaic grouping and sorting metadata. |
| `core_quality` | Demonstration QC label. |
| `synthetic_metadata`, `demo_only` | Flags showing that the metadata is synthetic demo data. |

See [Mosaic Mode](../workflows/mosaic.md) for the full mosaic workflow and
[Projects and Samplesheets](projects-and-samplesheets.md) for the samplesheet
CSV format.
