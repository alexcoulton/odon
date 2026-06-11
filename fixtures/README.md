# Fixtures

`synthetic_5ch.ome.zarr` is a small checked-in OME-Zarr pyramid for local testing and loader regressions.
It has 5 channels (`DAPI`, `CD3`, `PanCK`, `Ki67`, and `Collagen`) and 4 pyramid levels.

Regenerate it from the repository root with:

```bash
python3 scripts/generate_ome_zarr_fixture.py --overwrite
```

Run the app's local IO sanity check against it with:

```bash
cargo run -- --check fixtures/synthetic_5ch.ome.zarr
```

`tma_100x1mb.zip` is a synthetic 100-core TMA example for mosaic mode and
samplesheet workflows. After unzipping, import
`tma_100x1mb/synthetic_tma_samplesheet.csv` from the Odon Project panel, click
`Select all`, then click `Open mosaic (100)`.

The TMA samplesheet uses relative `path` and `segpath` values, so the unzipped
folder can be moved as a unit without editing the CSV.
