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

## Test Fixture Policy

Fixtures required by the normal `cargo test --all-targets` suite must be checked
in here or generated deterministically as part of the repository setup. A
required test must fail clearly when its fixture is missing; it must not return
success without exercising the named behavior.

Large or non-redistributable fixtures belong to the explicit extended suite.
Those tests use Rust's `#[ignore]` attribute and verify that the expected fixture
exists when run explicitly. The current extended fixtures are:

| Test data | Expected location |
| --- | --- |
| Xenium transcript points | `data.zarr/points/transcripts/points.parquet` |
| ImageJ hyperstack | `1.tif` |
| Pyramidal OME-TIFF | `1_pyramid_crop.ome.tif` |

Run one extended test after supplying its fixture, for example:

```bash
cargo test opens_pyramidal_ome_extended_fixture -- --ignored
```

Do not add patient data or other restricted source material to the repository.
Every new binary fixture should document its origin, license, expected metadata,
and regeneration command in this file or an adjacent provenance file.
