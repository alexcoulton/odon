# Fixtures

`synthetic_5ch.ome.zarr` is a small checked-in OME-Zarr pyramid for local testing and loader regressions.

Regenerate it from the repository root with:

```bash
python3 scripts/generate_ome_zarr_fixture.py --overwrite
```

Run the app's local IO sanity check against it with:

```bash
cargo run -- --check fixtures/synthetic_5ch.ome.zarr
```
