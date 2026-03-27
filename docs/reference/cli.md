# CLI

## Launch The App

```bash
cargo run
```

## Open A Project

```bash
cargo run -- --project "/path/to/project.json"
```

## Open A Single Dataset

```bash
cargo run -- "/path/to/ROI1.ome.zarr"
```

## Open A Mosaic From A Project

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2"
```

## Open A Mosaic From A Samplesheet

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv"
```

## Set Mosaic Columns

```bash
cargo run -- --project "/path/to/project.json" --mosaic "TMA1v3,TMA2" --mosaic-cols 10
```

## Run A Coarse-Level Sanity Check

```bash
cargo run -- --check "/path/to/ROI1.ome.zarr"
```
