# Projects And Samplesheets

Projects and samplesheets are the main way to scale from a single ROI to a repeatable workspace.

## Project JSON

Project files are useful when you want to save:

- the list of ROIs in your workspace
- dataset metadata
- selection and focus state
- embedded masks
- layer grouping and related viewer state

Open a project with:

```bash
cargo run -- --project "/path/to/project.json"
```

## Samplesheet CSV

Samplesheets are useful when you want to drive a mosaic from tabular metadata.

Typical uses:

- bulk import many ROIs
- group or sort ROIs for a mosaic review session
- include metadata columns for labels
- provide optional segmentation paths

Open a mosaic from a samplesheet with:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv"
```

## Recommended Use

- use project files for saved workspace state and repeated review sessions
- use samplesheets for high-throughput ROI organization and mosaic setup
