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

You can also set the initial mosaic column count:

```bash
cargo run -- --mosaic-samplesheet "/path/to/samplesheet.csv" --mosaic-cols 10
```

### CSV format

A samplesheet is a CSV file with a header row and at least two columns. The first
two columns are interpreted as:

| Column | Meaning |
| --- | --- |
| `id` | Stable ROI/sample identifier shown in Odon. |
| `path` | Path to the dataset for that ROI. |

The recommended minimal file is:

```csv
id,path
ROI_001,rois/ROI_001.ome.zarr
ROI_002,/data/project/rois/ROI_002.ome.zarr
```

Relative paths are resolved relative to the samplesheet CSV file. Absolute paths
are used as written.

### Metadata columns

Every column after `id` and `path` is stored as ROI metadata. Metadata columns
can be used for browsing, sorting, grouping, project context, and mosaic layout.

Common optional columns include:

| Column | Meaning |
| --- | --- |
| `dataset` | Project dataset name for the ROI. If omitted during project import, Odon uses the project default dataset name. |
| `segpath` | Optional segmentation/object path for the ROI. Relative `segpath` values are resolved relative to the samplesheet CSV file. |
| `condition`, `batch`, `patient`, `region` | Example metadata columns for filtering, sorting, grouping, or review context. |

Example with metadata:

```csv
id,path,dataset,condition,batch,segpath
ROI_001,rois/ROI_001.ome.zarr,cohort_a,treated,B1,segmentations/ROI_001.parquet
ROI_002,/data/rois/ROI_002.ome.zarr,cohort_a,control,B1,/data/segs/ROI_002.parquet
```

TMA-style example:

```csv
id,path,dataset,segpath,tma_row,tma_col,well,patient_id,cohort,response,batch,core_quality
core_0001,cores/core_0001/image.ome.zarr,Synthetic TMA,cores/core_0001/objects/cells.parquet,A,1,A01,SYN-P001,Discovery,PR,B1,Good
core_0002,cores/core_0002/image.ome.zarr,Synthetic TMA,cores/core_0002/objects/cells.parquet,A,2,A02,SYN-P001,Discovery,SD,B1,Good
core_0003,cores/core_0003/image.ome.zarr,Synthetic TMA,cores/core_0003/objects/cells.parquet,A,3,A03,SYN-P002,Validation,PD,B1,Low tissue
```

In mosaic mode, columns such as `cohort`, `response`, `batch`, `tma_row`,
`patient_id`, and `core_quality` are available in the right-panel `Layout` tab
for grouping, sorting, secondary sorting, and labels. See
[Mosaic Mode](../workflows/mosaic.md) for the full layout workflow.

### Import behavior

When importing a samplesheet from the Project panel:

- rows with an empty `id` or empty `path` are skipped
- import fails if no usable rows remain
- the current project ROI list is replaced by the imported rows
- the first imported ROI is focused and selected
- Odon reports the number of imported ROIs and metadata columns

Use `Export Samplesheet CSV...` from an existing project to generate a valid
starting template.

## Recommended Use

- use project files for saved workspace state and repeated review sessions
- use samplesheets for high-throughput ROI organization and mosaic setup
- use metadata columns deliberately; they become the controls that make mosaic
  sorting, grouping, and labels useful
