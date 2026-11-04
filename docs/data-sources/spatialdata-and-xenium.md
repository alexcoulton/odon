# SpatialData and Xenium

Odon supports SpatialData through two deliberately separate paths. Use the
Python-led adapter when Python already owns a `spatialdata.SpatialData` object.
The native Rust opener remains a conservative standalone compatibility path.

## Python-led SpatialData adapter

Install the optional integration on Python 3.11 or newer:

```bash
python -m pip install 'odon-client[spatialdata]'
```

The supported compatibility window is `spatialdata>=0.6,<0.9`; CI exercises
0.6 on Python 3.11 and 0.8 on Python 3.12 (the Python minimum declared by
SpatialData 0.8). Importing `odon` without this extra does not import or require
the scientific Python stack.

```python
import odon
from odon.integrations.spatialdata import display

with odon.connect() as app:
    shown = display(
        app,
        sdata,
        image="morphology",
        labels=["nuclei"],
        shapes=["cells"],
        points=["transcripts"],
        tables=["table"],
        table_columns={"table": ["cell_type", "score"]},
        coordinate_system="global",
        transfer="auto",
    )
    document = shown.wait_ready()
```

The official Python library interprets the container, element versions,
relationships, and coordinate graph. It sends Odon normalized generic
resources; it never calls `datasets.open_spatialdata`.

### Runnable synthetic example

[`examples/python_spatialdata_adapter_demo.py`](../../examples/python_spatialdata_adapter_demo.py)
creates a small real SpatialData store with an image, labels, shapes, points,
and a table, then opens that backed object through the adapter. From a repository
checkout with the SpatialData extra installed:

```bash
uv run --project python python examples/python_spatialdata_adapter_demo.py \
  --launch --screenshot /tmp/odon-spatialdata-adapter-demo.png
```

Omit `--launch` to connect to an Odon instance that is already running. Use
`--fixture-only` to create and inspect the source store without Odon. The script
prints and asserts the transfer decision for every element: the image and labels
must reference the same fixture root at `images/morphology` and `labels/nuclei`,
while points and the table-enriched shapes use small session-only GeoParquet
snapshots. Its reported `raster_copies_created_by_adapter` value is therefore
zero.

For a backed raster, Python sends only the existing store URI, element group
path (for example `images/morphology`), resolved affine transform, and
provenance. Rust opens the original OME-Zarr chunks lazily and owns IO, caches,
GPU upload, rendering, interaction, and application state. No duplicate raster
store is created.

Each element chooses its transfer independently:

| Policy | Result |
| --- | --- |
| `reference` | Require proven Odon-readable backing and fail rather than copy or stream. |
| `provider` | Serve selected raster chunks from Python over a tokenized loopback-only virtual OME-Zarr store. |
| `snapshot` | Explicitly adapt a selected vector element to GeoParquet; raster snapshots are never implicit. |
| `auto` | Prefer reference, use the provider for an unbacked/computed raster, and snapshot only selected incompatible vectors. |

A mapping selects policies per element, with `"*"` as the fallback:

```python
shown = display(
    app,
    sdata,
    image="morphology",
    shapes=["cells"],
    transfer={"morphology": "reference", "cells": "snapshot", "*": "auto"},
)
```

`shown.elements[name]` reports the actual source, format, group, transfer
decision, coordinate space, source lifetime, ownership, resource, and layer.
`shown.readiness` distinguishes source, resource, model, and presentation
readiness.

### Images and labels

A backed, self-contained image or labels element is referenced only when the
adapter can prove its stored group and current lazy graph still represent the
same pixels. Reopening the same Zarr graph may change generated Dask hashes, so
the proof also compares backing identities and normalized read-graph structure;
computed, sliced, or rechunked graphs fail closed.

Unbacked NumPy, Xarray, or Dask rasters use `ArrayProvider` under `auto` or
`provider`. It binds to loopback, uses an unguessable per-generation URL,
serves only requested chunks, applies bounded concurrency/backpressure, caches
recent chunks, and shuts down with the display. The current provider exposes
one resolution level. It does not use Odon's control bearer token and is always
session-lifetime/session-owned.

### Shapes, points, and tables

Compatible persisted shapes GeoParquet is referenced directly. Selected
in-memory or incompatible shapes and points are adapted individually to a
temporary GeoParquet resource; this never causes a raster copy. Points are
computed only when that selected vector snapshot is requested.

Tables are conservative. The caller must name scalar/categorical `obs`
columns. The adapter uses SpatialData's official table keys, requires exactly
one selected shapes target, joins by `instance_id`, and writes those columns
into that vector snapshot. Expression matrices are not copied wholesale.

Registered GeoParquet shapes and points are actor-owned selectable object
resources. Selection IDs can be returned to caller-managed Python code:

```python
selection = shown.selection("cells", limit=1_000)
selected_ids = selection.ids

# The caller decides how these IDs subset or annotate its own data.
selected_rows = sdata.shapes["cells"].loc[
    sdata.shapes["cells"]["instance_id"].astype(str).isin(selected_ids)
]
```

The selection snapshot reports its full count, bounded returned IDs, primary
ID, truncation flag, and generation. It does not imply shared mutable memory.

### Coordinate systems

Python resolves every selected element into one caller-selected or unambiguous
shared coordinate system, then normalizes it relative to the primary image.
The generic resource contract carries an invertible 3×3 homogeneous 2D affine
with explicit x/y axes, input-to-reference direction, and pixel-centre
convention. Rotation, shear, reflection, scale, and translation are rendered
without SpatialData-specific Rust transform parsing. Non-affine and unsupported
dimensional mappings fail before presentation.

### Remote sources

Group-addressed `file:`, HTTP(S), and S3 primary OME-Zarr resources use the
same generic open path. HTTP secondary images/labels can also be referenced.
S3 credentials remain in Odon's session credential state and never enter a
project file. An S3 direct reference is therefore session-lifetime even though
the underlying object is durable. At present, an S3-backed composite with
secondary raster layers must be adapted to supported durable URLs or opened as
a primary-only scene; Odon deliberately does not project S3 secrets into the
renderer-side external-layer registry.

### Refresh, cleanup, and persistence

`shown.refresh(sdata, element=...)` opens a complete new document generation
and waits for it before releasing the old session resources. It is replacement,
not invisible mutation of an existing reader.

`shown.close()` removes session-owned layers/resources, provider servers, and
temporary vector directories. Durable project-owned resources are not removed.

Provider and temporary-vector dependencies cannot be project-owned. Promote a
display explicitly:

```python
durable = shown.persist("analysis-display", overwrite=False)
durable.wait_ready()
app.projects.save_as("analysis.odon.project.json").wait()
```

Persistence checks disk space, materializes provider rasters chunk-by-chunk as
OME-Zarr, copies adapted vectors, registers every dependency as durable and
project-owned, and atomically opens the promoted composite. Failure cleans up
newly created outputs. It is the only operation that turns ephemeral adapter
state into durable project state.

## Current boundaries

- The primary element is an image.
- Selected names must be unique across image, labels, shapes, points, and table
  families because transfer policies and display handles are keyed by name.
- Raster sources must be supported OME-Zarr/Zarr or a live Python provider.
- Non-affine transformations and unsupported axes/codecs fail explicitly.
- Table projection currently rewrites the selected vector snapshot; a generic
  keyed-property resource for property-only refresh and label-index joins is a
  later optimization.
- A provider must remain connected. Unexpected provider loss surfaces as
  failed chunk reads; normal `close()` removes its registered dependencies.
- Direct-reference display assumes the source is quiescent for that document
  generation. Use `refresh()` after external writes.

## Native SpatialData compatibility opening

`app.datasets.open_spatialdata(...)` uses Odon's native compatibility reader.
It remains useful when no Python process is involved, but it is not the
canonical interpreter of evolving SpatialData semantics and the Python adapter
is permanently prohibited from wrapping it. Odon will retain this conservative
best-effort path until a separate product decision changes standalone support.

## Xenium

Current Xenium support includes dataset discovery from `experiment.xenium`,
morphology imagery, cell polygons, and transcript points. This is independent
of the Python-led SpatialData adapter.
