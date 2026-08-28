# Python-Led SpatialData Adapter Plan

Status: implemented with documented capability boundaries (2026-08-28)

Implemented:

- data resources carry a validated, persisted `group_path` distinct from their
  store URI;
- `datasets.open_resource` opens a registered local OME-Zarr/Zarr group as the
  primary document and atomically commits related registered layers;
- sync and async Python SDKs expose group-addressed registration and composite
  primary-resource opening;
- the optional `odon.integrations.spatialdata` adapter directly references
  backed local or supported remote images and labels and resolves a shared
  coordinate system into an arbitrary invertible 2D affine;
- reference proof rejects external backing and in-memory raster graphs that
  differ from a fresh metadata-only read;
- compatible GeoParquet is referenced while selected incompatible shapes,
  points, and table columns are adapted independently;
- unbacked/computed rasters use a bounded, tokenized loopback array provider;
- selectable vectors are actor-owned and expose selection IDs back to Python;
- refresh replaces a complete document generation before releasing the old
  one, while explicit persistence promotes ephemeral dependencies to durable
  OME-Zarr/GeoParquet resources; and
- Rust/Python tests plus oldest/newest supported SpatialData CI lanes enforce
  the generic path and prohibit calls to `datasets.open_spatialdata`.

Deliberate follow-on optimizations are a geometry-independent keyed-property
resource, label-index property joins, and credential-safe S3 secondary-raster
presentation. Current table projection rewrites only the selected vector, and
S3 composite scenes are limited to a primary raster unless secondary resources
have renderer-readable durable URLs. Automated release evidence still needs a
rendered screenshot/tile-counter no-copy test and a proactive viewer-level
provider-disconnect state; today, unexpected provider loss is explicit on the
next chunk read.

Target: make the Python `spatialdata` library the authoritative interpreter of
SpatialData objects and stores while Odon's Rust core reads, caches, and renders
generic referenced image, label, geometry, point, and property resources.

Related documents:

- `docs/design/complete-application-python-api-plan.md`
- `docs/design/control-protocol-v1.md`
- `docs/design/python-api-plan.md`
- `docs/reference/python-api.md`
- `docs/reference/python-api-contracts.md`
- `docs/advanced/python-api-limitations.md`
- `docs/data-sources/spatialdata-and-xenium.md`

## Executive Summary

Odon should not maintain a second, complete implementation of the evolving
SpatialData container model in Rust. The canonical Python integration should
instead use the official Python `spatialdata` library to discover elements,
resolve coordinate systems, interpret element-format versions, and determine
where each element is backed. Python then sends Odon a normalized manifest of
generic resources. Rust opens those resources directly and remains responsible
for data IO, decoding, caching, GPU upload, rendering, interaction, analysis,
and application state.

For a normal backed SpatialData Zarr store, image pixels do not pass through
Python and are not duplicated. Python identifies the existing store and image
group, for example `file:///data/sample.zarr` plus
`images/morphology`; Odon opens that OME-NGFF group and reads its chunks lazily.
Python reads only the metadata needed to construct the manifest.

```text
SpatialData object or store
          |
          | official Python APIs: elements, backing, transforms, tables
          v
Python SpatialData adapter
          |
          | bounded generic resource/document manifest
          v
Odon control actor ---------------------> semantic application state
          |
          | store URI + group path
          v
Rust OME-Zarr/Parquet readers -> caches -> renderer
```

The existing `datasets.open_spatialdata` command remains a native compatibility
path for standalone Odon. The new Python integration must not implement itself
by calling that command. Its purpose is to replace SpatialData-specific Rust
discovery with generic resource contracts, not to wrap the existing Rust
adapter.

## Architectural Decision

### Python owns SpatialData semantics

The optional Python adapter owns:

- recognizing a `spatialdata.SpatialData` object;
- enumerating images, labels, points, shapes, and tables;
- checking whether an element is backed and self-contained;
- locating the backing store and element group when direct reference is safe;
- understanding supported SpatialData container and element-format versions;
- choosing and resolving a target coordinate system;
- composing SpatialData transformations into a normalized affine transform;
- determining which tables annotate which regions;
- selecting a transfer policy independently for each element; and
- preserving source, element, and transformation provenance.

Version-specific compatibility code belongs in this adapter. It can be tested
against multiple Python `spatialdata` releases and updated without changing
Odon's renderer or control protocol.

### Rust owns generic data and rendering semantics

The Rust application owns:

- opening local and remote stores;
- interpreting supported OME-NGFF image and label groups;
- reading supported Parquet/GeoParquet geometry and point resources;
- validating the resource descriptor against the actual source;
- bounded chunk and table IO;
- decoded-tile and GPU-resource caches;
- layer lifecycle, readiness, ordering, visibility, style, and revisions;
- document and viewport state;
- object indexing, filtering, selection, analysis, measurement, and export; and
- persistence rules for durable resource descriptors.

Rust should not need to know that a resource originated in a SpatialData
container. It should maintain OME-NGFF, Zarr, Parquet, geometry, table, and
coordinate contracts because those are general Odon capabilities.

### The control connection is not an image data plane

JSON-RPC carries metadata, commands, stable IDs, revisions, task state, and
small bounded values. It must not carry whole images or large arrays.

Large data uses one of these per-element policies:

1. direct reference to an existing store;
2. direct reference to an original non-SpatialData source Odon supports;
3. session-scoped chunk provision by Python;
4. explicit materialization to a stable interchange format; or
5. an explicit durable snapshot.

Materialization must never be a silent side effect of the default policy.

## Goals

- Display backed SpatialData images without reading their pixels through
  Python and without creating a second OME-Zarr store.
- Move SpatialData discovery, version handling, coordinate resolution, and
  table relationships out of Rust and into an optional Python adapter.
- Define resource and document contracts that are also useful for Xarray,
  Dask, NumPy, GeoPandas, AnnData, Cellpose, and other integrations.
- Allow each SpatialData element to choose the least expensive safe transfer
  route independently.
- Support local stores first without preventing HTTP, S3, or future Python
  chunk providers.
- Make source lifetime, readiness, failure, revision, and persistence explicit.
- Preserve standalone native SpatialData opening during migration.
- Reject unsupported transforms or sources clearly rather than silently
  approximating them.

## Non-Goals

- Embedding Python in the Odon process.
- Sending full image arrays through JSON-RPC.
- Requiring a complete SpatialData store to be rewritten before display.
- Making Python responsible for rendering or per-frame drawing commands.
- Reimplementing arbitrary Dask graphs in Rust.
- Treating a session-backed Python object as project-durable without an
  explicit persistence operation.
- Supporting every vendor reader or every SpatialData element in the first
  milestone.
- Removing the native Rust SpatialData opener before a standalone replacement
  has been agreed and verified.
- Supporting non-affine spatial transformations in the first milestone.

## Baseline Before Implementation

The following records the starting point that motivated this design. It is
retained as historical context; the implemented status and current boundaries
are summarized above and in the phase statuses below.

### Native SpatialData path

`datasets.open_spatialdata` accepts a local root path and selected element
names. The Rust implementation discovers conventional `images`, `labels`,
`points`, `shapes`, and `tables` subtrees, reads Zarr v2 or v3 node attributes,
recognizes expected encoding markers, assumes current Parquet payload
locations, and reduces transforms to 2D scale and translation.

This path is useful native compatibility, but it couples Odon to SpatialData's
container layout and element encodings. The Python SDK currently exposes this
same path directly, so it does not yet provide an independent Python-led
integration.

### Generic data and layer path

The existing resource registry already provides stable IDs, URI references,
formats, coordinate metadata, ownership, provenance, revisions, project
round-tripping, and session cleanup. External layers support image, labels,
objects, points, shapes, masks, and annotations.

Important gaps are:

- a resource has a URI but no explicit store-root/group-path distinction;
- registered images are installed primarily as external overlays rather than
  opened as a primary document resource;
- external image realization currently requires a local file URI;
- the coordinate contract represents scale and translation, not a full affine;
- there is no atomic generic composite-document open operation;
- descriptor registration can complete before renderer resource readiness;
- table/property resources are not yet a complete generic join mechanism; and
- `register_numpy()` writes a temporary Zarr store and is not a zero-copy or
  streaming path.

### Reusable Rust capabilities

The implementation should reuse rather than replace:

- local and remote OME-Zarr stores and datasets;
- the actor-owned open/task lifecycle;
- native tile decoding, caches, prefetch, and memory controls;
- external resource and layer identity/ownership;
- native object and point loaders;
- viewer loading and presentation readiness;
- project resource persistence; and
- sync/async Python task and event semantics.

## SpatialData Source Classes

The adapter must distinguish these cases rather than treating every
`SpatialData` object identically.

### Backed and self-contained

The selected element is stored in a local or remote SpatialData Zarr store and
its data dependencies are contained in that store. The adapter can prove the
store and group identify the pixels represented by the Python element.

Preferred policy: `reference`.

Python sends the store URI, group path, resolved transform, and provenance.
Odon reads the existing chunks. No image copy or Python image computation is
required.

### Backed but not self-contained

The `SpatialData` object has an associated store, but a selected lazy element
depends on files outside that store or on a wider Dask graph. The associated
SpatialData path alone is not sufficient proof that Odon would read the same
pixels.

Preferred policy: reference a provably equivalent original source when Odon
supports it; otherwise use a provider or require explicit materialization.

### Vendor-reader result

`spatialdata-io` or application code has created a `SpatialData` object over
TIFF, HDF5, vendor Zarr, Parquet, CSV, or other inputs. There may be no canonical
SpatialData store.

Preferred policy: resolve each element separately. A TIFF or OME-Zarr image can
be referenced directly if Odon supports it, while in-memory shapes may be
adapted to GeoParquet. Do not copy a large image merely because a smaller
overlay needs adaptation.

### In-memory or computed

The selected element has no source Odon can read, or its Python value differs
from every backing source.

Preferred policy: session provider when available. Before that milestone,
require the caller to request a snapshot or report that the element cannot be
referenced. `auto` must not silently write a large image.

## Generic Resource Contract

The exact protocol names must be frozen during Phase 0, but the semantic model
should include the following.

### Store and element addressing

A referenced Zarr resource needs two distinct fields:

```json
{
  "source": {
    "kind": "uri",
    "uri": "file:///data/sample.zarr",
    "group_path": "images/morphology"
  },
  "format": "ome-zarr",
  "kind": "image"
}
```

The store root and group path must not be conflated into an operating-system
path. Explicit separation supports local directories, HTTP stores, S3 prefixes,
Zarr v2/v3, and multiple elements in one store.

Requirements:

- `group_path` is relative, normalized, and cannot escape the store root;
- the actor validates source accessibility and actual metadata;
- credentials are referenced through session-owned credential state and never
  embedded in a project descriptor;
- one store may back several resources without duplicating store/runtime state;
- snapshots report normalized source identity and readiness; and
- project persistence requires a durable, reopenable source.

### Image and label descriptors

The primary source of truth remains the NGFF metadata read by Rust. Python may
provide validated expectations and SpatialData-derived presentation metadata:

- element name and kind;
- axes and units;
- selected target coordinate system;
- element-to-document affine transform;
- expected shape, dtype, channel names, and multiscale count;
- source generation/fingerprint where available; and
- provenance identifying the SpatialData store, element, and adapter version.

Rust compares relevant expectations with the opened resource and returns a
structured mismatch rather than rendering ambiguous data.

### Coordinate contract

Replace the scale/translation-only resource boundary with a generic affine
contract. It must define:

- input and output axes;
- matrix shape and axis ordering;
- transform direction;
- pixel-centre convention;
- units;
- reference document coordinate system; and
- behavior for axes not present in a particular element.

Python resolves the SpatialData coordinate graph into this contract. Rust does
not interpret SpatialData transformation sequences.

For the first shippable slice, unsupported rotation, shear, axis mapping, or
dimensionality must fail capability validation. They must not be discarded.
Generic affine raster rendering can then expand the accepted set without a
SpatialData-specific change.

### Geometry, points, and properties

Generic vector resources should use Odon-supported Parquet or GeoParquet
contracts with explicit geometry, coordinate, identifier, and property columns.
The adapter may directly reference a compatible stored payload or materialize
only the selected vector element.

Tables require a generic property association contract:

```text
property resource
  target layer ID
  target key column
  property key column
  selected scalar/categorical columns
  optional selected feature values
```

The first implementation may join selected table columns into a temporary
GeoParquet object resource. A later keyed property resource should update
properties without rewriting geometry and should support label-index joins.

## Generic Document Composition

Opening the primary image and installing related layers should be one retained,
actor-owned operation. Candidate API names include `datasets.open_resource`,
`documents.open`, or `scenes.open`; Phase 0 must choose one canonical domain.

Illustrative Python, not a frozen signature:

```python
image = app.data.register(
    "file:///data/sample.zarr",
    format="ome-zarr",
    group_path="images/morphology",
    coordinate_space=image_space,
)
labels = app.data.register(
    "file:///data/sample.zarr",
    format="ome-zarr",
    group_path="labels/nuclei",
    coordinate_space=labels_space,
)

task = app.documents.open(
    primary=image,
    layers=[odon.LayerSpec(labels, kind="labels", name="Nuclei")],
)
document = task.wait()
```

Required semantics:

- the primary image becomes the document resource, not an overlay on an
  unrelated previously open dataset;
- all descriptors are validated before commit where practical;
- failure reports the resource and phase that failed;
- cancellation is cooperative and does not leave an unowned partial scene;
- resource readiness and presentation readiness remain distinguishable;
- one commit establishes stable resource/layer IDs and ordering;
- the result reports document, resource, layer, coordinate-system, and task
  identities; and
- sync and async SDKs expose the same contract.

## Python SpatialData Integration

### Packaging

The integration must remain optional because the base `odon-client` package has
no required third-party runtime dependencies and `spatialdata` has a substantial
scientific dependency graph.

Phase 0 must choose between:

- an `odon-client[spatialdata]` extra with an `odon.integrations.spatialdata`
  module; or
- a separately versioned `odon-spatialdata` distribution.

A separate distribution offers the clearest independent compatibility matrix;
an extra offers the simplest installation. Either choice must keep imports of
the base SDK functional without SpatialData installed.

### Proposed user surface

Illustrative API:

```python
import odon
import odon_spatialdata

with odon.connect() as app:
    display = odon_spatialdata.display(
        app,
        sdata,
        image="morphology",
        labels=["nuclei"],
        shapes=["cells"],
        points=["transcripts"],
        tables=["table"],
        coordinate_system="global",
        transfer="auto",
    )
    display.wait_ready()
```

The initial display handle should expose:

- the retained open task and final document snapshot;
- mapping from SpatialData element names to Odon resource and layer IDs;
- selected coordinate system and resolved transforms;
- actual transfer policy chosen per element;
- source, resource, model, and presentation readiness;
- structured warnings and unsupported-element reasons; and
- explicit `close()` cleanup for session-owned adapted resources.

The implemented handle includes generation-safe `refresh()`, selection-ID
round-tripping, cleanup, and explicit durable promotion. A future keyed-property
resource may add property-only replacement. These operations retain
snapshot/revision semantics rather than imply automatic shared mutable memory.

### Selection rules

- A primary image is required for the first milestone.
- The adapter may select an image automatically only when exactly one legal
  candidate exists.
- The caller chooses the target coordinate system unless only one unambiguous
  common system exists.
- Elements that cannot map into the selected system are reported before open.
- Default table projection is conservative; large expression matrices are not
  copied wholesale.
- Every omitted or downgraded element produces a structured reason.

### Adapter invariant

The adapter must not call `datasets.open_spatialdata`. Tests should enforce this
by using a control server/build where that method is unavailable while generic
resource/document methods remain available.

## Transfer Policy

Expose an explicit policy with behavior similar to:

| Policy | Behavior |
| --- | --- |
| `reference` | Require proven Odon-readable backing; never materialize or stream. |
| `provider` | Require a live session provider; never write a snapshot. |
| `snapshot` | Explicitly materialize selected incompatible elements and own their lifecycle. |
| `auto` | Prefer reference, then provider; never silently snapshot a large raster. |

The result records the chosen policy for every element.

### Direct-reference proof

The adapter may use `reference` only when it can establish that:

- Odon can access the source URI;
- the selected element is backed by the identified store/group or another
  equivalent supported source;
- the Python element's pixel values have not diverged from that source;
- the source metadata and codecs are supported by the connected Odon build;
- the selected transformation is representable by Odon; and
- the source will remain valid for the intended ownership lifetime.

If Python metadata has changed but pixel values remain backed, the adapter may
send the current resolved transform separately. If equivalence of pixel values
cannot be proved, `reference` fails rather than opening stale data.

### Provider path

The generic Python array provider exposes metadata and chunks on demand from
NumPy, Xarray, Dask, or an in-memory Zarr store. It is not specific to
SpatialData.

The first prototype may expose a session-authenticated, read-only virtual Zarr
store over loopback so Odon can reuse its existing Zarr storage and tile stack.
A later binary or local-socket protocol can replace that transport if measured
performance requires it.

Provider requirements:

- bind only to loopback or a local inter-process transport;
- use a session-scoped unguessable credential;
- never expose Odon's private control bearer token;
- provide bounded concurrent requests and backpressure;
- support cancellation and source generations;
- expose multiscale levels or an explicit single-level limitation;
- permit Odon tile caching without repeated Python computation;
- fail the resource clearly when Python disconnects; and
- remain session-owned and non-persistable until explicitly snapshotted.

### Snapshot path

Snapshots remain valuable for durability, reproducibility, unsupported source
formats, and operation after Python disconnects. They are not the default
interactive display path.

Snapshot behavior must specify:

- which elements are materialized;
- estimated bytes and available disk space before starting large writes;
- output format and format version;
- task progress and cancellation;
- temporary versus caller-selected durable location;
- cleanup behavior; and
- whether the resulting descriptor can become project-owned.

## Consistency and Lifetime

### Read-only source assumption

Direct-reference display initially treats source data as immutable for the
document generation. Concurrent Python writes to the same Zarr group can expose
mixed metadata and chunk generations unless the store provides stronger atomic
semantics.

The initial contract should require the source to be quiescent while opening.
Refresh creates a new resource/document generation; it does not mutate the
existing reader invisibly.

### Ownership

- A durable local/remote reference may be session-, project-, or user-owned if
  Odon can reopen it without ephemeral credentials.
- A Python provider is session-owned.
- A temporary adapted vector/table resource is session-owned unless promoted
  through an explicit persistence operation.
- A project-owned document cannot depend on a session provider or temporary
  directory.
- Closing a display removes its session-owned layers/resources only after the
  document has released them.

### Python disconnect

Direct durable references can continue if their ownership permits it. Provider
resources fail or are removed according to their disconnect policy. The viewer
must show an explicit unavailable-resource state rather than stale pixels that
appear current.

## Native Rust SpatialData Strategy

During migration:

- keep `datasets.open_spatialdata` for native standalone workflows;
- classify it as a compatibility adapter, not the canonical Python route;
- document and test the specific SpatialData layouts it supports;
- avoid expanding it solely to match Python SpatialData semantic features;
- share generic readers and renderers with the new resource path; and
- do not deprecate it until the product decision for no-Python desktop users is
  settled.

Possible long-term outcomes are:

1. retain the Rust opener as a conservative best-effort path;
2. have the native UI invoke a managed Python adapter sidecar; or
3. remove native SpatialData discovery after an explicit support transition.

That decision is independent of making the Python adapter canonical.

## Implementation Phases

### Phase 0 — Freeze boundaries and fixtures

Status: complete.

- Choose canonical protocol/resource/document method names.
- Define store-root/group-path addressing and path-safety rules.
- Define the affine coordinate contract and supported first-slice subset.
- Choose optional-extra versus separate-distribution packaging.
- Define supported Python `spatialdata` versions and fixture matrix.
- Inventory local and remote backed stores, vendor-reader outputs, dirty
  elements, Zarr v2/v3, codecs, shapes, points, labels, and tables.
- Add this initiative to the application-surface manifest and coverage matrix.
- Record which existing Rust SpatialData behaviors remain compatibility-only.

Exit criterion: reviewed schemas, error taxonomy, capability entries, and test
fixtures exist before implementation commits freeze accidental API names.

### Phase 1 — Generic referenced Zarr elements

Status: complete for local, HTTP, and primary S3 resources.

- Add explicit group paths to referenced Zarr resources.
- Reuse one store/runtime for several groups in the same source.
- Open a registered image resource as the primary document.
- Load registered image/label groups from local sources through the actor.
- Validate metadata, axes, type, and supported transform before commit.
- Report resource readiness separately from presentation readiness.
- Add typed sync and async Python wrappers.
- Add resource replacement/generation behavior.
- Preserve project/session ownership invariants.

Exit criterion: Python can open an arbitrary supported OME-Zarr group as the
primary document and a second group as a label/image layer without a
SpatialData-specific method or an intermediate copy.

### Phase 2 — Backed SpatialData image/label adapter

Status: complete.

- Add the optional SpatialData package/module.
- Enumerate images and labels with official Python APIs.
- Resolve a caller-selected coordinate system.
- Prove direct-reference eligibility for local backed elements.
- Register the existing store/group and open a generic document.
- Return the typed display handle and per-element policy report.
- Reject unbacked, dirty, unsupported-codec, and unsupported-transform cases
  with structured alternatives.
- Ensure the adapter works when `datasets.open_spatialdata` is unavailable.

Exit criterion: a real backed SpatialData object displays its selected image
and label directly from the original store, with no temporary image store and
no Python pixel computation.

### Phase 3 — Composite vectors and tables

Status: complete for direct/snapshot vectors and selected-column table
projection. The generic keyed-property optimization remains deferred.

- Add atomic multi-resource document composition.
- Resolve and attach selected shapes and points.
- Direct-reference compatible Parquet/GeoParquet payloads where safe.
- Materialize only incompatible selected vector elements when explicitly
  allowed.
- Project selected table annotations onto object properties.
- Add a generic keyed-property resource design for geometry-independent
  updates and label properties.
- Report complete element-to-resource/layer provenance.

Exit criterion: one backed SpatialData object can produce a coordinated Odon
document with image, labels, shapes, points, and selected table properties;
only elements that require adaptation are copied.

### Phase 4 — Remote stores and complete affine support

Status: complete for generic affine rendering, HTTP composites, and primary S3
resources. Credential-safe S3 secondary raster presentation remains deferred.

- Support group-addressed HTTP and authenticated S3 resources.
- Reuse credential sessions without persisting secrets.
- Add generic raster/geometry affine rendering needed by SpatialData mappings.
- Test axis mapping, rotation, shear, extra images, and multiple coordinate
  systems.
- Add capability negotiation for codecs, Zarr versions, and transform kinds.

Exit criterion: supported remote backed stores and affine-aligned multi-element
scenes use direct reference with the same semantics as local stores.

### Phase 5 — Python chunk provider

Status: complete for the loopback OME-Zarr transport, bounded request-time
failure, and normal display cleanup. A proactive viewer-level disconnect state
and any binary/local-socket replacement remain contingent follow-ons.

- Define and benchmark a generic read-only array-provider transport.
- Implement provider-backed image and label resources.
- Support Xarray/Dask/NumPy and in-memory Zarr adapters.
- Add bounded scheduling, cancellation, generations, and failure projection.
- Add optional memory/disk chunk caching without whole-image materialization.
- Define provider shutdown and display cleanup.

Exit criterion: a genuinely in-memory or computed SpatialData raster can be
viewed without first writing a complete OME-Zarr store, while Python remains
connected.

### Phase 6 — Persistence and round-trip workflows

Status: complete for explicit promotion, project descriptors, generation-safe
refresh, and object selection IDs. Property-only refresh remains deferred with
the keyed-property resource.

- Promote selected session resources to durable snapshots explicitly.
- Save/reopen project-owned composite documents.
- Refresh selected elements into a new generation.
- Round-trip Odon object selections into caller-managed Python data.
- Refresh selected object properties without rewriting geometry where the
  keyed-property contract permits it.
- Decide the long-term native Rust SpatialData opener policy.

Exit criterion: interactive adapter sessions can deliberately become durable
projects and can exchange selected semantic results without implying shared
mutable memory.

## Testing Strategy

### Rust contract and resource tests

- local Zarr v2 and v3 store root plus nested group paths;
- normalized group paths and traversal rejection;
- multiple resources sharing one store;
- image versus labels validation;
- primary-resource open without a UI frame;
- task completion, cancellation, and structured source errors;
- resource/model readiness before presentation readiness;
- revision conflict and resource replacement behavior;
- session cleanup and project persistence validation;
- unsupported codec, axis, dimensionality, and transform errors; and
- local, HTTP, and S3 behavior as their phases land.

These tests use generic OME-Zarr fixtures and should not import or understand
SpatialData.

### Python adapter tests

- supported current and oldest-supported `spatialdata` releases;
- single-scale and multiscale images;
- labels, shapes, points, and tables;
- local backed and self-contained elements;
- backed but non-self-contained elements;
- unbacked and computed elements;
- in-memory metadata transform changes;
- ambiguous image and coordinate-system selection;
- per-element transfer policy and provenance;
- sync and async parity where the integration exposes both; and
- absence of any call to `datasets.open_spatialdata`.

Version-specific shims must remain isolated in the adapter and have explicit
tests identifying the upstream versions they support.

### End-to-end no-copy evidence

The backed-image acceptance test must demonstrate all of the following:

- the source image store exists before the adapter call;
- the adapter creates no temporary raster store;
- Python does not compute or iterate image chunks;
- Odon opens the exact source URI and group path;
- Odon's tile counters show Rust-side reads and decoded cache entries;
- the image reaches resource and presentation readiness; and
- Python memory does not grow in proportion to the full image size.

A spy/guarded Dask graph or store should fail the test if Python attempts pixel
computation. A Rust-side instrumented store should prove that Odon performed
the reads.

### Visual and semantic integration tests

- image/label/shape/point alignment in a shared coordinate system;
- channel identity and contrast defaults;
- label and object selection behavior;
- table-derived categorical and continuous styling;
- screenshot evidence for representative fixtures;
- cleanup after display close and Python disconnect; and
- clear unavailable-resource presentation after provider failure.

## Acceptance Criteria

### First shippable milestone

- A Python `SpatialData` adapter opens a selected backed local image directly
  from its existing SpatialData Zarr store.
- Python supplies the exact store URI, group path, and resolved coordinate
  metadata; Rust performs all pixel IO.
- The operation does not create a second image store or compute the raster in
  Python.
- The same generic path opens a non-SpatialData OME-Zarr group.
- A selected backed labels element can be added from the same store.
- Unsupported sources and transforms fail before misleading presentation.
- Resource, model, and presentation readiness are observable.
- The adapter never calls `datasets.open_spatialdata`.
- Sync and async generic SDK contracts have parity.
- Existing native SpatialData opening continues to work at its documented
  compatibility level.

### Complete adapter milestone

- Images, labels, shapes, points, and selected table properties can be composed
  into one Odon document.
- Transfer policy is explicit and reported per element.
- Compatible backed resources are never copied merely because they originated
  inside SpatialData.
- Remote backed resources work through supported credential and store paths.
- In-memory/computed rasters can use a session provider without whole-image
  disk materialization.
- Durable project save refuses ephemeral dependencies or promotes them only
  through an explicit snapshot operation.
- Compatibility with supported `spatialdata` releases is enforced in Python CI
  rather than through SpatialData-specific Rust parser branches.

## Risks and Mitigations

### Upstream backing details may not be public enough

The official Python API may not expose a stable mapping from every element to
its exact backing store/group. Prefer public APIs. Keep unavoidable
version-specific extraction in one adapter module, fail closed, and consider
contributing the required backing-descriptor API upstream.

### A backed object may have diverged in memory

`is_backed()` alone does not prove that the selected Python value is identical
to the stored element. Require a direct-reference proof, record the source
generation where possible, and fail to provider/snapshot choices when
equivalence is uncertain.

### OME-NGFF and codec support still evolves

This plan removes SpatialData container interpretation from Rust; it does not
remove the need to maintain OME-NGFF/Zarr readers. Advertise supported Zarr,
NGFF, dtype, dimensionality, sharding, and codec capabilities through runtime
introspection and return structured incompatibility errors.

### Transform support can produce plausible misalignment

Never drop unsupported transformations. Python resolves the graph; Rust
validates the resulting affine capability. Add screenshot and coordinate-probe
tests before accepting additional transform kinds.

### Sequential resource calls can expose partial documents

Use an actor-owned composite open transaction before claiming full multi-element
support. Until then, the adapter must clean up its own partial session resources
on failure and report the limitation.

### Python lifetime can be mistaken for durable ownership

Provider-backed and temporary adapted elements remain visibly session-owned.
Project save validates every dependency. Durable promotion is always explicit.

### Standalone users may still need native discovery

Keep the existing native opener during migration and make its supported layout
boundary explicit. Do not use the Python-led architecture as an implicit
decision to require Python for every Odon user.

## Documentation and Surface Updates

Each implementation phase must update, as applicable:

- `api/application-surface.json`;
- `api/state-ownership-ledger.json`;
- `docs/design/test-coverage-matrix.md`;
- `docs/reference/python-api.md`;
- `docs/reference/python-api-contracts.md`;
- generated Python API member reference;
- `docs/advanced/python-api-limitations.md`;
- `docs/data-sources/spatialdata-and-xenium.md`; and
- the Python package installation and compatibility documentation.

User documentation must distinguish:

- native Rust compatibility opening;
- Python-led direct-reference SpatialData display;
- session provider display;
- explicit snapshot/persistence; and
- supported versions, transforms, stores, and codecs.

## Phase 0 Decisions

1. The integration ships as the optional `odon-client[spatialdata]` extra and
   lazy `odon.integrations.spatialdata` module. Base SDK imports remain free of
   scientific dependencies.
2. `datasets.open_resource` is the canonical primary-resource and composite
   document-open method.
3. A referenced Zarr source is represented as a store `uri` plus a normalized
   relative `group_path`. Local `file:`, HTTP(S), primary S3 with session-held
   credentials, and loopback provider sources use this contract.
4. The initial Python compatibility window is `spatialdata>=0.6,<0.9`; CI must
   cover its oldest and newest supported releases. Rust validates the concrete
   Zarr/NGFF metadata and codecs it opens rather than trusting a Python version.
5. The released coordinate boundary is an invertible 3×3 homogeneous 2D
   input-to-reference affine with explicit axes and pixel-centre convention.
   Rotation, shear, reflection, scale, and translation are supported;
   non-affine or unsupported-dimensional mappings fail closed.
6. The first table-property projection joins explicitly selected AnnData
   `obs` columns into the selected shapes GeoParquet by `instance_id`. A keyed
   property resource is a later optimization.
7. Provider transport is a loopback-only, tokenized, read-only virtual
   OME-Zarr store with bounded concurrency and an LRU chunk cache. A different
   transport requires benchmark evidence.
8. Native `datasets.open_spatialdata` remains a standalone compatibility path;
   the Python adapter is permanently prohibited from calling it.

## Recommended First Vertical Slice

The smallest slice that proves the architecture is:

1. extend a data resource with `group_path`;
2. add a generic task that opens a registered local OME-Zarr image resource as
   the primary document;
3. implement an optional Python helper that accepts a backed `SpatialData`,
   resolves one image element, and registers its existing store/group;
4. prove that Python never computes image chunks and creates no temporary
   raster store;
5. verify Rust-side tile reads, readiness, and a rendered screenshot; and
6. run the same generic open path on an ordinary non-SpatialData OME-Zarr group.

That slice validates the central decision before work expands into vectors,
tables, remote stores, affine transforms, or providers.
