# Continuous Numeric Object-Property Styling Plan

Status: implemented on macOS; Windows and Linux GPU smoke remains a release-validation task

Date: 2026-08-25

## Outcome

Odon will colour segmentation objects directly from numeric properties without requiring Python
to manufacture categorical quantile columns. The native UI, Python API, actor model, saved state,
single-view and multi-viewport renderers will share one declarative colour-mapping contract.

The motivating workflow is comparison of mean Channel 1 intensity across the Cellpose and
InstanSeg results. Python will calculate and attach the numeric measurement once, then give all
four views the same explicit domain and palette. Equal colours will therefore mean equal numeric
values across segmentations.

Multi-window application support is not part of this work.

## Original Limitation

`color_property` currently means categorical colouring. Odon converts each distinct property
value to a string, builds one geometry/fill group per value, assigns a hashed colour, and exposes
checkbox and colour overrides for those values. This is appropriate for phenotype or cluster
labels but not for measurements with thousands of distinct values.

Numeric columns already exist in the object property store and analysis API. The missing pieces
are a continuous presentation model, numeric colour payloads, a renderer path that does not create
one draw group per value, and a continuous legend/editor.

## Public Contract

Add a tagged `color_mapping` object to object style. Existing top-level `color_rgb`,
`color_property`, and categorical legend calls remain valid and keep their current behaviour.

```python
app.objects.set_style(
    fill_cells=True,
    fill_opacity=0.72,
    color_mapping={
        "mode": "continuous",
        "property": "mean_channel_1",
        "palette": "viridis",
        "domain": [4_000.0, 42_000.0],
        "scale": "linear",
        "reverse": False,
        "out_of_range": "clamp",
        "missing_color_rgb": None,
    },
)
```

The same value is accepted by `viewport.objects.set_style(...)` and the async API. Mosaic object
style will use the same contract; it will not introduce a separate colour model.

### Mapping modes

- `single`: use the existing object colour.
- `categorical`: use a property plus the existing per-value visibility and colour overrides.
- `continuous`: map a numeric property through a numeric domain and colour palette.

Supplying the legacy `color_property="phenotype"` without `color_mapping` continues to select
categorical mode. Style snapshots return the canonical `color_mapping` in addition to legacy
fields needed by existing clients.

### Continuous fields

| Field | Contract |
| --- | --- |
| `property` | Non-empty property name. It must resolve to numeric data when the object resource is ready. |
| `palette` | Initially `viridis`, `magma`, `plasma`, `inferno`, `cividis`, `turbo`, or `gray`; alternatively a custom list of colour stops. |
| `domain` | Two finite numbers with `min < max`, or `"auto"`. Auto uses the full unfiltered source minimum and maximum. |
| `scale` | `linear` or `log10`. Log domains and auto-domain samples must be strictly positive. |
| `reverse` | Reverse the selected palette without changing the domain. |
| `out_of_range` | `clamp` to endpoint colours or `hide` values outside the domain. |
| `missing_color_rgb` | RGB colour for null, non-numeric, or invalid values; `null` means transparent. |

Custom stops contain a position from 0 to 1 and an RGB triplet, contain at least two entries, are
strictly ordered, and include positions 0 and 1. The actor validates the complete mapping
atomically before committing it.

Auto domains are deliberately based on the full source, not the active filter, so colours do not
shift when a filter changes. The style response reports the resolved domain, numeric count, and
missing count. Python should use an explicit domain whenever several results must be compared.

Constant-valued properties use the palette midpoint. Non-finite values never enter domain
statistics. A requested continuous property that is unavailable remains canonical but reports an
unavailable presentation state until its lazy column loads; it must not silently fall back to
categorical colours.

## Architecture

### 1. Shared typed presentation model

Introduce serializable Rust types for `ObjectColorMapping`, `ContinuousColorMapping`,
`ContinuousDomain`, `ContinuousScale`, `OutOfRangeMode`, and palette/custom stops. Put validation
and colour interpolation in a renderer-independent module.

The actor owns the requested mapping and its resolved domain. Renderer projections consume an
immutable, generation-tagged mapping. Native UI actions submit the same style command as Python;
the UI must not mutate renderer state first.

Add numeric summaries to `ControlObjectResource` while the object source is parsed. Min/max/count
is sufficient for the first implementation and makes auto-domain resolution constant-time on the
actor thread. Lazy numeric columns update the summary when they materialize.

### 2. Backward-compatible state and persistence

Extend the default object snapshot, style patch validation, per-viewport projection, native-layer
presentation, and mosaic object style with `color_mapping`.

Extend `ObjectProjectDisplayState` with an optional typed mapping using Serde defaults. Old project
state migrates as follows:

- no `color_property_key` becomes single mode;
- `color_property_key` plus existing overrides becomes categorical mode;
- a new `color_mapping` takes precedence when present.

This is an additive migration and does not require a project format version increase. Saved views
must retain the complete mapping and not reduce a continuous configuration to `cell_color_by`.
Deep links carry the complete typed mapping in one JSON-encoded `object_color_mapping` query
parameter (`object_colour_mapping` is accepted as a parsing alias). This keeps custom stops,
missing-value handling, reversal, domain, and scale atomic rather than splitting them among
loosely coupled query fields.

### 3. Per-object colour payload

Do not adapt the categorical group builder to continuous values. It would produce thousands of
groups and draw calls.

Build a cached `ObjectColorPayload` containing one RGBA value per object plus its source/mapping
generation. The payload is produced from the numeric column and canonical mapping. Filtering
changes visibility without recomputing numeric values; palette, domain, or property changes
invalidate only the colour payload, not geometry.

The GPU polygon-fill and line renderers will upload this payload as an RGBA8 object-index texture.
Their existing object IDs index the colour texture, allowing all continuously coloured fills or
outlines to render in one geometry pass. Selection state remains a separate texture and selected
or primary styling overrides the property colour.

The CPU fallback reads the same payload. Far-zoom proxy points receive matching per-point colours
rather than quantizing back to categorical groups. Missing, hidden, filtered, and out-of-range
objects use alpha zero where required.

Categorical rendering remains on its existing path for the first milestone. Once continuous mode
is stable, using the same indexed-colour payload for categorical objects can be considered as a
separate performance simplification.

### 4. Native controls and legend

Replace the single `Color by` selector with a mode-aware control:

1. select `Single`, `Categorical`, or `Continuous`;
2. list categorical candidates for categorical mode and numeric candidates for continuous mode;
3. for continuous mode, edit palette, reverse, auto/manual minimum and maximum, out-of-range
   behaviour, and missing colour;
4. show a gradient legend with the property name and formatted minimum, midpoint, and maximum.

The current category checklist and overrides remain visible only in categorical mode. Numeric
columns with few distinct values may appear in both property lists; Odon will not guess the mode
from cardinality.

Extract the existing annotation Turbo mapping into the shared colour-map module where practical,
so annotations and objects agree on named palette output. Changing annotation presentation itself
is not required for this milestone.

### 5. Python API

The generic `set_style(color_mapping=...)` call works immediately through the actor schema. Add
sync and async convenience methods with identical arguments:

```python
app.objects.color_by_continuous(
    "mean_channel_1",
    palette="viridis",
    domain=(4_000.0, 42_000.0),
    fill_cells=True,
    fill_opacity=0.72,
)
```

Provide the same helper on explicit viewport object resources and mosaic objects. Keep custom
palette stops as mappings rather than introducing a mandatory Python plotting dependency.

The runtime method schema must describe the nested union and enumerate named palettes and modes;
the generated API manifest/reference is then updated from that schema.

## Implementation Sequence

### Milestone 1: contract and actor state

- Add the typed mapping and pure interpolation/normalization functions.
- Add numeric property summaries to actor object resources.
- Validate and expose `color_mapping` through active-view and explicit-viewport style methods.
- Add projection, revision, unavailable-state, project round-trip, and legacy migration tests.
- Extend mosaic actor style state with the same validated mapping.

Exit condition: Python can commit and read back a continuous style while Odon is covered, and the
state survives project save/open even before the renderer consumes it.

### Milestone 2: accurate rendering

- Build generation-keyed per-object RGBA payloads from inline and columnar numeric properties.
- Add indexed-colour textures to polygon fills and object outlines.
- Add the corresponding CPU fallback and proxy-point path.
- Preserve filters, opacity, selection overlays, transforms, and fast-rendering LOD behaviour.
- Reuse immutable geometry and colour payloads across viewports when mapping signatures match;
  retain independent payloads when domains or palettes differ.

Exit condition: numeric fills and outlines are visually continuous at every LOD, and changing
domain/palette does not rebuild or duplicate object geometry.

### Milestone 3: UI, SDK, and persistence completion

- Add the continuous controls and gradient legend to single-view and mosaic properties panels.
- Add sync/async Python convenience methods and examples.
- Preserve the complete mapping in saved views and add deep-link fields.
- Update object workflow, Python contract, API reference, feature inventory, and limitations docs.

Exit condition: the same mapping can be created from native UI, Python, a saved view, and a deep
link, and every route produces the same actor state and rendered result.

### Milestone 4: motivating comparison and performance gate

- Remove the generated quantile category property from the Cellpose/InstanSeg comparison script.
- Retain only exact `mean_channel_1` measurements.
- Compute one shared explicit domain and apply it to all four segmentation results.
- Capture full-window screenshots showing at least two results and the numeric gradient legend.
- Record render timing and cache counters with roughly 25,000 to 45,000 objects per result.

Exit condition: switching among all four segmentations preserves Channel 1, domain, palette, and
camera; identical numeric values have identical colours; interaction remains smooth.

## Verification Matrix

| Layer | Required checks |
| --- | --- |
| Colour math | Exact endpoints and interpolation, reversed and custom palettes, constant domains, missing values, clamp/hide. |
| Actor | Validation failures are atomic; explicit viewport styles remain independent; revisions/events are correct; covered-window commands complete. |
| Properties | Numeric detection works for GeoJSON and lazy GeoParquet columns; auto summary excludes null and non-finite data. |
| Projection | Latest mapping wins after delayed frames; unavailable properties recover after load without losing configuration. |
| Renderer | CPU/GPU colours agree within one RGB unit; fills, outlines, proxy points, filters, transforms, and selections agree. |
| Persistence | New mapping round-trips; old categorical projects load unchanged; saved views preserve explicit domains. |
| Python | Sync/async wrappers send identical payloads; method-schema and API-manifest evidence include every field. |
| Performance | Continuous mode has bounded draw calls, no per-distinct-value geometry, no object-geometry duplication, and no panel/camera interaction regression. |
| Platforms | GPU smoke on macOS, Windows, and Linux after deterministic Rust tests pass. |

## Acceptance Criteria

- A numeric property can be styled continuously without adding a derived categorical column.
- An explicit numeric domain produces identical colours across different object sources and
  viewports.
- Categorical styling and legend overrides remain backward compatible.
- Null and out-of-domain semantics are explicit and tested.
- Continuous styling covers fills, outlines, proxy points, filters, and selection overlays in GPU
  and CPU paths.
- Actor state, project persistence, saved views, native controls, Python, and mosaic presentation
  all use the same contract.
- Rendering cost depends primarily on visible geometry, not the number of distinct numeric values.
- The four-result Cellpose/InstanSeg comparison uses the native continuous capability and no
  quantile workaround.

## Implementation And Verification Evidence

All four implementation milestones are complete in the Rust model, actor schema, renderer,
native single-view and mosaic controls, persistence/deep-link paths, and sync/async Python SDK.
The legacy `color_property` contract remains categorical; canonical snapshots additionally expose
the tagged `color_mapping` union.

The motivating large-image workflow is
[`examples/python_instanseg_large_cycif_pilot.py`](../../examples/python_instanseg_large_cycif_pilot.py).
It independently measures every polygon and stores only the exact `mean_channel_1` value. It
compares Cellpose DAPI (26,412 objects), InstanSeg DAPI nuclei (32,324), InstanSeg multiplex nuclei
(42,817), and InstanSeg multiplex cells (43,386) using one explicit Viridis domain of
`[810.043, 44586.087]`. Every result has a measurement for every object; there is no quantile or
categorical display column.

MCP-driven macOS GPU smoke evidence:

- [Cellpose result and continuous legend](../assets/images/screenshots/raw/continuous-object-color-cellpose-macos.jpg)
- [InstanSeg multiplex-cell result and Properties controls](../assets/images/screenshots/raw/continuous-object-color-instanseg-macos.jpg)

The final 43,386-object capture reported document, object geometry, resources, canvas, and
presentation ready, with no pending work. The deterministic 45,000-object actor test retains one
shared object resource and proves camera projection does not copy its geometry. In the debug test
profile on this Mac, the ignored continuous-payload diagnostic built all 45,000 colours in
17.768 ms, reused the cached payload in 0.033 ms, and reported the exact 180,000-byte RGBA payload.
The frame-planning diagnostic reported 0.1077 ms single-view and 0.3138 ms split-view EMA with one
document, dataset, decoded-tile cache, and primary object-geometry instance. Timings remain
diagnostic output rather than brittle cross-platform assertions.

The final local run passed 503 deterministic Rust tests (five ignored diagnostics or unavailable
extended fixtures) and all 155 Python tests. These suites cover palette math, atomic validation, lazy and inline
properties, projection, GPU/CPU payload semantics, filters, selection, persistence, saved views,
deep links, SDK parity, generated API evidence, and the four-result workflow. Windows and Linux
GPU smoke is not claimed from this macOS workstation and remains part of cross-platform release
validation; it does not require a different public contract or architecture.
