# Odon Multi-Viewport Architecture and Implementation Plan

Status: first two-viewport milestone implemented and verified (2026-08-21)

Primary milestone: show the same OME-Zarr image and segmentation in two
side-by-side viewports, with linked navigation and independent object-property
fill styling

Related documents:

- `docs/design/complete-application-python-api-plan.md`
- `docs/design/control-protocol-v1.md`
- `docs/reference/python-api.md`
- `docs/design/test-coverage-matrix.md`

## Executive Summary

Odon should support multiple native Rust viewports inside one viewer workspace.
Python will describe the workspace, create and configure viewports, choose which
state is linked, and set each viewport's presentation. Python will not draw the
canvases or inject per-frame rendering instructions.

The first target is a comparison workspace:

```text
+------------------------------+------------------------------+
| Marker A                     | Marker B                     |
| same image                   | same image                   |
| same cells                   | same cells                   |
| fill = property A            | fill = property B            |
|                              |                              |
+------------------------------+------------------------------+
        linked camera, plane, and cell selection
```

This is not safely achieved by constructing two `OmeZarrViewerApp` values. The
current application combines dataset access, loaders, caches, GPU renderers,
camera state, channel presentation, object data and style, selections, tools,
panels, screenshots, and project state in one large type. Duplicating it would
duplicate expensive state and introduce conflicting project and editing
ownership.

Instead, the implementation should split the current single viewer into:

1. a shared document containing sources, scientific data, mutable annotations,
   loaders, and reusable caches;
2. lightweight viewport state containing camera, plane, presentation, canvas
   bounds, and transient interaction state;
3. a workspace containing the layout, active viewport, link groups, and shared
   shell UI; and
4. a render scheduler that unions and fairly prioritizes requests from all
   visible viewports.

The existing `viewer.*` API remains compatible by targeting the active viewport.
New `viewer.viewports.*` and `viewer.workspace.*` methods provide explicit
multi-viewport control. All of this belongs in the central Rust control layer,
so the native UI, Python SDK, and curated MCP adapter share the same semantics.

### Implementation status (2026-08-21)

The first two-viewport milestone described by this plan is implemented. The
shipping boundary is deliberately one or two same-document viewports in a
single horizontal or vertical split; nested grids and more than two views
remain later work.

Acceptance evidence:

- `src/viewports.rs` owns stable IDs, the two-view limit, layout validation,
  links, active-view state, local revisions, and restore/migration invariants.
- `OmeZarrViewerApp` keeps one document/resource owner and swaps independent
  navigation and presentation state for each canvas. Object selection and
  scientific edits remain shared, while only the active canvas can edit.
- CPU, raw-GPU, label, SpatialData-image, and TIFF request paths retain the
  union of both live views. Shared-resource counts and cache entry counts are
  exposed in `viewer.workspace.get` for runtime inspection.
- The central control registry, root dispatcher, sync Python client, and async
  Python client expose explicit workspace, viewport, and canonical fixed
  link-group resources. Legacy
  singleton calls continue to address the active viewport. MCP remains a
  deliberately curated subset rather than automatically exposing every new
  viewport operation.
- Workspace version 1 persists both viewports, links, local revisions, camera,
  plane, channel/group state, object style/filter state, and overlay
  presentation. Legacy project state migrates to one viewport.
- Per-viewport and whole-workspace screenshot targeting are implemented, with
  crop and queue behavior covered independently.
- The full Rust suite passes (219 tests plus 4 ignored diagnostic or
  external-fixture tests), as do all 86 Python tests. Focused tests cover the two-view scheduler,
  shared-key deduplication, independent presentation, link propagation,
  revision conflicts, persistence, screenshots, registry contracts, MCP
  compatibility, and sync/async Python parity.

The following ideas elsewhere in this document remain intentional expansion
points rather than first-milestone promises: recursive split trees, more than
two viewports, multiple named link groups, per-viewport selection sets,
detached windows, unrelated datasets, and workspace deep links.

## Product Goals

### Required for the first usable release

- One open single-dataset viewer can contain one or two viewports.
- The two viewports can be laid out horizontally or vertically.
- A viewport has a stable ID, optional title, and visible active-state marker.
- Both viewports share the open image source, object geometry, object property
  tables, masks, annotations, and tile resources.
- Camera and multidimensional plane can be independent or linked.
- Channel visibility, colors, contrast, object fill property, colormap,
  opacity, overlay visibility, and layer order can differ per viewport.
- Cell identity and selection are shared by default, so selecting a cell in one
  viewport highlights the same cell in the other.
- Only the active viewport receives destructive editing gestures.
- Existing single-viewport behavior and existing `viewer.*` clients continue to
  work without modification.
- Sync and async Python APIs can create, configure, link, inspect, and remove
  viewports.
- The implementation has automated state, control-contract, Python, renderer,
  and regression coverage.

### Later capabilities enabled by the architecture

- Grid and nested split layouts.
- More than two viewports.
- Independent selection sets or multiple named selection link groups.
- Linked cursor and crosshair display.
- Saving and restoring complete comparison workspaces.
- A mosaic canvas alongside a single-ROI canvas.
- Multiple datasets with compatible coordinate spaces.
- Multiple operating-system windows backed by one document.

These later capabilities should influence stable IDs and schemas, but they
should not enlarge the first milestone.

## Non-Goals for the First Milestone

- Running Python inside Odon.
- Letting Python render a canvas or send drawing commands every frame.
- Arbitrary egui widget trees inside viewport canvases.
- Comparing unrelated datasets or resolving arbitrary coordinate transforms.
- Multiple native windows.
- Making Mosaic itself multi-viewport.
- Duplicating all side panels for every viewport.
- Persisting a complete workspace before the in-memory state model is stable.
- Supporting more than two viewports or arbitrary nested layouts in the UI.

## Current Architecture Findings

### Root mode owns exactly one viewer canvas

`src/root_app.rs` currently models the application as one of:

```rust
enum Mode {
    Project { project_space: ProjectSpace },
    Single(OmeZarrViewerApp),
    Mosaic { mosaic: MosaicViewerApp, ret: ReturnToSingleState },
    Transition,
}
```

`RootApp::update` delegates to exactly one viewer. Control dispatch also matches
directly on `Mode::Single` and `Mode::Mosaic` throughout the file. There are
roughly 390 occurrences of these mode matches, so introducing a new mode and
copying every branch would be high-risk. A viewer facade or workspace boundary
is needed to contain this migration.

### The single viewer mixes five different kinds of ownership

`OmeZarrViewerApp` in `src/app.rs` owns all of the following:

- dataset/store/runtime and background loaders;
- decoded tile, texture, label, histogram, and pinned-memory state;
- one camera, one canvas rectangle, and one render/fallback generation;
- channel metadata and presentation;
- object geometry, properties, filtering, selection, style, analysis, and GPU
  renderers;
- masks, annotations, spatial layers, and Xenium layers;
- active layer, tools, gestures, undo, hover, and focus;
- side-panel tabs and other window-level UI;
- screenshot state and project persistence.

Its `ui_canvas` method both processes input and draws every content type. Calling
the method twice would overwrite `last_canvas_rect`, route gestures through
shared transient state, and render both canvases with the same camera and layer
presentation.

### Mosaic is not a multi-viewport implementation

`MosaicViewerApp` draws several sources into one canvas. It still has one
camera, one `last_canvas_rect`, one channel presentation, one active layer, and
one set of panels. Its source sharing and request scheduling are useful prior
art, but its user model is different from multiple independent canvases.

### Object data and presentation are inseparable today

`ObjectsLayer` currently contains:

- shared geometry, property storage, indexes, LODs, and loaded source state;
- global filtering and analysis caches;
- global selected object indices;
- presentation fields such as visibility, opacity, fill state, color mode, and
  `color_property_key`; and
- mutable CPU/GPU rendering caches.

Consequently, changing `cell_color_by` for one canvas changes it for every draw
of that layer. The motivating comparison requires this type to be split or
fronted by separate data, selection, analysis, and presentation components.

### The GPU raw-tile path is naturally shareable

`RawTileKey` identifies a view plane, pyramid level, tile coordinate, and
channel. It does not include channel color, contrast, or a render generation.
The uploaded raw tile can therefore be shared while each viewport supplies its
own `ChannelDraw` presentation during composition.

The CPU-composited path differs: `TileKey` contains a global `render_id`, and
the loader rejects work that is not part of its single latest render. That path
must gain multiple presentation consumers or a shared-raw/per-viewport-compose
equivalent. Multi-view must not work only on machines with the preferred GPU
path.

### Persistence and the control protocol assume one active view

`ProjectRoiViewState`, `ProjectViewSpec`, and deep links store one camera and
one set of channel/object presentation fields. The public API similarly exposes
singleton methods such as `viewer.camera.get`, and events use sources such as
`viewer:active`. Screenshots capture the one `last_canvas_rect`.

These are compatibility contracts. They should become active-viewport aliases,
not be removed or silently change meaning.

## Target Ownership Model

### High-level structure

```text
RootApp
  |
  +-- ViewerWorkspace
        |
        +-- ViewerDocument (shared)
        |     +-- dataset/source metadata
        |     +-- loaders and shared tile caches
        |     +-- object/mask/annotation data
        |     +-- scientific selections and edit history
        |     +-- project association
        |
        +-- ViewportCollection
        |     +-- ViewportState A
        |     +-- ViewportState B
        |
        +-- WorkspaceLayout
        +-- active_viewport_id
        +-- ViewportLinkController
        +-- shared panels/dialogs/task presentation
```

The names can change during implementation, but these ownership boundaries are
requirements.

### Shared document state

`ViewerDocument` owns data whose identity must not change merely because a new
view of it is created:

- `OmeZarrDataset`, store, remote runtime, dimensional metadata, and coordinate
  transforms;
- raw and label loaders;
- raw tile data and reusable GPU textures;
- source-level histogram/channel statistics caches;
- channel identity, names, notes, and immutable metadata;
- segmentation/object geometry, property columns, spatial indexes, and LODs;
- mask polygon data and annotation records;
- external data resources and provenance;
- stable layer and object IDs;
- project/ROI association;
- mutable scientific edits, undo/redo history, and dirty state; and
- long-running data/analysis tasks.

### Per-viewport state

Each `ViewportState` owns a stable `ViewportId` and:

- camera and last allocated canvas rectangle;
- view plane and x/y/z slice selection;
- channel visibility, order, color, window, and group presentation;
- chosen active layer for inspection;
- layer visibility, order, opacity, and presentation overrides;
- object fill property, color mapping, level visibility, fill opacity, outline
  style, and legend state;
- smooth-pixel preference, scale bar, HUD, and tile-debug visibility;
- per-viewport target-level and fallback continuity state;
- hover state and non-destructive in-progress selection gestures;
- screenshot targeting state; and
- local revision counters used by control operations.

Mutable annotation data must not be copied into this state.

### Workspace and shell state

`ViewerWorkspace` owns:

- ordered viewport collection;
- active viewport ID;
- validated layout tree;
- link groups and propagation policy;
- side-panel visibility and selected tabs;
- dialogs, file pickers, and top-level status;
- global tool choice; and
- the rule that only an active viewport may perform edits.

The existing left and right panels render once. Their camera, channel, layer,
and object-presentation controls inspect and mutate the active viewport. A
compact header on each canvas shows its title, link status, and active state.

### State classification decisions

| State | First milestone ownership | Rationale |
|---|---|---|
| Dataset/store/source transforms | Document | One scientific source and coordinate system |
| Raw tile and uploaded texture cache | Document | Avoid duplicate I/O, decoding, and GPU memory |
| Camera | Viewport, optionally linked | Required for independent or synchronized navigation |
| Plane/slice | Viewport, optionally linked | Supports synchronized or comparative plane views |
| Channel metadata and notes | Document | Same channel identity in every view |
| Channel visibility/color/window/order | Viewport | Core comparative presentation |
| Object geometry/property columns/index | Document | Same cells and property table |
| Object fill/color/legend style | Viewport | Required motivating use case |
| Object visibility filter | Viewport presentation | Enables comparison of subsets without copying data |
| Object selection identities | Document, linked by default | The same selected cells should highlight everywhere |
| Analysis task/results | Document | Avoid duplicate computation and ambiguous ownership |
| Analysis plot UI state | Workspace/active viewport | One side panel targets the active view |
| Mask and annotation records | Document | Edits are scientific data |
| Mask/annotation visibility and style | Viewport | Comparative presentation |
| Layer spatial registration | Document by default | Registration is a property of the source alignment |
| Active layer | Viewport | Each canvas may inspect a different overlay |
| Tool mode | Workspace | Prevent conflicting edit modes |
| In-progress edit gesture | Active viewport/workspace | Exactly one editor at a time |
| Undo/redo | Document | One ordered history of shared mutations |
| Panels and dialogs | Workspace | Render once, target active viewport |
| Screenshot settings | Workspace | Common output policy with explicit target |

The object-filter decision is intentionally presentation-local for visual
comparison. Operations that compute or export from a filter must explicitly
name the viewport or provide a standalone query; they must not infer a filter
from whichever canvas was last active.

## Core Rust Types

The following shapes illustrate the intended boundaries rather than fixing
exact field names:

```rust
pub struct ViewerWorkspace {
    document: ViewerDocument,
    viewports: IndexMap<ViewportId, ViewportState>,
    active_viewport_id: ViewportId,
    layout: WorkspaceLayout,
    links: ViewportLinkController,
    shell: ViewerShellState,
}

pub struct ViewportState {
    id: ViewportId,
    title: String,
    navigation: ViewportNavigation,
    presentation: ViewportPresentation,
    render_state: ViewportRenderState,
    interaction: ViewportInteractionState,
    last_canvas_rect: Option<egui::Rect>,
    revision: u64,
}

pub enum WorkspaceLayout {
    Viewport(ViewportId),
    Split {
        axis: SplitAxis,
        ratio: f32,
        first: Box<WorkspaceLayout>,
        second: Box<WorkspaceLayout>,
    },
}
```

Use an opaque serialized ID such as `viewport-<uuid>` or another collision-safe
identifier. Do not expose vector indices as identity.

### Canvas interface

Extract canvas input and output from the monolithic `ui_canvas` method:

```rust
struct CanvasFrameInput<'a> {
    viewport_id: &'a ViewportId,
    rect: egui::Rect,
    is_active: bool,
    document: &'a ViewerDocument,
    viewport: &'a ViewportState,
}

struct CanvasFrameOutput {
    activate: bool,
    navigation_change: Option<NavigationChange>,
    presentation_commands: Vec<ViewerCommand>,
    selection_command: Option<SelectionCommand>,
    edit_command: Option<EditCommand>,
    repaint: bool,
}
```

In practice egui and renderer borrowing may require a different split. The
important property is that one canvas cannot directly and invisibly mutate a
second canvas. Outputs are applied after drawing/input collection through
workspace commands and the link controller.

### Viewer facade for root dispatch

Do not add another set of hundreds of `RootApp` mode branches. Introduce a
small facade implemented by the single-dataset workspace and, where meaningful,
Mosaic. Existing root control handlers call the facade, which resolves either
the explicitly requested viewport or the active viewport.

This facade also provides a practical migration path from
`Mode::Single(OmeZarrViewerApp)` to `Mode::Viewer(ViewerWorkspace)` without
rewriting all control behavior at once.

## Rendering and Scheduling

### Shared raw resources, independent draw plans

For every frame, each visible viewport produces a `ViewportRenderPlan`:

- visible world rectangle;
- selected plane;
- target and fallback pyramid levels;
- required channel raw-tile keys;
- channel composition parameters;
- overlay draw parameters; and
- priority relative to interaction and active state.

The workspace scheduler then:

1. unions raw keys from all visible viewports;
2. deduplicates keys before submitting work;
3. retains active keys until no viewport needs them;
4. allocates a fair request budget across viewports;
5. prioritizes coarse readiness and the actively manipulated viewport without
   starving the other viewport; and
6. draws each canvas with its own composition and overlay presentation.

The current `set_active_keys` call cannot be invoked independently per viewport,
because the final call would discard the first viewport's keys. It must receive
the union from the workspace scheduler.

### GPU path

Keep one raw tile cache and texture upload path per document. Each viewport
passes separate channel colors/windows and transforms to composition. Audit the
shared offscreen buffers in `TilesGl`: consecutive paint callbacks must not
leak composition state, clear another canvas, or assume the canvas is the full
egui viewport.

### CPU path

Replace the singleton `latest_render_id` behavior with a multi-consumer model.
Acceptable implementations include:

- shared decoded raw tiles plus per-viewport CPU composition caches; or
- presentation-scoped composed keys and a scheduler that tracks the latest
  generation for every viewport rather than one global latest generation.

Whichever approach is chosen must:

- permit two live presentation generations simultaneously;
- deduplicate decode work;
- cancel generations per viewport;
- prevent responses from one viewport being mistaken for another; and
- retain the current coarse-to-fine fallback behavior independently per canvas.

### Object rendering

Refactor `ObjectsLayer` into conceptual components:

```text
ObjectDocument
  geometry, properties, indexes, LODs, source/loading state

ObjectSelectionState
  stable selected object IDs and named selections

ObjectAnalysisState
  result caches and task state

ObjectPresentation
  visibility, filter, fill property, color mapping, opacity, legend

ObjectRenderCache
  reusable geometry plus buffers keyed by data/presentation revisions
```

Geometry buffers should be shared. Property-derived color/fill buffers may be
cached by a presentation signature so equal viewport styles reuse them while
different styles coexist. A viewport presentation change must not mutate the
document or clear another viewport's render state.

### Performance acceptance criteria

For two identical, linked viewports showing the same channels and plane:

- opening the second viewport does not open the source a second time;
- raw tile requests are deduplicated by key;
- raw texture data is stored once;
- object geometry/property storage is stored once;
- only viewport state and presentation-specific buffers grow per viewport; and
- navigation remains responsive while both canvases stream tiles.

Record benchmark numbers before extraction and after the first two-viewport
implementation. Set numeric memory/frame-time budgets from those measurements
rather than guessing them in this plan.

## Interaction and Linking

### Active viewport

Clicking a canvas or its header makes it active. The active viewport:

- receives side-panel property edits;
- is the target of legacy `viewer.*` methods;
- receives keyboard navigation and edit shortcuts;
- is the default screenshot target; and
- is visually distinguishable without obscuring image data.

Hover alone must not change the active viewport.

### Link groups

Linking is explicit and field-based. The first milestone supports:

- `camera` — center and zoom;
- `plane` — plane mode and x/y/z selection;
- `selection` — documented as shared by default; and
- optionally `cursor` if it can be added without delaying the milestone.

Channel and object presentation are deliberately unlinked in the motivating
comparison. The schema should nevertheless allow later fields such as
`channels`, `layer_visibility`, and `timepoint`.

A navigation mutation is applied as one Rust transaction:

1. validate the source change;
2. calculate compatible target changes;
3. update the source and link-group members;
4. assign revisions;
5. publish one causally connected event batch; and
6. repaint affected canvases.

Do not implement links by listening to public events and issuing another public
command. That creates feedback loops, intermediate inconsistent frames, and
unclear revision conflicts.

Camera links use document world coordinates. A future multi-dataset link is
allowed only when coordinate-space compatibility can be proven or an explicit
mapping exists.

## Native UI Design

### Initial layout controls

Add a compact workspace control near existing viewer layout controls:

- Single view
- Split horizontally
- Split vertically
- Swap viewports
- Link/unlink navigation
- Close active viewport

Splitting clones the active viewport's presentation and navigation by default,
then gives the clone a new stable ID. This provides an immediately useful
comparison before the user changes one property.

### Canvas chrome

Each canvas gets a small, nonintrusive header or overlay containing:

- editable/default title;
- active-state indication;
- camera/plane link indicator; and
- close or context menu affordance.

The image drawing rectangle must exclude a persistent header if the header is
outside the canvas. Camera fitting and screenshots must use the actual image
rectangle, not the containing split cell.

### Panels and editing

Panels render once and show the active viewport's presentation. Shared data
edits should be labeled as affecting all views; presentation edits should be
labeled with the target viewport title when more than one viewport exists.

Only the active viewport can start a mask edit, transform, lasso, rectangle
selection, or similar tool gesture. Shared document changes become visible in
all viewports immediately. Switching active viewport cancels or explicitly
commits the current transient gesture; it must never transfer half a gesture to
another canvas.

## Central Control Protocol

### Compatibility rule

All existing singleton methods remain valid. In a multi-viewport workspace they
operate on the active viewport unless the method is intrinsically document-wide.
Examples:

- `viewer.camera.get` returns the active viewport camera;
- `viewer.camera.set` changes the active viewport and propagates configured
  links;
- `viewer.channels.set_presentation` changes the active viewport presentation;
- object style/filter methods change the active viewport presentation;
- mask polygon creation changes the shared document; and
- object analysis tasks operate on explicitly supplied data/query inputs, with
  active-viewport filter use opt-in rather than implicit.

Document this behavior and return `viewport_id` in affected snapshots.

### New resources and methods

Recommended canonical surface:

```text
viewer.workspace.get
viewer.workspace.layout.get
viewer.workspace.layout.set

viewer.viewports.list
viewer.viewports.get
viewer.viewports.create
viewer.viewports.clone
viewer.viewports.rename
viewer.viewports.remove
viewer.viewports.set_active

viewer.viewports.camera.get
viewer.viewports.camera.set
viewer.viewports.camera.fit
viewer.viewports.planes.get
viewer.viewports.planes.set
viewer.viewports.channels.get
viewer.viewports.channels.set
viewer.viewports.layers.get
viewer.viewports.layers.set
viewer.viewports.objects.style.get
viewer.viewports.objects.style.set
viewer.viewports.objects.filter.get
viewer.viewports.objects.filter.set

viewer.viewport_links.list
viewer.viewport_links.create
viewer.viewport_links.update
viewer.viewport_links.remove
```

Every viewport-specific method requires `viewport_id`. Do not overload every
legacy request with an optional selector as the only multi-view API; explicit
resource paths make discovery, permissions, documentation, and future
deprecation clearer.

### Example layout schema

```json
{
  "type": "split",
  "axis": "horizontal",
  "ratio": 0.5,
  "first": {"type": "viewport", "viewport_id": "viewport-left"},
  "second": {"type": "viewport", "viewport_id": "viewport-right"}
}
```

The first implementation validates a single split and exactly two leaves. The
recursive schema permits grid-like nested splits later without breaking the
protocol. Ratios must be finite and clamped to a usable range, for example
`0.1..=0.9`.

### Example viewport snapshot

```json
{
  "viewport_id": "viewport-left",
  "title": "Marker A",
  "active": true,
  "camera": {
    "center_world_lvl0": [1420.0, 880.0],
    "zoom_screen_per_lvl0_px": 0.75
  },
  "plane": {"mode": "xy", "z": 4},
  "presentation_revision": 27,
  "navigation_revision": 31,
  "canvas_available": true
}
```

Separate navigation and presentation revisions prevent an unrelated style
change from causing a camera conflict. Workspace layout and document data have
their own revisions.

### Example link-group schema

```json
{
  "link_group_id": "comparison-navigation",
  "viewport_ids": ["viewport-left", "viewport-right"],
  "fields": ["camera", "plane", "selection"]
}
```

### Events

Add structured events such as:

```text
viewer.viewports.created
viewer.viewports.removed
viewer.viewports.active_changed
viewer.viewports.navigation.changed
viewer.viewports.presentation.changed
viewer.workspace.layout.changed
viewer.viewport_links.changed
```

Use `source = "viewport:<id>"` and include `viewport_id`, revisions, link
transaction ID, and initiating session where applicable. A linked update should
identify both the initiating viewport and affected viewports.

Continue emitting existing active-view events such as
`viewer.camera.changed` when the active viewport's compatible legacy state
changes. This preserves subscriptions while new clients migrate to explicit
viewport events.

### Tasks and cancellation

Data loading, analysis, and exports are document tasks. A task may record the
originating viewport for provenance but must survive harmless active-viewport
changes. Presentation-only changes are immediate commands unless they initiate
property loading.

Removing a viewport cancels only tasks and screenshot requests exclusively
owned by that viewport. It must not cancel shared source/property work still
needed by the document or another viewport.

## Python SDK Design

Expose resource objects in both synchronous and asynchronous clients:

```python
workspace = app.viewer.workspace.get()

left = app.viewer.viewports.active()
right = app.viewer.viewports.clone(left.id, title="Marker B")

app.viewer.workspace.set_layout(
    split="horizontal",
    viewports=[left.id, right.id],
    ratio=0.5,
)

app.viewer.viewport_links.create(
    viewports=[left.id, right.id],
    fields=["camera", "plane", "selection"],
)

left.objects.set_style(
    fill_cells=True,
    color_property="marker_a",
    fill_opacity=0.65,
)
right.objects.set_style(
    fill_cells=True,
    color_property="marker_b",
    fill_opacity=0.65,
)
```

Convenience wrappers should sit on canonical protocol methods, not implement
linking or state mirroring in Python. A higher-level helper may make the common
case concise:

```python
comparison = app.viewer.viewports.compare(
    layout="horizontal",
    linked=["camera", "plane", "selection"],
    titles=["Marker A", "Marker B"],
)

comparison.left.objects.set_style(color_property="marker_a", fill_cells=True)
comparison.right.objects.set_style(color_property="marker_b", fill_cells=True)
```

The helper must return stable resource handles containing viewport IDs. Resource
handles should tolerate active-viewport changes and raise a specific not-found
error after their viewport is removed.

Async calls have the same semantics:

```python
right = await app.viewer.viewports.clone(left.id, title="Marker B")
await asyncio.gather(
    left.objects.set_style(color_property="marker_a", fill_cells=True),
    right.objects.set_style(color_property="marker_b", fill_cells=True),
)
```

Concurrent presentation commands use each viewport's presentation revision, so
they do not conflict merely because they target different canvases.

## MCP Policy

The MCP adapter should use the same canonical central methods, but it does not
need to expose every low-level multi-viewport operation immediately. A useful
curated first surface is:

- list/inspect viewports;
- create a two-view comparison;
- set the active viewport;
- set layout and navigation links; and
- configure each viewport's channel and object presentation.

Existing MCP viewer tools continue to target the active viewport. No separate
MCP-only multi-view implementation should be created.

## Screenshots, Saved Views, Projects, and Deep Links

### Screenshots

Extend screenshot capture with explicit scope:

```text
viewer.screenshot.capture { viewport_id, ... }
viewer.workspace.screenshot.capture { ... }
```

The first captures the named canvas without adjacent headers/panels. The second
captures the composed workspace canvases and may optionally include viewport
headers. Legacy capture targets the active viewport.

Each viewport needs independent pending/in-flight screenshot identity, even if
the save worker remains shared. Readback coordinates must use that viewport's
canvas rectangle.

### Saved views

Keep `ProjectViewSpec` as an individual view specification and make capture
operate on the active or explicitly named viewport. Add a separate, versioned
`ProjectWorkspaceSpec` only after the runtime model is stable:

```text
ProjectWorkspaceSpec
  version
  layout
  active_viewport_id
  viewport specs
  link groups
```

This avoids changing the meaning of existing saved views. A workspace spec may
refer to per-view specs rather than duplicating their schema.

### Project state migration

Existing project files with `ProjectRoiViewState` load as one viewport. New
workspace state must be optional and versioned. Saving a one-viewport project
should remain compatible where practical. Never discard multi-view state
silently when an older representation cannot contain it; report that only the
active view will be exported or require explicit confirmation in the UI.

### Deep links

Existing deep links apply to the active viewport and retain their current URL
form. A future workspace link should use a versioned compact workspace payload
or a saved workspace reference. Do not multiply every current query parameter
by left/right viewport prefixes.

## Implementation Sequence

Every phase should leave Odon shippable and preserve the single-viewport path.

### Phase 0: Characterization and invariants

- Add tests that characterize current camera, plane, channel presentation,
  object style/filter, selection, screenshot target, and project-view behavior.
- Add render-planning tests around active raw keys and render generations.
- Record source-open count, raw request count, cache memory, and representative
  frame timings for one viewport.
- Create a checked-in state-ownership inventory for fields extracted from
  `OmeZarrViewerApp` and `ObjectsLayer`.

Exit criterion: current behavior is protected before structural movement.

### Phase 1: Introduce one-viewport workspace boundaries

- Add `ViewportId`, `ViewportNavigation`, `ViewportPresentation`,
  `ViewportRenderState`, and `ViewerWorkspace` with exactly one viewport.
- Move camera, canvas rectangle, plane, presentation, fallback, and transient
  interaction state behind the viewport boundary in small steps.
- Render panels once against `active_viewport()`.
- Add a viewer facade for `RootApp` control dispatch.
- Preserve all existing behavior and serialized/API results.

Exit criterion: the application still displays exactly one canvas, but no
canvas-specific code relies on an implicit global viewport.

### Phase 2: Extract the shared document

- Move dataset, stores, loaders, source metadata, scientific layers, edits, and
  shared caches into `ViewerDocument`.
- Split channel metadata from channel presentation.
- Split mask/annotation data from viewport visibility/style.
- Define document commands for shared mutations.
- Keep one viewport and prove no behavior regression.

Exit criterion: creating another `ViewportState` would not duplicate a source,
loader, scientific layer, or edit history.

### Phase 3: Multi-consumer image rendering

- Make canvas rendering accept an explicit viewport.
- Generate render plans for all visible viewport leaves.
- Union and deduplicate raw active keys.
- Add fair per-viewport request scheduling.
- Make target-level/fallback state independent per viewport.
- Make `TilesGl` safe for consecutive different-canvas compositions.
- Replace the CPU loader's singleton latest-generation assumption.
- Add a developer-only/hardcoded two-canvas split first.

Exit criterion: two canvases can independently pan, zoom, choose planes, and
render different channel presentations from one shared source and cache.

### Phase 4: Object and overlay presentation separation

- Split `ObjectsLayer` according to the document/selection/analysis/
  presentation/render-cache model.
- Support simultaneous object property styles and filters.
- Share selection identities and display selection in both canvases.
- Separate presentation for label outlines, masks, annotations, spatial layers,
  and external layers.
- Ensure edits in the active viewport update shared data and redraw both views.

Exit criterion: the exact marker-A/marker-B comparison works through internal
Rust commands.

### Phase 5: Workspace UI and link controller

- Add horizontal, vertical, and single layouts.
- Add clone, close, swap, rename, and active-viewport controls.
- Add canvas chrome and active indication.
- Implement transactional camera/plane/selection link groups.
- Gate tools and keyboard behavior on the active viewport.
- Define behavior for activation during an in-progress gesture.

Exit criterion: the motivating workflow can be completed entirely in native UI
with clear active and linked state.

### Phase 6: Central API and Python SDK

- Add typed central commands, registry metadata, permissions, revisions, and
  events for workspace/viewports/links.
- Implement explicit viewport variants for camera, plane, channel, layer,
  object style, and object filter operations.
- Preserve singleton methods as active-viewport aliases.
- Add sync and async Python resources and typed snapshots.
- Add the `compare` convenience helper and an interactive example script.
- Regenerate the Python API reference from the registry.
- Add a curated MCP comparison surface if useful.

Exit criterion: the motivating workflow runs reliably from IPython/vim-slime
without direct GUI interaction.

### Phase 7: Output and persistence integration

- Add explicit viewport and whole-workspace screenshots.
- Allow `ProjectViewSpec` capture from an explicit viewport.
- Add versioned workspace persistence after its schema is stable.
- Define project migration and downgrade behavior.
- Add future workspace deep-link representation only if there is a concrete
  workflow requiring it.

Exit criterion: comparison workspaces can be reproduced and captured without
weakening old project/deep-link behavior.

### Phase 8: Hardening and expansion gate

- Run the complete Rust and Python suites plus GPU/manual smoke tests.
- Add memory, I/O deduplication, and navigation performance benchmarks.
- Test slow/failed remote sources and viewport removal during loading.
- Audit session cleanup, task cancellation, and extension-layer ownership.
- Update user, Python, MCP, and developer documentation.
- Decide from evidence whether to permit more than two viewports.

Exit criterion: first milestone is stable enough to enable by default.

## Test Strategy

### Pure Rust state tests

- Create, clone, rename, activate, reorder, and remove viewport state.
- Reject duplicate/missing IDs and invalid layout leaves.
- Prevent removal of the final viewport.
- Validate layout ratios and the first-milestone two-leaf limit.
- Verify cloned presentation is independent while document references are
  shared.
- Verify navigation links propagate once with no feedback loop.
- Verify unlinked fields remain unchanged.
- Verify revisions are scoped to workspace, navigation, presentation, and
  document data.

### Render-planning and cache tests

- Union active raw keys from overlapping viewports.
- Deduplicate identical keys and retain keys needed by either viewport.
- Fairly schedule two disjoint visible regions.
- Cancel stale work for only the affected viewport.
- Maintain independent coarse/fine fallback histories.
- Accept two simultaneous CPU presentation generations.
- Reuse raw resources when only color/contrast differs.
- Keep object property render buffers for two styles simultaneously.

### Interaction tests

- A click activates a viewport; hover does not.
- Panels mutate only the active viewport's presentation.
- Linked camera movement updates both; unlinking stops propagation.
- A selection in either viewport highlights identical object IDs in both.
- Only the active viewport begins an edit gesture.
- Switching/removing a viewport resolves transient gestures safely.

### Control protocol tests

- Registry schemas and mode availability for every new method.
- Stable viewport IDs in snapshots and events.
- Explicit viewport targeting independent of active viewport.
- Legacy `viewer.*` targeting of the active viewport.
- Conflict behavior for each revision domain.
- One causal event batch for linked changes.
- Session cleanup does not remove user/project viewports or shared data.
- Removing a viewport produces clear not-found responses for stale handles.

### Python tests

- Sync and async parity for all viewport/workspace/link resources.
- Resource handles keep targeting their ID after active viewport changes.
- `compare` emits canonical calls and returns correct handles.
- Concurrent async changes to different viewports do not conflict.
- Events deserialize with viewport and link transaction metadata.
- Existing Python tests pass unchanged.

### Persistence and screenshot tests

- Legacy project/deep-link state loads into one viewport.
- Individual saved-view capture is identical for the active viewport.
- Workspace state round-trips once introduced.
- Per-viewport screenshot captures the correct canvas bounds and style.
- Workspace screenshot contains both canvases in layout order.

### End-to-end acceptance test

Using the synthetic or another checked-in fixture:

1. open one OME-Zarr source;
2. load one segmentation with at least two scalar properties;
3. clone the active viewport and split horizontally;
4. link camera, plane, and selection;
5. fill the left viewport by property A;
6. fill the right viewport by property B;
7. pan, zoom, and change plane in either canvas;
8. select a cell and observe it in both;
9. capture each viewport and the workspace; and
10. verify one source instance and deduplicated raw requests.

The same test should be expressible through native Rust commands and the Python
SDK. Visual GPU checks can supplement, but not replace, semantic assertions.

## Risks and Mitigations

### Large monolithic extraction

Risk: moving many fields at once creates pervasive borrow and behavior bugs.

Mitigation: introduce a one-viewport workspace first, extract state category by
category, and keep characterization tests green at every step.

### Root control-dispatch duplication

Risk: adding `Mode::Multi` multiplies approximately 390 existing mode matches.

Mitigation: add a viewer facade and turn the existing single mode into a
workspace rather than adding a parallel implementation.

### Cache thrash and unfair loading

Risk: two distant views double working-set pressure or let one active canvas
starve the other.

Mitigation: union keys, instrument ownership/priority, allocate fair budgets,
and grow caches from measured combined visible demand within memory limits.

### GPU state leakage between callbacks

Risk: a shared offscreen compositor assumes one full-window canvas.

Mitigation: test sequential canvas callbacks, scope clears/scissors and output
rects explicitly, and keep presentation inputs immutable per draw.

### Presentation accidentally mutates scientific data

Risk: filters, color properties, and visibility are currently embedded beside
object data and analysis state.

Mitigation: make presentation a separately clonable value and require document
commands for scientific mutations.

### Ambiguous legacy API behavior

Risk: scripts unexpectedly control whichever view was most recently clicked.

Mitigation: preserve active-view semantics but return `viewport_id`, document
the rule, and encourage explicit handles in new multi-view scripts.

### Persistence schema churn

Risk: saving the workspace too early freezes an awkward internal model.

Mitigation: stabilize runtime and API schemas first; keep old individual view
specs; introduce a separate versioned workspace spec later.

## Milestone Definition of Done

The first multi-viewport milestone is complete when all of the following are
true:

- Two same-document viewports render side by side or vertically.
- They share dataset, raw tile, object data, and edit ownership.
- They support independent channel and object presentation.
- Camera and plane linking can be enabled and disabled.
- Shared cell selection behaves consistently.
- Active-viewport panels and edit gating are unambiguous.
- The CPU and GPU render paths both support two current views.
- Existing single-view native behavior, control methods, MCP behavior, and
  Python scripts remain compatible.
- The central API and sync/async Python SDK can reproduce the full motivating
  comparison.
- Per-viewport and workspace screenshots target the correct rectangles.
- Automated tests cover state, scheduling, control contracts, Python wrappers,
  compatibility, and persistence migration.
- Measurements demonstrate that source I/O, raw tiles, and object geometry are
  not duplicated per viewport.

## Recommended First Implementation Slice

Begin with Phases 0 and 1 only. Specifically:

1. add characterization tests for camera, plane, presentation, and object style;
2. introduce stable `ViewportId` and a one-entry `ViewportCollection`;
3. move `camera` and `last_canvas_rect` into `ViewportNavigation`/
   `ViewportState`;
4. make canvas and control camera methods resolve the active viewport; and
5. keep the UI and public API visibly unchanged.

This slice establishes the seam on which every later feature depends while
remaining small enough to review and bisect. The first actual split canvas
should not be added until both the shared document boundary and multi-consumer
render scheduling exist.
