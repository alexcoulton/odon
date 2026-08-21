# Grid and Layout-Tree Milestone Plan

Status: proposed next milestone

This milestone replaces Odon's fixed one/two-viewport split with a validated
layout tree and provides grid conveniences for common multi-channel comparison
workflows. It builds on the shared-document and per-viewport presentation
boundaries established by the first multi-viewport milestone.

The initial supported product limit should be nine simultaneous viewports. This
supports useful layouts through 3 by 3 while keeping rendering, interaction,
and memory behaviour testable. Odon must report the limit through its
capability and workspace snapshots. The limit may be raised later only after
performance evidence supports doing so; the API must not imply that viewport
count is unbounded.

## Goals

- Support between one and nine viewports in one single-image workspace.
- Allow arbitrary nested horizontal and vertical splits.
- Provide concise APIs for regular grids without introducing a separate grid
  rendering architecture.
- Preserve stable viewport identities and independent presentation state.
- Keep image data, decoded tiles, GPU resources, object geometry, annotations,
  edits, and selection identity document-shared.
- Maintain fair rendering and bounded caches as viewport count increases.
- Make every layout operation available through the central Rust command API,
  the Python SDK, and compatible MCP surfaces.
- Preserve all existing one- and two-viewport Python and protocol calls.
- Persist and restore arbitrary layouts through a versioned project schema.

## Non-goals

- Unbounded viewport counts.
- Different unrelated datasets in different viewports.
- Detached operating-system windows.
- Arbitrary Python-defined widgets inside the native canvas layout.
- Independent scientific documents, annotation histories, or object geometry
  per viewport.
- Independent selection identities in the core layout milestone.
- Multi-viewport Mosaic mode.

## 1. Canonical layout model

Use a binary split tree as the canonical Rust representation:

```rust
pub enum WorkspaceLayout {
    Viewport {
        viewport_id: ViewportId,
    },
    Split {
        node_id: LayoutNodeId,
        axis: SplitAxis,
        ratio: f32,
        first: Box<WorkspaceLayout>,
        second: Box<WorkspaceLayout>,
    },
}
```

`SplitAxis::Horizontal` means side-by-side children and
`SplitAxis::Vertical` means top-and-bottom children, matching the current API
terminology.

A 2 by 2 workspace is represented as:

```text
Vertical split
├── Horizontal split
│   ├── Viewport 1
│   └── Viewport 2
└── Horizontal split
    ├── Viewport 3
    └── Viewport 4
```

A regular grid is convenience syntax that generates a balanced tree. Rendering
therefore has one canonical layout representation instead of separate split
and grid implementations.

Every split node has a stable opaque `LayoutNodeId`. Native UI and external
clients can use the ID to resize or restructure a particular branch without
depending on path indices that change after edits. A full declarative layout
request may omit new node IDs and let Rust assign them; layout snapshots always
return assigned IDs.

The viewport collection remains separate from the layout tree. Viewport slots
own identity and state; tree leaves only determine placement.

## 2. Layout and workspace invariants

The Rust workspace must validate these invariants atomically:

- A workspace contains between one and the configured maximum number of
  viewports.
- Every layout leaf references an existing viewport.
- Every live viewport occurs exactly once in the tree.
- Viewport IDs and split-node IDs are unique.
- Split ratios are finite and remain within a usable range such as
  `0.1..=0.9`.
- Tree depth and total node count are bounded independently of the viewport
  limit to protect all protocol entry points from pathological payloads.
- A one-viewport workspace has one leaf and no redundant split nodes.
- Removing a viewport collapses its redundant parent by promoting the sibling.
- The final viewport cannot be removed.
- Removing the active viewport selects a deterministic replacement, preferably
  the next leaf in visual traversal order and then the preceding leaf.
- Layout mutations either succeed completely or leave the workspace unchanged.
- Layout mutations advance the workspace revision once and respect
  `if_revision` conflict guards.
- Presentation and navigation revisions do not change merely because a leaf is
  moved.

The validator should produce structured errors with the invalid node ID or
viewport ID where possible. Validation must occur in Rust even when Python has
already performed local convenience checks.

## 3. Generalized viewport lifecycle

Replace the current "clone into the second slot" assumption with operations
that insert viewports relative to a target leaf:

```python
new_view = app.viewer.viewports.clone(
    source,
    split=source,
    axis="horizontal",
    ratio=0.5,
)
```

This clones `source` and atomically replaces its leaf with a split containing
the original and new viewport. The caller may select whether the new viewport
appears first or second.

Support these lifecycle and layout operations:

```python
viewport.remove()
viewport.move(beside=other, position="right")
workspace.swap(first, second)
workspace.equalize(node)
workspace.set_split_ratio(node, 0.65)
workspace.maximize(viewport)
workspace.restore_layout()
```

Removing a viewport must also:

- cancel only that viewport's pending screenshot and render work;
- release its presentation-generation and transient interaction state;
- retain shared document data and work still needed by other viewports;
- collapse the affected branch without changing sibling state; and
- make old Python handles fail with a clear resource-not-found error.

Maximization is transient workspace state. It displays one viewport using the
full canvas without destroying or rewriting the underlying layout tree.
Restoring returns to the exact prior tree and ratios.

## 4. Grid convenience API

Python should make regular channel-comparison grids concise:

```python
grid = app.viewer.viewports.create_grid(
    source=app.viewer.viewports.active(),
    rows=2,
    columns=3,
    titles=["DAPI", "CD3", "CD8", "PanCK", "FOXP3", "Ki67"],
    linked=("camera", "plane"),
)

for viewport, channel in zip(grid.viewports, grid.titles):
    viewport.set_visible_channels([channel])
```

Existing viewports can be arranged declaratively:

```python
app.viewer.workspace.set_grid(
    [
        [dapi, cd3, cd8],
        [panck, foxp3, ki67],
    ]
)
```

`create_grid()` should clone states, construct the tree, create the navigation
link group, and publish events as one central transaction. This avoids visible
intermediate layouts and a chain of conflicting workspace revisions.

`set_grid()` generates a balanced tree and submits one atomic layout change.
The grid builder should support:

- explicit rows of viewport handles or IDs;
- a flat sequence with `rows` and/or `columns`;
- row-major ordering;
- optional row and column weights;
- equal weights by default; and
- validation of rectangularity and complete viewport membership.

The first implementation need not preserve aligned draggable column boundaries
between independently nested rows. Grid generation must, however, assign
identical initial ratios to matching rows and columns. If independently
draggable rows prove confusing in usability tests, linked splitter constraints
can be designed as a later extension without changing viewport identity.

## 5. Declarative central API

The central API accepts and returns the complete canonical tree:

```json
{
  "type": "split",
  "node_id": "layout-1",
  "axis": "vertical",
  "ratio": 0.5,
  "first": {
    "type": "split",
    "node_id": "layout-2",
    "axis": "horizontal",
    "ratio": 0.5,
    "first": {"type": "viewport", "viewport_id": "viewport-1"},
    "second": {"type": "viewport", "viewport_id": "viewport-2"}
  },
  "second": {
    "type": "viewport",
    "viewport_id": "viewport-3"
  }
}
```

Python exposes the same form:

```python
app.viewer.workspace.set_layout(
    {
        "type": "split",
        "axis": "vertical",
        "ratio": 0.5,
        "first": {
            "type": "split",
            "axis": "horizontal",
            "ratio": 0.5,
            "first": {"type": "viewport", "viewport_id": dapi.id},
            "second": {"type": "viewport", "viewport_id": cd3.id},
        },
        "second": {
            "type": "viewport",
            "viewport_id": segmentation.id,
        },
    }
)
```

Add or generalize these central methods:

```text
viewer.workspace.layout.get
viewer.workspace.layout.set
viewer.workspace.layout.validate
viewer.workspace.layout.split
viewer.workspace.layout.move
viewer.workspace.layout.swap
viewer.workspace.layout.ratio.set
viewer.workspace.layout.equalize
viewer.workspace.layout.grid.set
viewer.workspace.layout.maximize
viewer.workspace.layout.restore
viewer.viewports.grid.create
```

`layout.validate` is read-only and returns the normalized tree, assigned-node
preview information, computed leaf order, and validation errors without
changing application state. Mutating methods return the resulting workspace
revision and affected viewport/node IDs.

Every method must be registered in the central command registry with JSON
schemas, availability, permission classification, task classification, events,
and manifest parity. Python and MCP must call these commands rather than
implementing their own Rust-bypassing state transitions.

## 6. Python resource design

Add stable `LayoutNode` or `SplitNode` handles alongside existing stable
`Viewport` handles:

```python
layout = app.viewer.workspace.get_layout()
split = layout.node("layout-2")
split.set_ratio(0.7)
split.equalize()
```

Recommended synchronous resources:

```text
app.viewer.workspace
app.viewer.workspace.layout
app.viewer.viewports
app.viewer.viewport_links
```

The async client exposes identical semantics with awaited network operations.
Local Python validation should catch simple type, shape, and unknown-axis
errors, while Rust remains authoritative for identity, revision, and workspace
invariants.

High-level helpers should return stable handles and the resulting layout:

```python
grid = app.viewer.viewports.create_grid(...)
grid.viewports
grid.layout
grid.rows
grid.columns
```

## 7. Backward compatibility

Existing calls remain valid:

```python
comparison = app.viewer.viewports.compare(...)
app.viewer.workspace.set_layout("horizontal", ratio=0.5)
app.viewer.workspace.swap()
```

Compatibility rules:

- `"single"`, `"horizontal"`, and `"vertical"` remain accepted shorthand
  for one- and two-leaf trees.
- `compare()` continues returning stable `left` and `right` handles.
- The existing no-argument `swap()` retains its two-viewport meaning and
  rejects ambiguous larger layouts with an actionable error.
- Legacy `app.viewer`, `app.channels`, and `app.objects` calls continue
  targeting the active viewport.
- Existing explicit viewport methods do not change signatures.
- Existing viewport IDs survive layout migration and rearrangement.
- The v1 fixed comparison link-group calls remain compatible.
- Existing single/two-view project state migrates to an equivalent layout
  tree.

The workspace response should expose both summary information and the canonical
tree:

```json
{
  "viewport_count": 6,
  "max_viewports": 9,
  "layout_kind": "tree",
  "layout": {"type": "split"},
  "active_viewport_id": "viewport-3"
}
```

Advertise explicit capabilities such as `layout_tree_v1`, `viewport_grid_v1`,
and `max_viewports: 9` so clients can adapt to older Odon processes.

## 8. Layout evaluation and rendering

Do not recursively construct ad hoc child `egui` layouts around rendering
calls. First evaluate the complete tree into geometry using a pure function:

```text
available rectangle
       ↓
layout-tree evaluation
       ↓
viewport rectangles + splitter rectangles
       ↓
render every visible viewport exactly once
```

The evaluator should accept:

- the canonical tree;
- the available rectangle;
- splitter thickness;
- header height;
- minimum canvas width and height; and
- optional maximized viewport state.

It should return deterministic viewport rectangles, splitter rectangles,
visual traversal order, and any minimum-size degradation diagnostics. Keeping
this pure makes the geometry independently testable and avoids the inherited
child-layout bug discovered during the two-viewport GPU smoke test.

Rendering requirements:

- Raw tile keys remain unioned and deduplicated across all visible viewports.
- Decoded source tiles and GPU textures remain document-shared.
- Every viewport retains independent presentation, render generation,
  target/fallback continuity, and canvas rectangle.
- Hidden viewports during maximization submit no render requests.
- Every visible viewport receives a non-zero minimum request budget.
- The active viewport receives a bounded scheduling preference.
- Remaining capacity is weighted by visible canvas area and current motion or
  refinement need.
- Scheduling uses round-robin or deficit accounting so a small pane cannot be
  permanently starved by a large active pane.
- Removing one viewport cancels only generations and requests no longer needed
  by another viewport.
- Object-presentation and composed-tile caches become bounded LRUs because up
  to nine independent styles and composites may coexist.
- Live diagnostics report visible, hidden, active, scheduled, and starved
  viewport counts along with shared-cache reuse.

The current one/two-view budget helper must become an N-consumer scheduler with
deterministic tests. Cache keys must continue excluding viewport identity when
the source samples are otherwise identical.

## 9. Native interaction and workspace UI

Native UI work includes:

- draggable splitters;
- clear active-viewport headers;
- close, clone, split, and maximize controls in each header;
- a workspace menu with 1 by 2, 2 by 1, 2 by 2, 2 by 3, 3 by 2, and 3 by 3
  presets;
- equalize-branch and equalize-all commands;
- minimum pane sizes and clear diagnostics when the window is too small;
- keyboard focus traversal in visual leaf order; and
- accessible labels containing viewport title and position.

Splitter hit regions must take precedence over canvas gestures so dragging a
splitter cannot pan or edit the image underneath. Clicking or beginning a
canvas gesture activates that viewport; hover alone does not. Only the active
viewport may begin scientific edits, and activation/removal/maximization must
resolve transient gestures safely.

Programmatic move and swap support is required in this milestone. Drag-and-drop
pane rearrangement is desirable but may follow after the command semantics are
stable.

## 10. Navigation and link groups

The core channel-grid workflow needs one navigation group containing every
viewport:

```python
app.viewer.viewport_links.update(
    viewports=grid.viewports,
    fields=("camera", "plane", "selection"),
)
```

Generalize the current comparison link group so membership is no longer
restricted to two IDs. One camera or plane mutation must propagate through one
causal transaction, update each affected navigation revision once, and publish
one structured event batch without feedback loops.

Multiple independent named navigation groups, such as two synchronized pairs
inside a 2 by 2 workspace, are useful but not required to unlock the first grid
workflow. They may be included late in this milestone if the generalized
controller remains simple; otherwise they become the next focused milestone.

Selection identity remains document-shared in this milestone. The API must not
suggest that removing `selection` from a link group creates an independent
selection set.

## 11. Persistence and migration

Introduce project workspace persistence version 2:

```json
{
  "version": 2,
  "active_viewport_id": "viewport-3",
  "layout": {"type": "split"},
  "viewports": [],
  "link_groups": []
}
```

Requirements:

- Migrate version 1 single, horizontal, and vertical layouts losslessly.
- Round-trip every valid tree, viewport state, ratio, title, stable viewport
  ID, and stable layout-node ID.
- Reject corrupt duplicate, missing, oversized, or overly deep trees without
  partially applying them.
- Define clear behaviour for a project containing more viewports than a build's
  advertised maximum.
- Do not serialize transient maximization as the base layout unless the user
  explicitly captures that presentation.
- Preserve existing saved-view behaviour for explicit viewport capture.
- Document downgrade behaviour for older Odon versions that do not understand
  workspace version 2.

## 12. Screenshots and output

- Per-viewport screenshots continue targeting stable viewport IDs.
- Workspace screenshots cover the full evaluated tree rectangle.
- Capture traversal follows deterministic visual leaf order: top-to-bottom,
  then left-to-right according to tree geometry.
- Splitters and viewport headers follow existing canvas-versus-window capture
  semantics.
- A maximized workspace screenshot captures the visible maximized state;
  callers may explicitly request the underlying full layout if needed.
- Removing a viewport clears only its queued captures.
- Project-view capture requires an explicit viewport when several presentation
  filters exist.

## 13. Events and revisions

Continue using separate workspace, navigation, presentation, and document
revision domains. Add sufficient event data for remote clients to update a
layout model without refetching unrelated application state.

Events should include:

```text
viewer.viewports.created
viewer.viewports.removed
viewer.viewports.active_changed
viewer.workspace.layout.changed
viewer.workspace.maximized_changed
viewer.viewport_links.changed
```

Layout events include the workspace revision, initiating session, operation,
affected node and viewport IDs, and either the normalized tree or an explicit
instruction that clients must refetch it. Atomic grid creation should emit one
causally grouped transaction rather than one user-visible layout change per
new viewport.

## 14. Testing strategy

### Pure layout tests

- Validate single leaves and arbitrary valid trees through nine leaves.
- Reject duplicate/missing viewport IDs and duplicate node IDs.
- Reject invalid ratios, excessive depth, excessive nodes, and excessive
  viewport counts.
- Generate every regular grid shape through 3 by 3.
- Verify visual traversal order.
- Remove every possible leaf and verify parent collapse and active fallback.
- Move, swap, equalize, maximize, and restore without changing viewport state.
- Verify mutations are atomic under errors and revision conflicts.
- Round-trip canonical JSON and project persistence representations.

### Geometry tests

- Child rectangles remain inside their parent.
- Siblings never overlap except for explicitly modelled splitter hit padding.
- Children plus splitter exactly cover their parent along the split axis.
- Nested ratios produce expected deterministic rectangles.
- Headers remain above their corresponding canvases.
- Minimum-size degradation is deterministic and reported.
- Maximization returns one full-size visible canvas without mutating the tree.

Property-based tests are particularly valuable for arbitrary valid trees,
repeated removal, normalization, and geometry partitioning.

### Render-planning and cache tests

Run semantic cases with one, two, four, six, and nine visible viewports:

- identical raw tile requests are read and decoded once;
- different channel composites remain independent;
- active preference does not starve any visible peer;
- a small viewport receives bounded service;
- removing one viewport cancels only its unique work;
- hidden maximized peers stop requesting work;
- independent fallback histories remain correct;
- CPU workers accept all live render generations;
- object geometry remains single-owned while nine styles coexist; and
- decoded, composed, GPU, and object-presentation caches remain bounded.

### Interaction tests

- Splitter dragging does not activate canvas pan/edit gestures.
- Canvas clicks activate the correct leaf.
- Side panels always target the active viewport.
- Keyboard traversal follows visual order.
- Only the active viewport may edit shared scientific data.
- Removing, moving, or maximizing safely resolves transient gestures.
- Linked navigation propagates once to all group members.

### Control and Python tests

- Registry schemas, permissions, availability, events, and manifest parity for
  every new method.
- Declarative tree validation at every protocol entry point.
- Stable viewport and node handles after unrelated rearrangements.
- Clear stale-handle errors after removal or branch collapse.
- Sync and async parity for layout, grid, split, move, resize, maximize, and
  restore calls.
- Existing `compare()`, two-view layout, and legacy active-view tests pass
  unchanged.
- Concurrent mutations use workspace revisions predictably.
- MCP calls reach the same central command implementation.

### Persistence and screenshot tests

- Version 1 migration produces an equivalent tree.
- Version 2 round-trips all supported layout shapes.
- Corrupt layouts do not partially restore.
- Individual screenshots target the correct viewport.
- Workspace screenshots include every visible leaf in geometric order.
- Maximized capture and underlying-layout capture have explicit semantics.

## 15. Performance and release gates

Benchmark one, two, four, six, and nine viewports using overlapping and disjoint
camera regions, CPU and GPU paths, local and slow sources, and identical versus
different channel presentations.

Release gates:

- Layout evaluation and frame planning remain a small, measured fraction of
  the frame budget at nine viewports.
- Every visible viewport receives work within a deterministic bounded number
  of scheduling rounds.
- Shared raw reads and decodes scale with unique source tile keys, not viewport
  count.
- GPU raw-cache ownership remains singular.
- Object geometry ownership remains singular.
- Per-viewport state and bounded presentation caches account for incremental
  memory growth.
- Removing a viewport promptly drops only its unique work and generations.
- The complete Rust and Python suites remain green.
- Formatting, generated-documentation, manifest-parity, lint, and compile
  checks pass.
- Live GPU smoke verification demonstrates a 3 by 3 channel grid with one
  shared source and distinct composites.

If nine-view performance fails these gates, keep the architecture and API but
ship a lower advertised maximum supported by evidence. The maximum is a runtime
capability, not a protocol constant clients should assume.

## 16. End-to-end acceptance workflow

Using a checked-in synthetic multi-channel OME-Zarr fixture:

1. Open one source and load one segmentation.
2. Create a 3 by 3 grid from the active viewport in one API transaction.
3. Assign a different channel or channel composite to every viewport.
4. Assign at least two different object colour properties across the grid.
5. Link camera and plane navigation across all nine viewports.
6. Pan, zoom, and change plane from multiple source viewports.
7. Verify all linked canvases move once and retain independent presentation.
8. Select one cell and verify shared selection identity in every canvas.
9. Resize nested splitters, equalize a branch, move two leaves, and maximize
   one viewport.
10. Restore the complete layout and verify stable IDs and state.
11. Capture individual, maximized, and full-workspace screenshots.
12. Save, close, and restore the project.
13. Verify one document/source, one decoded cache, one GPU raw cache, and one
   object-geometry owner throughout.

The workflow must be expressible through native controls and the Python SDK.
GPU captures supplement but do not replace semantic assertions and resource
ownership counters.

## 17. Implementation phases

### Phase 0: Characterization and compatibility lock

- Capture current one/two-view layout, interaction, event, persistence,
  screenshot, and performance behaviour in tests.
- Record current Python and MCP calls that must remain compatible.
- Establish 1/2/4/9-view benchmark fixtures and counters.

Exit criterion: current behaviour and the expansion performance baseline are
reproducible.

### Phase 1: Pure layout-tree foundation

- Add `LayoutNodeId`, `SplitAxis`, and recursive `WorkspaceLayout`.
- Implement validation, traversal, normalization, grid generation, insertion,
  removal collapse, move, swap, and equality operations.
- Add the pure rectangle evaluator and comprehensive state/geometry tests.
- Keep the running application on the current layout path initially.

Exit criterion: arbitrary valid trees can be transformed and evaluated without
application or rendering state.

### Phase 2: Generalize workspace lifecycle

- Replace `MAX_VIEWPORTS = 2` assumptions with an advertised tested cap.
- Replace fixed `ViewportLayout` and `split_ratio` workspace fields with the
  canonical tree.
- Generalize create, clone, activate, rename, move, and remove operations.
- Preserve stable IDs, revisions, shared ownership, and stale-handle behaviour.

Exit criterion: application state safely owns and mutates one through nine
viewports before rendering them simultaneously.

### Phase 3: Recursive canvas allocation and interaction

- Integrate the pure rectangle evaluator with `egui`.
- Render headers, canvases, and splitter hit regions from evaluated geometry.
- Add active-view interaction, splitter dragging, minimum sizes, maximize, and
  restore.
- Retain the one-active-editor invariant.

Exit criterion: nested layouts render and interact correctly without ad hoc
child-layout geometry.

### Phase 4: N-consumer rendering and cache bounds

- Replace two-consumer budgets with fair N-consumer scheduling.
- Generalize active key unions and live CPU render-generation sets.
- Bound composed-tile and object-presentation caches.
- Add removal, hidden-pane, slow-source, and starvation tests.

Exit criterion: one, two, four, and nine viewports render independently while
sharing source work and keeping memory bounded.

### Phase 5: Native workspace UI

- Add split/clone/remove/maximize header controls.
- Add grid presets, ratio controls, equalization, and layout diagnostics.
- Add keyboard focus traversal and accessibility labels.
- Add programmatic move/swap UI; defer drag-and-drop rearrangement if needed.

Exit criterion: users can construct and manage useful layouts without Python.

### Phase 6: Central API and Python SDK

- Register every tree, grid, lifecycle, and maximize command centrally.
- Add normalized schemas, events, capabilities, revision guards, and manifest
  entries.
- Add matching sync and async Python resources and stable layout-node handles.
- Preserve `compare()` and current two-view conveniences.
- Update MCP mappings to use the same commands.

Exit criterion: the complete workflow can be controlled without native UI and
all surfaces share one implementation.

### Phase 7: Link generalization

- Permit the canonical navigation group to contain all workspace viewports.
- Verify one causal propagation transaction across N members.
- Decide from implementation evidence whether multiple named groups belong in
  this milestone or the following one.

Exit criterion: a channel grid can navigate synchronously without feedback
loops or revision ambiguity.

### Phase 8: Persistence, screenshots, and migration

- Add workspace persistence version 2 and version 1 migration.
- Generalize workspace and per-viewport capture geometry.
- Define maximized capture and downgrade behaviour.
- Add full round-trip and output tests.

Exit criterion: complex workspaces can be saved, restored, and captured without
loss of identity or presentation.

### Phase 9: Hardening and release

- Run complete Rust/Python suites and all static/documentation gates.
- Execute CPU, GPU, local, remote, overlapping, and disjoint benchmarks.
- Perform the 3 by 3 live acceptance workflow.
- Audit cleanup, task cancellation, memory bounds, extension-layer ownership,
  and session behaviour.
- Update user, Python, MCP, developer, limitation, and application-surface
  documentation.
- Select the shipped maximum from measured evidence.

Exit criterion: the advertised viewport count is demonstrably fair, bounded,
compatible, persistent, and usable.

## Recommended starting point

Begin with Phases 0 and 1. The first implementation change should be the pure
layout-tree model and rectangle evaluator, introduced behind the existing
one/two-view behaviour. This gives lifecycle, rendering, UI, protocol, and
persistence work a stable geometry foundation while minimizing regression risk
to the feature that has just shipped.
