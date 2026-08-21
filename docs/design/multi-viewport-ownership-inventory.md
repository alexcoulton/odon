# Multi-Viewport State Ownership Inventory

Status: implemented ownership contract for the two-viewport milestone

This inventory is the review checklist for state moved behind Odon's viewport
workspace boundary. It describes logical ownership; the shared document is
still physically hosted by `OmeZarrViewerApp`, while `ViewerViewportState` is
the capture/apply boundary used for each canvas. A field listed as shared must
never be copied merely because a viewport is cloned.

## Shared document state

| State | Current owner/evidence | Invariant |
|---|---|---|
| `dataset`, `store`, remote runtime and transforms | `OmeZarrViewerApp` | One open source and coordinate system per workspace |
| CPU/TIFF tile loaders and decoded cache | `TileLoaderHandle` / `CpuDecodedTileCache` | One loader and presentation-independent decode cache; colour/contrast are not cache keys |
| Raw GPU loader and textures | `RawTileLoaderHandle` / `TilesGl` | Active keys are the union of all canvases |
| Label loader, label geometry and label textures | `OmeZarrViewerApp` | Source labels load once; outline presentation is viewport-local |
| Channel identity, name, note and immutable metadata | `OmeZarrViewerApp.channels` | `ViewerViewportState::apply` copies only visibility, colour and window |
| Segmentation/object geometry, properties, indexes and base LODs | `ObjectsLayer` | Viewport cloning does not reload or duplicate object data |
| Object selection identities and focus | `ObjectsLayer` | Selection is document-shared and visible in both canvases |
| Mask polygons and raster source data | `mask_layers` | Polygon edits, undo and dirty state are shared; styles are viewport-local |
| Annotation records and loaded parquet data | `annotation_layers` | Records/resources are shared; visibility/category styles are viewport-local |
| SpatialData/Xenium source data | `spatial_layers`, `spatial_image_layers`, `xenium_layers` | Loaded data and provenance are shared; presentations are viewport-local |
| Scientific edits, undo/redo and project dirty state | `OmeZarrViewerApp` | Exactly one ordered document history |
| Analysis, property loading, measurements and exports | object layers / control task system | Tasks survive active-viewport changes and use explicit filter provenance |
| Project/ROI association | `ProjectSpace` | All viewports belong to the same open document |

## Per-viewport state

The complete executable list is `ViewerViewportState` in `src/app.rs`.

| State group | Representative fields | Notes |
|---|---|---|
| Navigation | `camera`, canvas rectangle, plane mode, x/y/z slices | Has its own navigation revision and optional camera/plane linking |
| Render continuity | active/previous render IDs, target/fallback levels, visible-world history | Prevents one canvas invalidating the other's progressive rendering |
| Channels | selected channel, visible/colour/window values, order, group presentation | Channel identity remains shared |
| Layer inspection | active layer, selected layer sets, channel/overlay order | Panels target the active viewport |
| Primary objects | display style, filter and filter cache, visibility, opacity, outline/fill/legend state | Geometry, properties and selection are not copied |
| Labels and points | label-outline style, cell-point visibility/style | Presentation only |
| Masks and annotations | ID-keyed visibility/style records | Mutable polygons/annotation rows remain shared |
| SpatialData and Xenium | ID-keyed shape/image/object/point/cell/transcript presentations | Loaded source resources remain shared |
| Decorations and sampling | smooth pixels, scale bar, HUD and tile-debug visibility | Selected at each viewport draw callback |
| Screenshot targeting | stable viewport ID and canvas rectangle | Pending requests are cancelled only with their target viewport |
| Revisions | navigation and presentation revisions | Workspace/document revisions remain separate |

## Workspace and shell state

| State | Current owner/evidence | Invariant |
|---|---|---|
| Ordered viewport slots, IDs and titles | `ViewportWorkspace<ViewerViewportState>` | One or two stable-ID slots |
| Layout and active viewport | `ViewportWorkspace` | `single`, `horizontal`, or `vertical`; a split has exactly two leaves |
| Camera/plane/selection link policy | `ViewportLinks` | Navigation propagation is one transaction; selection is shared in this milestone |
| Side panels, selected tabs and dialogs | `OmeZarrViewerApp` shell fields | Rendered once and bound to the active viewport |
| Tool mode and transient edit gesture | `OmeZarrViewerApp` shell fields | Only the active canvas can start destructive edits; activation/removal cancels transient state |
| Screenshot save worker and common output settings | viewer/root screenshot queue | Target identity remains viewport-specific |

## Prohibited ownership regressions

- Do not construct a second `OmeZarrViewerApp` for a second viewport.
- Do not place object selection, mask polygons, annotation records, undo state,
  task handles, stores, loaders or source caches in `ViewerViewportState`.
- Do not key raw/decoded tile resources by colour, contrast, opacity or viewport
  ID when the underlying source samples are identical.
- Do not let a filter-sensitive analysis/export infer a filter in a multi-view
  workspace; require `viewport_id`, `filter_query`, `use_all_objects`, or an
  explicit active-view opt-in.
- Do not allow a non-active viewport to begin destructive editing gestures.

## Verification anchors

- `src/viewports.rs`: identity, layout, cloning, limits, links and revisions.
- `ViewerViewportState::capture/apply`: the per-view boundary.
- `control_viewport_workspace_snapshot`: runtime sharing counters.
- `tile_worker_accepts_two_live_viewport_generations` and
  `composited_tiff_loader_shares_decode_across_viewport_presentations`: shared
  CPU decode evidence.
- Multi-viewport characterization tests in `src/app.rs`: presentation,
  persistence, linking, interaction and filter-provenance behavior.
