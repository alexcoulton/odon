# Odon Feature Inventory

This document groups Odon's current capabilities into presentation-friendly categories.

## Core Viewer

- Native desktop application for multiplexed spatial proteomics and spatial transcriptomics review.
- Rust implementation focused on responsive interaction with large image pyramids.
- Primary support for OME-Zarr / OME-NGFF imagery.
- Single-ROI viewer for detailed inspection.
- Multi-ROI mosaic viewer for cohort, TMA, and project-scale review.
- GPU-backed image compositing.
- Tile-based, viewport-driven loading.
- Coarse-to-fine rendering, with overview levels shown before fine tiles arrive.
- Local OME-Zarr loading.
- Remote-store capable viewing where configured.
- Drag-and-drop opening of supported local datasets.
- Project panel for loading projects, importing samplesheets, and discovering OME-Zarr roots.

## Image And Channel Viewing

- Multi-channel image display.
- Additive channel compositing.
- Per-channel visibility toggles.
- Active channel selection.
- Previous/next channel stepping.
- Per-channel colour controls.
- Per-channel contrast minimum and maximum controls.
- Histogram-guided contrast adjustment.
- Compact top-bar contrast controls.
- Right-panel detailed contrast controls.
- Channel list filtering.
- Channel sorting by name or visibility.
- Manual channel ordering.
- Channel notes.
- Channel grouping.
- Channel group colour inheritance controls.
- Visible-channel grouping from deep links.
- Smooth-pixel toggle for linear versus nearest-neighbour display.
- Side-panel show/hide controls for focused viewing.
- Fit-to-view for image or active target.
- Canvas pan and zoom.
- Double-click fit behavior.

## Multidimensional Viewing

- XY-first image inspection model.
- Plane selector for datasets with additional view planes.
- Slice slider and arrow buttons for multidimensional datasets.
- Plane-by-plane inspection of OME-Zarr datasets with a `z` axis.
- Safeguards for operations that only make sense in XY view.

## Large-Image Performance

- On-demand loading of visible tiles.
- Coarse levels appear early during navigation.
- Finer levels replace coarse tiles as they arrive.
- Tile prefetch behavior around the active viewport.
- Contrast changes without reloading decoded tiles in normal workflows.
- Fast object rendering for large polygon/object overlays.
- Option to disable fast object rendering for exact figure links.
- Memory tab for inspecting and controlling tile/pyramid memory behavior.
- Manual channel/level pinning in RAM.
- RAM estimates and warning/danger labels for pinning.
- Focused-ROI and all-ROI pinning in mosaic mode.
- Unpin/unload controls for pinned levels.

## Projects And Workspaces

- Project JSON workspace format.
- Project loading from GUI.
- Recent project access.
- Project save.
- ROI list management.
- ROI selection and focus state.
- Project metadata storage.
- Embedded masks in project state.
- Layer grouping and viewer state storage.
- Saved project view states.
- View capture and apply workflows.
- ROI Selector panel for navigating project ROIs.
- Previous/next ROI navigation.
- Project-linked segmentation/object paths.
- Segmentation search roots.
- Automatic segmentation/object matching.
- Object cache preloading for repeated review.
- Export samplesheet CSV from project data.

## Samplesheets

- CSV samplesheet import from the GUI.
- Samplesheet-driven project creation.
- Required `id` and `path` columns.
- Optional `dataset` column.
- Optional `segpath` column for object/segmentation data.
- Arbitrary metadata columns for browsing, sorting, grouping, and labels.
- Relative path resolution from the samplesheet location.
- Absolute path support.
- TMA-style metadata support, such as row, column, well, patient, cohort, response, batch, and quality.
- Select all, select visible, range select, and multi-select ROI workflows.

## Mosaic Mode

- Multi-ROI viewing on a shared canvas.
- Mosaic opening from selected project ROIs.
- Mosaic opening from imported samplesheets.
- Mosaic opening from OME-Zarr folder-tree discovery.
- Mosaic opening from command line for scripted demos.
- Shared channel visibility across ROIs.
- Shared channel colour and contrast across ROIs.
- Global channel settings for fair cross-ROI comparison.
- Fit full mosaic.
- Double-click to focus one ROI.
- Previous/next core navigation.
- Fit Cells layout mode.
- Native Pixels layout mode.
- Configurable mosaic column count.
- Group ROIs by metadata column.
- Sort ROIs by metadata column.
- Secondary sort.
- Group labels.
- Group spacing/gap controls.
- Text Labels layer for ROI metadata labels.
- Multiple label columns per ROI.
- Mosaic Memory tab for optional RAM pinning.
- Per-ROI segmentation/object path discovery via `segpath`.
- Broad mosaic object/segmentation review.

## Object And Overlay Data

- OME-Zarr label-group discovery.
- NGFF labels rendered as outlines.
- Segmentation object loading from GeoParquet.
- Segmentation object loading from Parquet.
- Segmentation outline loading from GeoJSON / JSON.
- CSV point overlay loading.
- Mask GeoJSON loading.
- SpatialData image, point, shape, and label-like element discovery.
- Xenium dataset discovery from `experiment.xenium`.
- Xenium morphology image loading.
- Xenium cell polygon loading.
- Xenium transcript point loading.
- Object geometry expected in level-0 image coordinates.
- Object offsets and transforms for viewer/project alignment state.
- Lazy property loading for wide object tables.
- Property discovery for loaded and lazy columns.

## Segmentation Object Display

- Segmentation object visibility control.
- Object opacity control.
- Object outline width control.
- Object fill toggle.
- Fill opacity control.
- Selected-object fill opacity control.
- Single-colour object display.
- Single, categorical, and continuous numeric property colouring through a shared typed mapping.
- Named and custom continuous palettes, linear/log10 scales, automatic/fixed domains, reversal,
  clamp/hide semantics, missing-value colour, and a numeric gradient legend.
- Categorical legend generation.
- Legend category show/hide controls.
- Legend category colour overrides.
- Fast rendering toggle.
- Reload object source.
- Clear object layer.
- Polygon object rendering.
- Point-like object rendering.
- Proxy-point rendering for very large object sets.
- Selection overlay display.

## Object Filtering And Queries

- Simple row-based object filters.
- Multiple filter clauses.
- AND / `All` filter logic.
- OR / `Any` filter logic.
- Filter count display, including visible objects after filtering.
- Filters affect rendering, counts, analysis, measurements, and export.
- Lazy loading of properties used in filters.
- Full boolean object query editor.
- Nested `and`, `or`, and `not` expressions.
- Parentheses in object queries.
- Equality and inequality operators.
- Numeric comparison operators.
- `contains`, `starts_with`, and `ends_with`.
- `in [...]` list membership.
- String, number, boolean, and null literals.
- Boolean-column shorthand, such as `zz_mask_cd3`.
- Backtick property names for columns containing spaces or punctuation.
- Deep-link support for simple filters.
- Deep-link support for arbitrary boolean object queries.
- MCP support for reading, setting, and clearing object filters.

## Object Selection

- Click object selection.
- Rectangle object selection.
- Additive rectangle selection.
- Lasso object selection.
- Clear selection.
- Selection count display.
- Selection overlay.
- Primary selected object.
- Zoom/focus workflows for selected objects.
- Selection elements for review workflows.
- Selection state used in analysis and export workflows.
- MCP tools for object selection inspection and rectangle selection.

## Analysis Workflows

- Analysis tab for loaded segmentation objects.
- Numeric object-property histograms.
- Histogram review over filtered object sets.
- Raw and arcsinh-transformed histogram views.
- Threshold value editing.
- Threshold dragging.
- Quantile threshold suggestions.
- K-means threshold suggestions.
- Object selection from threshold rules.
- Calls workflow for marker-positive / marker-negative review.
- Custom call definitions.
- Composite calls using multiple threshold rules.
- Call preset save/load.
- Bind threshold edits to active marker/channel.
- Image-channel to object-property mapping.
- Mapping settings dialog.
- Marker call failure marking.
- Live selection review where supported.
- Analysis warmup for project-linked objects.

## Measurements And Export

- Mean intensity measurement over polygon objects.
- Exact median intensity measurement over polygon objects.
- Measurement over selected pyramid level.
- Filtered-cells-only measurement option.
- Column prefix control for generated measurements.
- Measurements attached back to loaded object properties.
- Measured values immediately available in Analysis.
- Export enriched GeoParquet.
- Export enriched CSV.
- Export measured properties.
- Export derived call columns.
- Export selection columns.
- GeoParquet export preserves geometry.
- CSV export provides tabular output for external inspection.

## Threshold Regions

- Image-channel threshold region preview.
- Visible-region thresholding.
- Entire-image thresholding at selected pyramid level.
- Pyramid-level safeguards for large threshold previews.
- Threshold value control.
- Minimum component size control.
- Refreshable preview.
- Raster threshold preview overlay.
- Apply preview as editable mask polygons.
- Threshold-created masks for artefact, exclusion, tissue, or signal-positive regions.

## Mask Polygons

- Editable mask polygon layers.
- Draw polygon by clicking vertices.
- Close polygon by double-click, `Enter`, or clicking the first vertex.
- Cancel drawing with `Esc`.
- Remove last in-progress vertex with `Backspace`.
- Select polygon by edge or interior.
- Vertex handle editing.
- Whole-polygon move.
- Delete selected polygon.
- Right-click polygon delete action.
- Undo for mask edits.
- Undo for layer-move edits.
- Mask layer loading from GeoJSON.
- Mask layer export to GeoJSON.
- Project-state mask persistence.
- Threshold-created mask cleanup.

## Layer And Alignment Tools

- Layer list with visibility toggles.
- Active layer selection.
- Layer ordering.
- Channel and overlay layers.
- Move tool for selected visible layers.
- Transform tool for supported channel layers.
- Channel translation.
- Channel scaling.
- Channel rotation.
- Viewer/project alignment state without rewriting source image data.
- Overlay transform and offset troubleshooting workflows.

## Deep Links And Reports

- `odon://open` URL scheme.
- Open a project from a deep link.
- Open a specific ROI from a deep link.
- Flexible ROI matching by id, display name, path fragment, or source-key fragment.
- Sample/case/dataset disambiguation.
- Installed example dataset links.
- Active channel selection from links.
- Visible channel lists.
- Hidden channel lists.
- Channel ordering from links.
- Channel grouping from links.
- Channel group colour from links.
- Channel colours from links.
- Active-channel contrast from links.
- Per-channel contrast windows from links.
- Segmentation source selection from links.
- Bundled-label loading control from links.
- Object `Color by` from links.
- Object fill and selection-overlay controls from links.
- Fast object rendering control from links.
- Legend category visibility from links.
- Legend category colours from links.
- Object filters from links.
- Arbitrary boolean object queries from links.
- Camera center and zoom from links.
- OS URL registration for packaged builds.
- Existing-window forwarding through local single-instance IPC.
- Deep-link test page and installed synthetic example.

## MCP And Automation

- Bundled `odon_mcp` helper in release artifacts.
- Stdio MCP server for AI/coding clients.
- Local control bridge to the running Odon GUI.
- Current view inspection.
- Project ROI listing.
- Channel listing.
- Visible channel listing.
- Active channel get/set.
- Visible channel set/show/hide.
- Contrast get/set.
- Channel intensity statistics.
- Channel ordering.
- Channel group listing and editing.
- Side-panel get/set.
- Smooth-pixels get/set.
- Loading-state diagnostics.
- Project opening.
- OME-Zarr opening.
- TIFF / OME-TIFF opening.
- Mosaic samplesheet opening.
- ROI opening.
- Project saving.
- Object overlay visibility get/set.
- Object selection inspection.
- Object rectangle query.
- Object viewport query.
- Object rectangle selection.
- Object selection clearing.
- Object filter state inspection.
- Object filter query application.
- Object filter clearing.
- Camera get/set.
- Zoom in/out.
- Fit to view.
- Right-panel tab selection.
- Mosaic right-panel tab selection.
- Mosaic layout configuration.
- Return to Project page.
- Canvas screenshot capture.
- Full-window screenshot capture.
- Project-page screenshot capture.

## Command Line And Development

- GUI launch from source with `cargo run`.
- Open project from command line.
- Open single OME-Zarr dataset from command line.
- Open project mosaic from command line.
- Open samplesheet mosaic from command line.
- Set initial mosaic columns from command line.
- Run coarse-level sanity checks.
- Command-line deep-link testing.
- Build GUI and MCP helper from source.
- Build documentation locally with MkDocs.

## Packaging And Distribution

- GitHub release artifacts for macOS, Windows, and Linux.
- macOS DMG distribution.
- Windows installer distribution.
- Linux `.deb` distribution.
- Desktop app launch without a separate terminal window in normal packaged usage.
- Bundled MCP helper in release artifacts.
- Installed examples for packaged builds.
- Odon URL scheme registration in packaged builds where supported.

## Example And Utility Data

- Synthetic 5-channel OME-Zarr example.
- Packaged deep-link example for the synthetic 5-channel dataset.
- Synthetic TMA example with 100 cores.
- Synthetic TMA samplesheet with relative paths.
- Synthetic TMA object files in Parquet and GeoJSON forms.
- Synthetic clinical/demo metadata for layout demonstrations.
- OME-Zarr fixture generation script.
- TIFF to OME-Zarr conversion script.
- OME-Zarr rechunking script for viewer-optimized chunks.
- Recommended `bioformats2raw` conversion pattern.

## Known Scope Boundaries

- Odon is primarily a fast viewer and lightweight annotation/review tool.
- XY viewing is the main path.
- Z data is inspected plane by plane, not as a full volume renderer.
- Mosaic mode is strongest for image review.
- Some object, mask, and analysis workflows are more complete in single-image view.
- NGFF labels are currently outline-focused.
- Remote credentials are session-only.
- Heavy downstream statistics are expected to live in companion tools.
