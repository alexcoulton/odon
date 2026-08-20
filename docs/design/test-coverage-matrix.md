# Odon Feature Test Coverage Matrix

Status: Pre-refactor gate implemented
Date: 2026-08-20
Source inventory: `docs/odon-feature-inventory.md`

## How to Use This Matrix

This matrix groups every section of the feature inventory into independently
testable capability families. It is the working index for the pre-Python-API
test improvement project. Individual test cases should link back to an ID here,
and a capability should only move to `Protected` when its important success and
failure behavior is automated.

Coverage states:

- `Protected`: meaningful required automated coverage exists.
- `Partial`: some logic is covered, but important workflow behavior is not.
- `Planned`: an agreed test is in the pre-refactor plan.
- `Extended`: covered only by an explicit optional, scheduled, GPU, large-data,
  or packaging suite.
- `Manual`: intentionally manual with a documented check.
- `None`: no meaningful regression protection identified.

Priorities:

- `P0`: required for the central/Python API refactor gate.
- `P1`: high-value stabilization, started before or continued alongside the
  refactor when its seam is independent.
- `P2`: useful protection that does not block the refactor unless touched.

## Matrix

| ID | Capability family | Inventory coverage | Current state | Required automated protection | Priority |
| --- | --- | --- | --- | --- | --- |
| VIEW-01 | Application startup and viewer modes | Core Viewer | Extended | Headless project/single/mosaic construction is protected; native-window launch and shutdown remain a platform smoke exception | P0 |
| VIEW-02 | Local dataset opening and drag/drop routing | Core Viewer | Protected | Dataset-kind routing plus real single-viewer and mosaic open-to-ready workflows | P0 |
| VIEW-03 | Tile-driven coarse-to-fine viewing | Core Viewer, Large-Image Performance | None | Tile selection/state tests plus one render smoke | P1 |
| VIEW-04 | GPU image compositing | Core Viewer, Image And Channel Viewing | None | CPU render-input tests and canonical visual comparison | P1 |
| CAM-01 | Camera transforms, fit, pan, and zoom | Core Viewer, Image And Channel Viewing | Protected | Pure camera math plus single/mosaic control integration | P0 |
| CHAN-01 | Channel listing, active and visible state | Image And Channel Viewing | Protected | Headless real-fixture state transitions, selector errors, and visibility modes | P0 |
| CHAN-02 | Channel colours and additive compositing | Image And Channel Viewing | Protected | Colour state plus real-fixture CPU additive compositing; GPU output is extended | P0 |
| CHAN-03 | Contrast and histogram controls | Image And Channel Viewing | Protected | Contrast mutation/validation and asynchronous real-fixture histogram results | P0 |
| CHAN-04 | Filtering, sorting, notes, ordering, and groups | Image And Channel Viewing | Protected | Order/group controls, search/sort logic, notes and project restore | P0 |
| CHAN-05 | Smooth pixels and side-panel focused view | Image And Channel Viewing | Protected | Single/mosaic viewer-state and control characterization | P1 |
| PLANE-01 | Z/time/other plane selection | Multidimensional Viewing | Protected | Generated CZYX reads, orthogonal ranges, bounds and multidimensional TIFF selection | P0 |
| PLANE-02 | XY-only operation safeguards | Multidimensional Viewing | Protected | Measurement, pinned-level and channel-max wrong-plane assertions | P0 |
| PERF-01 | Tile loading, replacement, and prefetch | Large-Image Performance | None | Deterministic loader/tile-state component tests | P1 |
| PERF-02 | Fast object rendering | Large-Image Performance, Segmentation Object Display | Partial | Mode/state tests and representative render smoke | P1 |
| PERF-03 | Memory estimates and pin/unpin lifecycle | Large-Image Performance | Partial | Pin/unpin state, limits, single/mosaic tests | P1 |
| PROJ-01 | Project JSON loading and compatibility | Projects And Workspaces | Protected | Version/fixture, invalid-project and transactional failure tests | P0 |
| PROJ-02 | Project save/load round trip | Projects And Workspaces | Protected | ROI, view, mask, group, selection, preset, and mosaic round trip | P0 |
| PROJ-03 | ROI list, selection, focus, and navigation | Projects And Workspaces | Protected | State, focus, selection, matching and mosaic navigation tests | P0 |
| PROJ-04 | Saved ROI views and view presets | Projects And Workspaces | Protected | Validate/save/replace/alias/deep-link and round-trip tests | P0 |
| PROJ-05 | Segmentation search/matching and object preload | Projects And Workspaces | Partial | Candidate matching and preload lifecycle tests | P1 |
| SHEET-01 | Samplesheet parsing and relative paths | Samplesheets | Protected | Required component tests in `tests/data_contracts.rs` | P0 |
| SHEET-02 | Samplesheet validation and arbitrary metadata | Samplesheets | Protected | Required error and metadata tests in `tests/data_contracts.rs` | P0 |
| SHEET-03 | Samplesheet project construction/export | Samplesheets | Protected | Import-to-project, remote-row policy, export and write/read workflow | P0 |
| MOS-01 | Mosaic construction and shared channel state | Mosaic Mode | Protected | Real-fixture samplesheet mosaic and shared channel workflow | P0 |
| MOS-02 | Fit/native layout, columns, gaps, grouping, and sorting | Mosaic Mode | Protected | Pure fit/native layout, aspect, grouping, gap, missing-value, and sort tests | P0 |
| MOS-03 | Mosaic labels, focus, and navigation | Mosaic Mode | Protected | Label-column state, focus stepping/wrap and project persistence | P0 |
| MOS-04 | Mosaic object loading and memory controls | Mosaic Mode | Partial | Component and lifecycle tests | P1 |
| DATA-01 | OME-Zarr metadata and local reads | Core Viewer, Object And Overlay Data | Protected | V2/V3 attributes, CZYX axes, representative planes, histograms and tiles | P0 |
| DATA-02 | TIFF and OME-TIFF opening | Object And Overlay Data, MCP And Automation | Protected | Generated grayscale/RGB/multichannel/multidimensional decode; true pyramids are extended | P0 |
| DATA-03 | Remote HTTP and S3 stores | Core Viewer, Remote And Session State | Partial | URL/credential/prefix behavior protected; mock/real service IO remains extended | P1 |
| DATA-04 | SpatialData discovery and elements | Object And Overlay Data | Protected | Deterministic image/point/shape/label/table discovery and transforms | P1 |
| DATA-05 | Xenium discovery, imagery, cells, transcripts | Object And Overlay Data | Partial | Required minimal fixture plus extended real-data suite | P1 |
| DATA-06 | CSV points, GeoJSON, Parquet, and GeoParquet | Object And Overlay Data | Protected | CSV/GeoJSON/GeoParquet load, properties, geometry, export/reload and failures; vendor point Parquet is extended | P0 |
| LABEL-01 | NGFF label discovery and rendering | Object And Overlay Data | None | Discovery/alignment tests and visual smoke | P1 |
| OBJ-01 | Object layer loading and property discovery | Object And Overlay Data | Protected | Fixture-backed load, typed properties and generated measurement properties | P0 |
| OBJ-02 | Object visibility, opacity, fill, colour, and legends | Segmentation Object Display | Protected | Style persistence, categorical legends and CPU render preparation | P0 |
| OBJ-03 | Polygon, point, proxy, and selection rendering | Segmentation Object Display | Protected | Polygon/point/proxy geometry and selection rendering contracts | P0 |
| OBJ-04 | Reload, clear, and source lifecycle | Segmentation Object Display | Protected | Full load/use/export/clear lifecycle integration | P0 |
| FILT-01 | Simple object filters and All/Any logic | Object Filtering And Queries | Protected | Existing unit tests; add workflow count/render integration | P0 |
| FILT-02 | Boolean query grammar and evaluation | Object Filtering And Queries | Protected | Existing parser/evaluator tests; add lazy-property integration | P0 |
| FILT-03 | Filters affecting analysis, measurement, and export | Object Filtering And Queries | Protected | Filter-to-target, measurement-property and export integration | P0 |
| FILT-04 | Filter deep-link and MCP exposure | Object Filtering And Queries | Protected | Deep-link apply state, control semantics and adapter contracts | P0 |
| SEL-01 | Click, rectangle, additive, and lasso selection | Object Selection | Protected | Geometry, rectangle/additive/lasso and state workflows | P0 |
| SEL-02 | Clear, primary selection, count, and overlay | Object Selection | Protected | Selection snapshot, primary/count/clear and render behavior | P0 |
| SEL-03 | Selection-driven review, analysis, and export | Object Selection | None | Cross-subsystem integration workflow | P1 |
| ANALYSIS-01 | Histograms, transformations, and thresholds | Analysis Workflows | None | Deterministic analysis-state and calculation tests | P1 |
| ANALYSIS-02 | Threshold suggestions and selection | Analysis Workflows | None | Quantile/K-means and selected-ID tests | P1 |
| ANALYSIS-03 | Calls, composites, presets, and mappings | Analysis Workflows | None | State, persistence, and application tests | P1 |
| ANALYSIS-04 | Project-linked analysis warmup | Analysis Workflows | None | Loader/task lifecycle test | P1 |
| MEAS-01 | Mean/median polygon intensity measurement | Measurements And Export | Protected | Real known-pixel mean/median rasterized measurement at a pyramid level | P0 |
| MEAS-02 | Filtered measurement and generated properties | Measurements And Export | Protected | Filtered targets and generated-property filtering integration | P0 |
| EXPORT-01 | GeoParquet export and geometry preservation | Measurements And Export | Protected | Write/read geometry, properties and canonical ID round trip | P0 |
| EXPORT-02 | CSV, calls, properties, and selection export | Measurements And Export | Protected | CSV schema/rows/properties/selection golden assertions | P1 |
| THRESH-01 | Visible/full image threshold preview | Threshold Regions | Partial | Known raster, level safeguard, and preview state tests | P1 |
| THRESH-02 | Component filtering and polygonization | Threshold Regions | Partial | Existing edge cases plus additional raster fixtures | P1 |
| THRESH-03 | Apply preview as editable mask | Threshold Regions | None | Threshold-to-mask integration workflow | P1 |
| MASK-01 | Polygon drawing, editing, deletion, and undo | Mask Polygons | Protected | Polygon closure/edit model, clear and viewer undo transitions | P1 |
| MASK-02 | GeoJSON load/export | Mask Polygons | Protected | Geometry/metadata export and generic reload round trip | P1 |
| MASK-03 | Project mask persistence | Mask Polygons, Projects And Workspaces | Protected | Project and viewer restore including style, offset and geometry | P0 |
| LAYER-01 | Layer visibility, active state, and ordering | Layer And Alignment Tools | Protected | Active/visibility/order state plus project restore | P0 |
| LAYER-02 | Move and overlay translation | Layer And Alignment Tools | Protected | Channel/mask offsets, undo, baseline and persistence | P0 |
| LAYER-03 | Channel translation, scale, and rotation | Layer And Alignment Tools | Protected | Affine state and project round trip | P1 |
| LINK-01 | Deep-link parsing and aliases | Deep Links And Reports | Protected | Existing parser tests | P0 |
| LINK-02 | Project/ROI resolution and disambiguation | Deep Links And Reports | Protected | Matching/disambiguation plus parse-to-view application workflows | P0 |
| LINK-03 | Applying channel, object, legend, filter, and camera state | Deep Links And Reports | Protected | Channel/order/group/colour/contrast/camera/filter state plus staged object colours | P0 |
| LINK-04 | Installed example and test page | Deep Links And Reports | Manual | Packaged smoke test | P2 |
| LINK-05 | OS registration and existing-window IPC | Deep Links And Reports | None | IPC component test and packaged platform smoke | P1 |
| MCP-01 | MCP initialization and tool schemas | MCP And Automation | Protected | Required adapter contract tests in `src/mcp/tools.rs` | P0 |
| MCP-02 | MCP connection to running GUI | MCP And Automation | Protected | Ephemeral TCP envelope/delivery/reply smoke plus root control dispatch semantics | P0 |
| MCP-03 | Viewer/project/channel/camera tools | MCP And Automation | Protected | Project contracts and single/mosaic channel/panel/camera semantics | P0 |
| MCP-04 | Object/filter/mosaic/screenshot tools | MCP And Automation | Protected | Object/filter/mosaic semantics and screenshot queue/worker completion | P0 |
| MCP-05 | Packaged MCP helper | MCP And Automation, Packaging And Distribution | None | Artifact content and launch smoke | P2 |
| CLI-01 | Project, dataset, mosaic, and samplesheet launch | Command Line And Development | None | Argument-routing and representative process tests | P1 |
| CLI-02 | Initial columns, sanity checks, and deep-link arguments | Command Line And Development | Partial | Parser/dispatch tests | P1 |
| DEV-01 | GUI/MCP build and documentation build | Command Line And Development | Partial | CI check/build/docs jobs | P2 |
| PKG-01 | macOS DMG, Windows installer, and Linux package | Packaging And Distribution | Partial | Release artifact creation plus content assertions | P2 |
| PKG-02 | Desktop launch and URL registration | Packaging And Distribution | Manual | Per-platform packaged smoke | P2 |
| PKG-03 | Installed examples | Packaging And Distribution | None | Artifact path and open example smoke | P2 |
| FIX-01 | Synthetic OME-Zarr and deep-link example | Example And Utility Data | Protected | Metadata/pixel validation and single/mosaic open-to-ready workflows | P0 |
| FIX-02 | Synthetic TMA data, objects, and metadata | Example And Utility Data | Manual | Extract/import/mosaic/object workflow | P1 |
| UTIL-01 | Fixture generation | Example And Utility Data | Partial | Deterministic regeneration and metadata assertions | P1 |
| UTIL-02 | TIFF conversion and OME-Zarr rechunking | Example And Utility Data | None | Small input/output validation suite | P2 |
| SCOPE-01 | Documented mode and feature restrictions | Known Scope Boundaries | Protected | Invalid modes, planes, selectors, layouts and unsupported operations assert errors | P0 |

## Current Automated Evidence

The completed gate and current control/API regression suite include:

- 158 unique Rust tests and 163 executions under `cargo test --all-targets`.
- 160 passing executions, three explicit ignored extended fixtures, and no
  failures on the final local run.
- Fifteen passing Python SDK tests cover sync/async connections and disconnect
  cleanup, authenticated discovery, pushed events and iterator shutdown,
  awaitable tasks, revision forwarding, referenced layers, coordinate
  transforms, and declarative UI resources.
- Required CI tests on Linux, macOS, and Windows plus Linux LCOV generation.
- Checked-in and generated OME-Zarr/OME-TIFF data with pixel, plane, histogram,
  measurement, compositing, and failure assertions.
- Project, preset, ROI, samplesheet, mosaic, camera, layer, mask, screenshot,
  SpatialData, object, filter, selection, measurement, and export workflows.
- Single-viewer and mosaic semantic control characterization now exercised
  through the shared typed control core.
- Thin MCP JSON-RPC/schema tests and a parallel-safe ephemeral TCP bridge
  round-trip test.
- Control-protocol conformance tests for authentication, concurrent requests,
  bounded events, revisions, tasks, coordinate descriptors, ownership, project
  persistence, layer lifecycle, and UI tree validation/native bindings.

`Partial` does not mean that most of a capability is covered. It only means some
meaningful automated evidence exists.

## Exception Register

The only P0 row not marked `Protected` is `VIEW-01`, recorded as `Extended` for
the native desktop-window boundary. Project state, a real single viewer, and a
real samplesheet mosaic are all constructed headlessly in required tests. What
is excluded from normal CI is opening a platform window, initializing a real GL
context, processing OS close events, and asserting clean process shutdown.

This exception is narrow because those behaviors are controlled by eframe and
the platform window manager, are unstable on headless runners, and do not define
the semantic contract that the central/Python API refactor will change. Release
validation must still launch and close each packaged application on macOS,
Windows, and Linux. A future virtual-display smoke job can move `VIEW-01` to
`Protected` without changing the refactor gate's semantic tests.

GPU pixel output, real HTTP/S3 services, large vendor datasets, URL registration,
and installer contents remain P1/P2 extended or manual coverage. Their states
are visible in the matrix and are not being represented as normal-suite
protection.

## Refactor Gate View

The pre-refactor gate is satisfied when:

1. Every P0 row must be `Protected`, or have a recorded and accepted exception.
2. Required tests must run in pull-request CI.
3. Fixture-dependent required tests must not silently pass when data is absent.
4. MCP rows should be protected by semantic/control tests plus a thin adapter
   suite, not exhaustive bridge-internal tests.
5. Visual or platform-only behavior may remain `Extended` if its scheduled or
   release job is reliable and its limitations are documented.

This matrix should be updated in the same pull request that adds or materially
changes a feature test.
