# Odon Test Suite Improvement Plan

Status: Implemented pre-refactor gate
Date: 2026-08-20
Sequence: Complete the agreed pre-refactor gate before beginning the Python API
and central control API refactor

## Purpose

Odon has a useful collection of focused unit tests, but it does not yet have a
comprehensive regression suite for its viewer workflows, data sources, project
state, rendering, or external control surface. The planned Python API will
introduce a central typed control layer and move the existing MCP integration
onto it. Before that architectural change, Odon needs a trustworthy baseline
that distinguishes intentional changes from regressions.

This plan improves confidence in existing features without exhaustively testing
MCP or bridge internals that are expected to be replaced. The main investment is
in domain behavior, deterministic data fixtures, feature-level integration
tests, and a small number of representative end-to-end workflows. Those tests
will remain valuable after the control API refactor.

## Pre-Implementation Baseline

Snapshot taken on 2026-08-20 from the current working tree:

| Measure | Current result |
| --- | ---: |
| Unique Rust `#[test]` functions | 79 |
| Test executions under `cargo test --all-targets` | 80 |
| Passing test executions | 80 |
| Failing test executions | 0 |
| Rust source files under `src/` | 108 |
| Source files containing tests | 21 |
| Odon-owned `tests/` integration directory | None |
| Odon-owned coverage report | None |
| Tests run by the main CI workflow | None |
| Fixture-conditional tests doing no work in this checkout | 3 |

The duplicate execution is the checked-in OME-Zarr test running through both a
library and application test target. It is not an additional behavior test.

The current tests are strongest in:

- Object filtering, query parsing, property handling, and rectangle selection.
- Deep-link parsing.
- Array dimensionality and plane mapping.
- OME-TIFF metadata and channel/plane indexing.
- Project ROI matching and relative-path resolution.
- A small amount of settings and channel-panel state logic.
- Threshold-region polygonization.
- One checked-in synthetic OME-Zarr open/read test.

The main CI workflow currently runs `cargo check` on Linux, macOS, and Windows,
but does not run `cargo test`. The repository also has no configured line or
branch coverage measurement.

Three tests return successfully if external fixtures are absent:

- Xenium point Parquet loading.
- ImageJ hyperstack opening.
- Pyramidal OME-TIFF opening.

Those fixtures were absent at the baseline snapshot, so the tests appeared green
without exercising their target code. The initial implementation below corrects
that false-green behavior.

## Implementation Result

Implementation completed on 2026-08-20:

- Added `cargo test --all-targets` to the Linux, macOS, and Windows CI matrix.
- Added a Linux `cargo llvm-cov` job that uploads an LCOV artifact and prints the
  coverage summary without enforcing a premature threshold.
- Converted the three fixture-conditional false-green tests into explicit
  ignored extended tests that fail when run without their required fixture.
- Documented required and extended fixture policy in `fixtures/README.md`.
- Added the initial feature coverage matrix in
  `docs/design/test-coverage-matrix.md`.
- Added samplesheet load, validation, metadata, and write/read contract tests.
- Added project configuration and `ProjectSpace` save/load round-trip tests,
  including sources, masks, channel groups, ROI view state, view presets, mosaic
  state, focus, and selection.
- Added local dataset routing and Zarr v2/v3 attribute contract tests.
- Added camera transform, pan, zoom, clamping, and fit tests.
- Added minimal MCP initialization, tool-schema, uniqueness, notification, and
  error contract tests without locking down the bridge transport.
- Added deterministic generated grayscale, chunky RGB, and two-channel
  OME-TIFF tests that validate metadata, channel layout, physical pixel size,
  channel names/colours, and decoded pixels. Real pyramidal and hyperstack
  files remain explicit extended fixtures.
- Added a headless application characterization harness over the checked-in
  five-channel OME-Zarr fixture. It protects active/visible channel controls,
  atomic selector failure, visibility modes, contrast validation, side panels,
  smoothing, object-overlay visibility, empty-object control errors, pre-load ID
  filters, and deep-link application of channel order, grouping, colour,
  contrast, camera, and rendering mode.
- Added deterministic mosaic fit-cell, native-pixel, grouped-layout, metadata
  sorting, missing-value, aspect-ratio, padding, and group-gap tests.
- Added a real samplesheet-backed mosaic workflow covering shared channels,
  visibility, contrast, grouping, sorting, layouts, focus navigation, and project
  state capture.
- Added full GeoJSON object lifecycle coverage: property discovery, filters,
  measurement-derived properties, rectangle/additive selection, legends, CSV
  export, GeoParquet export/reload, and clear/reload state.
- Added deterministic CSV point-object import with row-level validation.
- Added real mean and median polygon measurements against independently checked
  pixels from the synthetic OME-Zarr fixture.
- Added generated CZYX OME-Zarr plane reads and generated multidimensional
  OME-TIFF Z-plane selection and decoding.
- Added asynchronous histogram and tile-worker tests over the checked-in fixture,
  including generation cancellation and additive CPU compositing.
- Added mask edit/GeoJSON/project contracts plus viewer-level layer ordering,
  visibility, translation, channel affine transforms, undo, and restore.
- Added screenshot worker encoding/completion and filename-policy tests.
- Added deterministic SpatialData image/label/point/shape/table discovery and
  transform tests, plus remote-store validation and prefix behavior.
- Added an ephemeral-port TCP bridge test for validation, request delivery, and
  app reply round trips. Fixed production-port internals are intentionally not
  frozen.
- Recorded narrow extended/manual exceptions for GPU pixel output, real cloud
  services, large vendor fixtures, desktop launch/URL registration, and packaged
  installer smoke tests. These do not replace semantic P0 coverage.

The implementation found and fixed four production defects:

- Failed project loads erased the currently open project instead of being
  transactional.
- Enriched GeoParquet export omitted canonical object IDs.
- Equal-geometry multidimensional OME-TIFF IFDs could not select a Z/T plane.
- A malformed CSV coordinate row could cause an unrelated numeric column to be
  silently substituted for X or Y.

Latest local verification on 2026-08-20:

| Measure | Result |
| --- | ---: |
| Unique Rust `#[test]` functions | 158 |
| Test executions under `cargo test --all-targets` | 163 |
| Passing test executions | 160 |
| Explicitly ignored extended tests | 3 |
| Python SDK test executions | 15 |
| Failing test executions | 0 |

Five data-module tests execute through both the library and application targets,
accounting for executions beyond the unique function count. The coverage job is
configured, but its numeric baseline will be established by the first CI run
because the local toolchain does not include `llvm-tools-preview`.

Since the gate snapshot, the Python API implementation added six unique Rust
tests and expanded the Python suite to 15 executions. These additions cover
discovery and token authentication, event backpressure and close wakeups,
optimistic revisions, task lifecycle, resource ownership and coordinates,
project descriptor persistence, declarative UI reconciliation and native
bindings, concurrent protocol requests, and end-to-end registry round trips
without weakening the original pre-refactor gate.

## Objectives

Before starting the Python API refactor, the test suite should:

- Run automatically for every pull request.
- Fail when required test data or required behavior is absent.
- Cover the semantic behavior that must survive the control API refactor.
- Exercise representative project, single-viewer, mosaic, layer, object, and
  screenshot workflows.
- Protect data and coordinate transformations independently of GPU rendering.
- Provide a deliberate but small MCP compatibility baseline.
- Make current gaps visible through a feature-to-test matrix.
- Establish measured coverage without optimizing for a misleading global
  percentage.
- Remain deterministic, reasonably fast, and portable across supported desktop
  platforms.
- Introduce reusable fixtures and test seams needed by the later central API,
  Python SDK, declarative UI, and Cellpose integration tests.

## Non-Goals

This pre-refactor work will not:

- Exhaustively test the fixed-port MCP bridge, its five-second timeout, or its
  current internal threading arrangement.
- Preserve successful responses containing `{"error": ...}` as desired behavior.
- Make the current untyped bridge request format a permanent compatibility
  contract.
- Build the new central control protocol or Python SDK early.
- Require pixel-identical GPU output across all hardware and drivers.
- Attempt to reach an arbitrary high global coverage percentage by testing
  trivial accessors.
- Automate every manual visual judgment before the API work can begin.
- Test the vendored TIFF crate as if its upstream test suite were Odon
  integration coverage.

## Guiding Principles

### Protect behavior, not the current architecture

Tests should describe outcomes such as visible channels, camera state, selected
objects, loaded ROIs, saved project state, or a written screenshot. Avoid tests
that depend on the current `RootApp` match statement, port number, channel type,
or private helper arrangement.

### Use the lowest useful test layer

Pure coordinate math belongs in unit tests. Project round trips belong in
fixture-backed integration tests. A complete open-and-render workflow belongs in
a small number of end-to-end smoke tests. Do not force every behavior through a
GUI window.

### Test data must be deterministic

Required fixtures are checked in, generated deterministically, or downloaded in
an explicit opt-in suite with checksum verification. A required test must not
return `Ok` merely because its data is missing.

### Separate semantic and visual correctness

Semantic tests verify state, geometry, loaded metadata, selections, transforms,
and commands. Visual tests verify a limited set of render outputs. Most feature
coverage should not depend on a GPU screenshot.

### Make asynchronous completion explicit

Tests should wait for defined readiness, task completion, or output existence.
Avoid fixed sleeps. This requirement will directly inform the future task and
event APIs.

### Treat coverage as evidence, not the goal

Measure line and branch coverage, track its direction, and inspect uncovered
critical code. A feature-risk matrix and meaningful assertions are more valuable
than maximizing one repository-wide number.

## Test Layers

### Unit tests

Fast tests for pure or narrowly stateful behavior:

- Parsers and validation.
- Coordinate transforms and dimensionality.
- Camera calculations.
- Channel resolution and state transitions.
- Layer ordering, ownership, and style state.
- Project serialization helpers.
- Object queries, filters, and geometry.
- Mosaic layout calculations.
- Data-source metadata interpretation.
- Declarative state models extracted from egui rendering.

Target: run on every platform in normal CI.

### Component tests

Tests for a substantial Odon subsystem with real fixture data but no complete
desktop window:

- Open an OME-Zarr store and read representative tiles/planes.
- Load a samplesheet into project state.
- Load GeoJSON, Parquet, GeoParquet, labels, masks, or points.
- Save and reload a project.
- Build a mosaic layout from project ROIs.
- Apply a deep link to a testable viewer-state model.
- Queue a screenshot request into a test screenshot sink.

Target: run on Linux in every pull request and on all platforms where inexpensive
and stable.

### Control characterization tests

Tests that record the externally meaningful results of current control
operations without promising the current transport implementation:

- Inspect current state.
- Change channels, contrast, panels, and camera.
- Open a project or ROI.
- Select and filter objects.
- Configure a mosaic.
- Request and complete a screenshot.

These tests should be written so their expected results can move to the future
typed control core. A temporary adapter or state harness is acceptable; the
fixed bridge envelope is not the subject under test.

### Adapter contract tests

Thin tests for MCP-specific translation:

- MCP initialization and tool listing.
- Tool schemas are valid and documented tool names remain present.
- Unknown tools and invalid arguments return appropriate MCP errors.
- Every MCP-exposed tool maps to a known semantic operation.
- A few representative calls cross the full MCP-to-Odon boundary.

Target: cover adapter behavior without repeating every central semantic test.

### End-to-end smoke tests

A small number of workflows using a real Odon application and deterministic
fixtures:

- Launch and open the synthetic OME-Zarr fixture.
- Open a project ROI and reach a ready single viewer.
- Open multiple ROIs in mosaic mode.
- Change a channel and camera state through external control.
- Load an object or label layer.
- Capture a screenshot and verify that it completes.
- Return to the project page and shut down cleanly.

These tests may initially run in a dedicated Linux CI job with a virtual display
and software rendering, then expand to platform smoke tests.

### Visual regression tests

Use a small curated set of canonical views:

- Synthetic multichannel image.
- Image plus label/object overlay.
- Mosaic arrangement.
- Project page or important UI state.

Compare perceptually with explicit tolerances rather than requiring identical
PNG bytes across GPU drivers. Store the view configuration and fixture alongside
each approved reference image. Visual updates require human review.

### Packaging smoke tests

Verify release artifacts contain and can locate:

- The Odon executable.
- `odon_mcp`.
- Required examples and fixtures intended for installation.
- Deep-link registration metadata or helper components.

Keep installer tests separate from ordinary unit and integration tests.

## Initial Feature-Risk Matrix

This is a preliminary prioritization. The full matrix should be derived from
`docs/odon-feature-inventory.md` during Phase 0.

| Area | Current protection | Main missing protection | Priority |
| --- | --- | --- | --- |
| OME-Zarr metadata/open | One checked-in fixture test | Axis variants, invalid metadata, tile reads, multiple planes | P0 |
| TIFF/OME-TIFF | Metadata/index unit tests | Required real fixtures and decoded plane integration | P0 |
| Projects | ROI matching/path tests | Save/load round trip, view state, masks, groups, failure cases | P0 |
| Samplesheets | Little or none | CSV validation, relative paths, metadata, project construction | P0 |
| Single viewer state | Small helper tests | End-to-end channels, camera, plane, layers, readiness | P0 |
| Channels/contrast | Search and alias helpers | Visibility, order, colour, grouping, contrast mutations | P0 |
| Camera | None | Fit, zoom, pan, viewport conversion, bounds | P0 |
| Objects | Good pure-logic coverage | Loading, property integration, styling, selection workflow | P0 |
| Layers/overlays | Sparse | Lifecycle, ordering, transforms, visibility, persistence | P0 |
| Mosaic | One RAM estimate test | Layout, grouping, sorting, shared channels, focus/navigation | P0 |
| Deep links | Parser coverage | Applying links to project/viewer state and IPC forwarding | P0 |
| MCP/control | None | Semantic characterization and small adapter contract suite | P0 |
| Screenshots | None | Completion, path handling, dimensions, representative output | P0 |
| SpatialData/Xenium | A few parsing/load tests | Discovery and complete layer construction | P1 |
| Labels | Sparse | Discovery, level alignment, outline/fill rendering state | P1 |
| Annotations | None | Create/edit/select/save/reload behavior | P1 |
| Masks | None | Import, visibility, exclusion, persistence | P1 |
| Remote stores | None | Mocked HTTP/object-store behavior, auth failures, retries | P1 |
| Memory/pinning | Small array helper tests | Pin/unpin lifecycle, limits, mosaic behavior | P1 |
| Thresholding | Polygonization and a few defaults | Preview, application, export, cancellation | P1 |
| Native panels/menus | Very limited | State-model tests and a few egui component tests | P2 |
| GPU renderers | Pure helper tests only | Canonical render smoke and visual comparisons | P1 |
| Packaging/deep-link registration | Build workflows only | Artifact content and launch smoke tests | P2 |

Priority meanings:

- P0: needed for the pre-Python-API regression gate.
- P1: should begin before the refactor and may be completed in parallel if its
  test seam is independent.
- P2: valuable stabilization work, but not a blocker unless the relevant code is
  touched by the refactor.

## MCP Scope During This Work

The current MCP feature set is user-visible and must not disappear accidentally,
but most current MCP internals are scheduled to change.

### Test now

- The documented MCP tool-name inventory.
- MCP `initialize`, `tools/list`, unknown-method, and malformed-request behavior.
- Input schema validity for every listed tool.
- Representative semantic workflows through MCP: inspect view, open data, set
  channels/contrast, move camera, and select/filter objects.
- Clear behavior when the GUI is unavailable.

### Do not lock down

- Fixed port `127.0.0.1:17870` as a permanent contract.
- Exact bridge thread names or channel types.
- Five-second timeout as desired behavior.
- The unversioned `{method, params}` internal envelope.
- The duplicated allow-list and `RootApp` dispatch layout.
- Current successful result envelopes that contain application errors.

### Migrate later

When the typed control core exists:

- Move semantic expectations to central command tests.
- Replace repeated MCP tool tests with a registry-wide mapping/schema test.
- Retain a few black-box MCP end-to-end tests.
- Remove temporary characterization harnesses once they are demonstrably
  superseded.

## Fixture Strategy

### Checked-in minimal fixtures

Keep small, deterministic fixtures for:

- Multiscale multichannel OME-Zarr.
- OME-Zarr with Z and/or time axes.
- Small pyramidal OME-TIFF.
- Small ImageJ hyperstack if it represents supported behavior.
- Project JSON with relative and absolute-path variants.
- Samplesheet CSV with representative metadata and error cases.
- Object polygons and points in supported formats.
- Labels and masks.
- SpatialData-style metadata and minimal elements.
- Xenium metadata and small representative tables where licensing permits.

Each binary fixture needs a short provenance file describing how it was created,
its license, expected metadata, and regeneration command.

### Deterministically generated fixtures

Prefer generation when it is simple and tests the file writer independently
enough to avoid circular verification. Existing fixture-generation scripts can
be adapted into a documented test-fixture workflow.

Generated fixtures must be byte-stable where feasible, version-pinned, and
created before tests rather than silently inside each test. If generation is
expensive, check in the output and test the generator separately.

### External or large fixtures

Large real-world files belong in an explicit extended suite. Requirements:

- Opt-in environment flag or dedicated CI job.
- Immutable URL or artifact version.
- Checksum.
- Cache key.
- Clear skip reporting.
- No impact on the required fast suite when unavailable.

Use Rust's `#[ignore]` with a documented invocation for genuinely optional tests
rather than returning success when a file is missing.

### Temporary outputs

Use unique test temporary directories and clean them after successful tests.
Tests must never write into user configuration, recent-project state, source
fixture directories, or the repository root.

Introduce a small shared fixture/path helper instead of repeating environment
and path assumptions.

## Testability Improvements

Some current Odon code combines state mutation, egui rendering, GPU interaction,
filesystem IO, and asynchronous loading. Tests should motivate narrow seams
without prematurely implementing the future public API.

### Extract state transitions

Move calculations and state transitions into functions or types that can be
tested without a window:

- Camera fit/zoom calculations.
- Channel visibility/order/group mutations.
- Mosaic layout and grouping.
- Layer lifecycle and ordering.
- Project view-state application.
- Deep-link application planning.
- Screenshot request validation and output planning.

The egui code calls these functions; tests verify them directly.

### Abstract side effects narrowly

Use small internal traits or injected services where necessary:

- Filesystem/output sink for screenshots and saves.
- Clock or task completion signal where timing matters.
- Dataset opener for error and transition tests.
- Repaint notifier for control queue tests.
- Remote object-store/HTTP client for deterministic network tests.

Do not introduce a broad mocking framework or duplicate the whole application
behind traits.

### Add a deterministic application harness

Create a test harness capable of:

- Constructing project, single-viewer, and mosaic state from fixtures.
- Executing a semantic action or current control operation.
- Pumping loader/task messages until a predicate or bounded timeout.
- Inspecting state snapshots.
- Substituting a screenshot sink.
- Avoiding user settings and filesystem locations.

The harness may initially sit under `tests/support/` or a `#[cfg(test)]` module.
Design it so it can later exercise the typed control core without a rewrite.

### Isolate renderer inputs

GPU renderer tests should separate:

- CPU preparation: tile selection, vertices, colours, label boundaries, bins,
  and transforms.
- GPU resource upload and shader execution.
- Final composite output.

Extensive unit tests can protect CPU preparation. A smaller GPU suite protects
the rendering integration.

## CI Plan

### Required pull-request jobs

#### Fast Rust tests

Run on Linux, macOS, and Windows:

```bash
cargo test --all-targets
```

Initially this can be added to the existing OS matrix. Cache Cargo compilation
artifacts only if cache complexity does not hide reproducibility issues.

#### Formatting and linting

If not already enforced elsewhere:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

Lint policy changes should be introduced separately from test behavior so a
large existing warning backlog does not block the first test job.

#### Linux component/integration tests

Run fixture-backed tests and headless-compatible application tests. Keep the
command simple, for example ordinary `cargo test`, unless a separate ignored or
feature-gated suite is genuinely required.

### Coverage job

Use `cargo llvm-cov` on Linux to produce:

- Human-readable summary.
- LCOV or Cobertura artifact.
- Per-file and per-module visibility.

The first coverage PR records a baseline and uploads the report without a hard
global threshold. After critical gaps are addressed:

- Prevent unexplained material regression from the agreed baseline.
- Apply stronger expectations to new pure logic and the future control core.
- Exclude generated code deliberately and visibly.
- Do not exclude difficult UI or renderer code merely to improve the number.

### Scheduled or release jobs

- GPU/visual smoke suite.
- Large fixture suite.
- Packaged artifact launch and MCP-presence checks.
- Remote-store tests using controlled local services.
- Platform-specific deep-link and installer checks.

Failures in required release protection must be actionable and not routinely
ignored as flaky.

## Proposed Test Layout

An illustrative structure:

```text
tests/
  support/
    mod.rs
    app_harness.rs
    fixtures.rs
    snapshots.rs
  fixtures/
    README.md
    projects/
    samplesheets/
    geojson/
    parquet/
  data_sources.rs
  project_roundtrip.rs
  samplesheet_import.rs
  viewer_state.rs
  mosaic_state.rs
  layers_and_objects.rs
  deep_link_application.rs
  control_characterization.rs
  mcp_contract.rs
  screenshots.rs
```

Fixtures already appropriately located under the repository's existing
`fixtures/` directory may remain there. Choose one documented convention rather
than duplicating large data.

Unit tests should continue to live beside their Rust modules. Cross-module
workflows belong in `tests/`.

## Phased Delivery

### Phase 0: Inventory and test policy — complete

Deliverables:

- Convert `docs/odon-feature-inventory.md` into a coverage matrix with columns
  for unit, component, control, end-to-end, visual, manual, and gap status.
- Classify features P0, P1, or P2 by regression risk and relevance to the API
  refactor.
- Record the current test and coverage baseline.
- Agree fixture location, naming, licensing, and regeneration policy.
- Agree what constitutes the pre-refactor gate.

Exit criteria:

- Every documented feature has a known coverage status.
- Every P0 feature has an owner or planned test task.
- Tests that intentionally remain manual are explicit.

### Phase 1: Make the current suite trustworthy in CI — complete

Deliverables:

- Add `cargo test --all-targets` to pull-request CI.
- Replace the three silent fixture-dependent passes with required fixtures,
  deterministic generators, or explicit ignored extended tests.
- Add a shared temporary-directory and fixture helper.
- Ensure tests do not read or write user settings.
- Add a Linux coverage-report job and record the baseline.

Exit criteria:

- CI runs and passes all required current tests on Linux, macOS, and Windows.
- Missing required fixtures fail clearly.
- There are no tests that claim success without exercising their named behavior.
- A browsable coverage artifact is produced.

### Phase 2: Protect core state and data contracts — complete for P0

Deliverables:

- Camera calculation tests.
- Channel visibility, order, groups, colour, and contrast state tests.
- Project save/load/view-state round-trip tests.
- Samplesheet validation and project-construction tests.
- Dataset axes, planes, coordinate, and transform tests.
- Required OME-Zarr and TIFF fixture-backed reads.
- Layer lifecycle, ordering, visibility, style, and transform tests.
- Expanded mosaic layout/group/sort tests.
- Annotation and mask persistence tests for supported behavior.

Exit criteria:

- P0 pure state transitions have deterministic tests.
- Core project and data contracts survive a save/load or open/read round trip.
- Coordinate-space tests cover the conventions the Python API will later expose.

### Phase 3: Application and control characterization — complete for P0

Deliverables:

- Deterministic application state harness.
- Deep-link parse-and-apply tests, not only parser tests.
- Single-viewer and mosaic transition tests.
- Readiness and loader-success/failure tests.
- Semantic characterization for each existing control operation.
- Screenshot request/completion tests with a controlled sink.
- A checked-in inventory of current MCP tools and schemas.
- Minimal MCP initialization, listing, validation, and representative call tests.

Exit criteria:

- Every existing control operation has either a reusable semantic test or an
  explicit documented reason it is covered by a higher-level workflow.
- MCP compatibility is protected without freezing bridge internals.
- The expected behavior to migrate into the central control core is executable,
  not only written in documentation.

### Phase 4: End-to-end workflow smoke suite — semantic workflows complete;
native-window portion extended

Deliverables:

- Headless or virtual-display Odon launch harness.
- Synthetic OME-Zarr open-to-ready workflow.
- Project ROI and mosaic workflows.
- Channel/camera manipulation through external control.
- Object or label layer load workflow.
- Screenshot completion workflow.
- Clean shutdown and temporary-output cleanup checks.

Exit criteria:

- The primary P0 workflow suite runs reliably in CI.
- Failures produce logs, state, and artifacts sufficient for diagnosis.
- No workflow relies on unbounded sleeps.

### Phase 5: Render and visual protection — CPU contracts complete; GPU visual
corpus extended

Deliverables:

- CPU-side renderer input tests for key image, label, object, point, and mosaic
  paths.
- A small canonical screenshot corpus.
- Perceptual comparison tooling and reviewed tolerances.
- A scheduled or required Linux visual job, based on measured stability.
- Clear process for intentional golden-image updates.

Exit criteria:

- Major missing layers, transforms, colours, or layouts are detected.
- Expected cross-platform or driver variance is understood and documented.
- Visual failures retain actual, expected, and difference artifacts.

### Phase 6: Gate review before the Python API refactor — complete

Deliverables:

- Review the feature coverage matrix and all remaining P0 gaps.
- Run the full required suite from a clean checkout.
- Review coverage by risk area, not only globally.
- Mark temporary characterization tests that will migrate to the new control
  core.
- Record known gaps and whether they block the refactor.

Exit criteria:

- All required CI jobs are green.
- Every P0 feature has meaningful automated regression protection.
- No required test silently skips for missing data.
- Current MCP semantic compatibility has a clear baseline.
- The team agrees that remaining gaps are understood and acceptable.
- The Python API refactor can begin with tests acting as its regression oracle.

## Suggested Pull Request Sequence

Keep early changes reviewable and avoid mixing broad production refactors with
test infrastructure:

1. Add tests to CI without changing application behavior.
2. Fix conditional fixture tests and document fixture policy.
3. Add coverage reporting and record the baseline.
4. Add project and samplesheet round-trip tests.
5. Add camera, channel, coordinate, and layer state tests.
6. Add mosaic state and layout tests.
7. Add deterministic application harness.
8. Add control characterization and minimal MCP contract tests.
9. Add end-to-end synthetic fixture workflow.
10. Add renderer preparation and selected visual regression tests.
11. Review the pre-refactor gate.

Production code extraction needed for testability should be delivered in focused
commits with behavior-preserving tests before and after the move.

## Preliminary Pre-Refactor Gate

The Python API work should not be blocked on perfect coverage. It should be
blocked on missing confidence in the behavior most likely to be affected.

Minimum proposed gate:

- `cargo test --all-targets` runs in CI on Linux, macOS, and Windows.
- Required tests use deterministic available fixtures.
- Coverage is measured and visible.
- Project load/save and samplesheet import have round-trip tests.
- OME-Zarr and TIFF have required open/read fixture tests.
- Camera and channel state transitions have unit tests.
- Single viewer, mosaic, layers, objects, and screenshot completion each have at
  least one integration workflow.
- Deep links are tested through application of state, not only parsing.
- Every existing external control method has a recorded semantic expectation.
- MCP has schema/listing tests and a small representative end-to-end suite.
- The primary synthetic dataset workflow runs from open through ready and
  screenshot.
- Remaining P0 gaps are explicitly accepted rather than unknown.

No global percentage is proposed as an initial gate until the first instrumented
coverage report is reviewed. After baseline review, set targeted expectations
for pure/domain modules and a non-regression rule for the repository as a whole.

## Success Measures

The plan is succeeding when:

- Regressions are found by CI rather than manual release testing.
- Test failures identify a semantic feature rather than an incidental private
  implementation.
- Required fixture tests cannot silently pass without their data.
- Developers can run the normal required suite locally with one Cargo command.
- Application workflow tests do not depend on arbitrary sleeps.
- Coverage reports point reviewers toward meaningful untested code.
- MCP adapter changes require few test changes when semantic behavior remains
  stable.
- Tests written before the refactor now exercise the shared typed control core.
- The central API refactor can simplify internal architecture while preserving
  demonstrated Odon behavior.

## Risks and Mitigations

### GUI tests become flaky

Mitigation: move most behavior below egui, use explicit readiness predicates,
limit full-window tests, and preserve diagnostics and artifacts on failure.

### Visual tests differ across GPUs

Mitigation: use deterministic software rendering where possible, perceptual
tolerances, a small corpus, and semantic renderer-input tests.

### Fixtures make the repository too large

Mitigation: design minimal synthetic fixtures, generate deterministic data, and
keep large real-world cases in a checksummed extended suite.

### Testability extraction accidentally starts the API refactor

Mitigation: keep seams internal, behavior-preserving, and narrowly motivated by
tests. Do not define the public protocol during this work.

### MCP tests are discarded immediately

Mitigation: test semantic outcomes and schema inventory, not bridge mechanics;
migrate expectations to the typed core and retain only thin adapter tests.

### Coverage work rewards low-value tests

Mitigation: review coverage alongside the feature-risk matrix and do not impose
an arbitrary global threshold initially.

### CI becomes too slow

Mitigation: keep unit/component tests in the required fast lane, cache carefully,
and move large, visual, remote, or packaged suites into clearly defined jobs.
Measure before introducing a new test runner.

## Deferred P1/P2 Decisions

1. Which existing TIFF and Xenium fixtures can be checked in, generated, or
   redistributed legally?
2. Should repository fixtures remain under `fixtures/` or should test-specific
   fixtures live under `tests/fixtures/`?
3. Which user settings and platform services require injection to construct a
   deterministic application harness?
4. Can egui/eframe and the current renderer run reliably with software rendering
   in Linux CI, or is a narrower render harness required?
5. Which screenshot comparison library or small custom comparator should be
   used?
6. What coverage baseline and critical-module targets are appropriate after the
   first `cargo llvm-cov` report?
7. Which MCP tool names must remain backward compatible through the control API
   refactor, and which may be intentionally deprecated?
8. Which packaging smoke tests are required on every release versus scheduled?

## Follow-Up After the Refactor Gate

1. Review the first CI LCOV artifact and record numeric line/branch baselines.
2. Run the three ignored extended tests when redistributable/checksummed fixtures
   are available.
3. Add virtual-display GPU visual and packaged launch/close jobs when their
   stability has been measured.
4. Keep the feature matrix current as P1/P2 areas are touched.
5. Begin the central control API refactor by moving the protected semantic
   control expectations behind the typed core.

The first measured coverage report may reorder P1 work, but the P0 semantic
baseline is now executable and can serve as the regression oracle for the
central API refactor.
