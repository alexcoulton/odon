# Control Actor Refactor Completion Plan

Status: in progress — Milestones 0 through 5 complete; Milestone 6 is next

Date: 2026-08-23

## Objective

Finish the control-actor refactor so that Python, MCP, native UI, menus, and deep links all use one
canonical command and state boundary, and so that ordinary application work continues when Odon is
covered, minimized, or placed in another Space.

The registered method migration is already complete. This plan is therefore not a route-migration
plan. It is an ownership-cleanup, compatibility-deletion, and release-evidence plan.

At completion:

- the control actor is the only canonical owner of remotely observable semantic state;
- all non-presentation commands and retained tasks can progress without an egui frame;
- the renderer consumes immutable projections and shared generation-tagged resources;
- the renderer cannot write semantic snapshots back into the actor;
- native UI commits use the same typed commands as Python and MCP;
- pixel-dependent work explicitly waits for presentation without blocking the actor mailbox;
- renderer-side command emulators and obsolete compatibility state are removed;
- covered, minimized, and separate-Space behavior is verified on real supported platforms; and
- the complete Rust, Python, schema, reference, and application-surface suites pass.

## Current baseline

The implementation starts this plan with the following evidence:

- 266 of 266 registered application methods are actor-owned in every supported mode and target;
- there are zero legacy or hybrid registered application routes;
- the TCP two-viewer comparison completes without a UI frame;
- resource and compute families have paused-frame tests;
- the production legacy UI request dispatcher and snapshot-event publisher are absent;
- native project, viewer, and mosaic commits enter typed actor commands;
- macOS visible, covered, minimized, and separate-Space verifier runs completed all 81 operations;
- the restored macOS window provisionally displayed the final projection without another API call;
- the most recent recorded cumulative suites passed; and
- source-level modularity guards cover the actor, application, registry, bridge, models, renderers,
  and major resource adapters.

The remaining gap is architectural rather than method-count based. `OmeZarrViewerApp`, `RootApp`,
and renderer adapters still contain compatibility mirrors, snapshot assemblers, test-only command
emulators, and mixed renderer/transient state that must be narrowed or deleted.

The stabilized migration base, executable ownership ledger, renderer-emulator retirement,
viewport/presentation migration, and object/mask/annotation migration are separately checkpointed
milestones. Milestone 5 has already removed renderer-owned settings persistence, redundant project
preload fields, renderer-local remote dataset I/O, renderer-side label discovery and loading, local
TIFF-plane and tile-policy mutation, threshold projection/draft overlap, renderer-owned image
histogram and automatic-contrast workers, and renderer-local memory-pinning workers.

### Current measured checkpoint

The executable ledger currently covers 289 concrete host fields. It has 264 retained fields and 25
fields in open `narrow` or `replace` rows. There are no open `delete` rows and no renderer semantic
command emulators.

| Milestone | Open rows | Open fields | Remaining ownership domains |
| --- | ---: | ---: | --- |
| 5 | 0 | 0 | complete |
| 6 | 9 | 12 | host requests, native command outboxes, root mode/deep-link/projection relays, mosaic shell state |
| 7 | 2 | 13 | single-view and mosaic screenshot/presentation state |
| **Total** | **11** | **25** | |

`Retain` means the field has a valid final role as a renderer resource/observation, transient UI
draft, shared actor resource, or platform effect. It does not mean that the renderer owns semantic
state. A row closes only when its fields have been split into those precise roles and the ledger and
source guards describe the resulting boundary.

## Completion gates

All four gates are required. Passing one gate must not be presented as completing another.

### Gate A — routing and authority

- Every registered semantic or resource method has only actor execution routes.
- Every native semantic commit enters the same typed command used by remote clients.
- No production renderer module implements command semantics.
- No renderer observation can overwrite actor-owned semantic state.
- No frame-driven fallback is constructed when the actor or TCP publication fails.

### Gate B — state ownership

- Each field in `OmeZarrViewerApp`, `RootApp`, and `MosaicViewerApp` is classified as actor
  projection, shared resource handle, renderer state, transient UI state, or platform effect.
- Durable semantic fields exist only in the actor model.
- Renderer projections retain only data required to render the current or latest generation.
- Large document, object, label, mask, and analysis resources cross the boundary through stable
  shared handles rather than per-projection copies.
- Test-only renderer command emulators have been replaced by actor-plus-projection fixtures.

### Gate C — background behavior

- Every non-presentation method family has a no-frame execution test.
- Long-running work is a retained, observable, cancellable task on a bounded worker service.
- Covered/minimized execution does not depend on repaint requests or `RootApp::update`.
- Projection coalescing retains the latest complete state while no frames are consumed.
- Returning to Odon renders that state without another API or UI command.
- Presentation work reports a generation-specific waiting phase and leaves the actor responsive.

### Gate D — release evidence

- Full Rust and Python suites pass on the final tree.
- Protocol schemas, generated Python reference, and the application-surface manifest are synchronized.
- Latency, bounded-queue, shared-resource, and frame-planning checks pass.
- macOS visible, covered, minimized, and separate-Space results have recorded visual sign-off.
- Supported Windows and Linux covered/minimized results pass, or the release records an explicit
  platform exclusion with its reason and reviewer.

## Rules for the remaining work

1. Ownership removal, not line count, determines the next target.
2. Do not split a compatibility implementation merely to preserve it in smaller files.
3. Migrate one coherent field family at a time and delete its superseded mutation path in the same
   change.
4. Preserve native interaction quality: drag, hover, drawing, and text-entry previews may remain
   transient and local, but their committed result must be a typed command.
5. Do not move GPU objects, egui state, window handles, or frame-local geometry into the actor.
6. The actor must never synchronously wait for a frame, filesystem operation, network operation,
   decoder, inference job, export, or GPU operation.
7. Worker results are immutable and generation-tagged; only the actor validates and commits them.
8. Add characterization evidence before deleting a path whose response, event, revision, or
   persistence behavior is not already explicit.
9. Keep each milestone independently testable and commit it before starting the next ownership
   domain.

## Milestone 0 — stabilize and checkpoint the current tree

Purpose: establish a trustworthy base containing the completed route migration, Wave 6 deletion,
acceptance harness, and recent modularization.

Work:

- audit tracked and untracked files and separate generated screenshots, local fixtures, and review
  artifacts from source changes;
- ensure every intended new module is tracked and every deleted legacy module is intentional;
- run formatting, compilation, Rust suites, Python suites, generated-reference checks, registry
  audits, and application-surface checks;
- reconcile the status statements in the actor plan, ownership inventory, and acceptance record;
- record the exact base commit and test counts; and
- commit the stabilized architecture and modularization in reviewable commits before beginning
  ownership deletion.

Exit criteria:

- the worktree contains no unexplained source or generated changes;
- all cumulative checks pass or every intentional unavailable fixture is recorded;
- the acceptance record describes the exact tested build; and
- the current work is committed with a reviewable history.

## Milestone 1 — make the ownership ledger executable

Purpose: replace broad Wave 7 labels with an exact deletion ledger.

Work:

- inventory every field of `OmeZarrViewerApp`, `RootApp`, and `MosaicViewerApp`;
- identify its canonical writer, projection source, renderer consumers, persistence consumers, and
  native interaction commit point;
- mark each field `retain`, `narrow`, `replace`, or `delete`;
- identify every production and `cfg(test)` method that directly mutates a renderer projection;
- distinguish harmless snapshot/telemetry reads from semantic snapshot assembly;
- add structural tests forbidding new unclassified semantic fields and production command handlers
  under `renderer_bridge`; and
- convert the ownership inventory into a checklist whose rows close only when code and tests land.

Exit criteria:

- every compatibility field and mutator has a named disposition and milestone;
- no remaining item is described only as “Wave 7 cleanup”;
- the ledger distinguishes actor projection mirrors from truly renderer-owned state; and
- source guards fail when a new renderer semantic authority is introduced.

## Milestone 2 — retire renderer command emulators

Purpose: ensure tests exercise the production actor boundary instead of preserving an obsolete
renderer-side control implementation.

Status: complete on 2026-08-23. The renderer semantic-emulator inventory fell from 55 methods to
zero. Application tests now use an in-process `AppModel` command boundary followed by renderer
projection, and `renderer_has_no_semantic_command_emulators` rejects any reintroduction of the
recognized mutation families under `renderer_bridge`.

First target: `src/app/renderer_bridge/viewports.rs`, whose large direct-control implementation is
compiled only for tests.

Work:

- introduce reusable test fixtures that submit typed commands to a real in-process actor;
- consume the resulting projection into a lightweight renderer fixture where renderer behavior is
  under test;
- move pure parsing and validation assertions to model/command tests;
- move viewport, channel, layer, mask, object, and project semantic assertions to actor tests;
- retain renderer tests only for projection application, transient interaction, clipping, resource
  installation, and presentation acknowledgement;
- delete test-only `control_create_*`, `control_set_*`, `control_remove_*`, and equivalent semantic
  emulators as their actor fixtures reach parity; and
- add a structural guard preventing new direct semantic emulators under `renderer_bridge`.

Exit criteria:

- `renderer_bridge` contains observation, projection, readiness, and presentation code only;
- tests no longer require a second implementation of application commands;
- actor response and revision assertions cover all removed emulators; and
- all affected app and actor suites pass.

## Milestone 3 — remove viewport and presentation mirrors

Purpose: make the actor model the sole semantic source for the most frequently changing state.

Status: complete on 2026-08-23. Every Milestone 3 ownership row is now an explicit retained actor
projection, shared resource, renderer cache, or transient UI draft; no mixed compatibility row or
open disposition remains. The last RootApp scale-bar mirror was deleted: the native menu submits a
typed command and its checkmark is updated only from the active actor viewport projection. The
persisted-workspace decoder is named for its actual actor-side role and is no longer described as
a renderer restore path. Source guards enforce the closed ledger, command-only native commits,
resource-only renderer observations, and projection-only renderer boundary.

Current progress: native viewport chrome always emits typed topology commands, including before
the first actor projection, and has no renderer-mutation fallback for clone, remove, activate,
rename, layout, ratio, swap, or links. Periodic renderer feedback now uses a renderer-only
observation payload; actor-owned workspace, navigation, channel, layer, panel, and mask semantics
are absent from that path. Renderer frames no longer advance navigation/presentation revisions or
propagate linked camera/plane state, and native camera-fit and plane controls have no direct
semantic fallback. Project compatibility restoration and the mutable per-frame viewport state
container remain to be narrowed before the first ownership slice is closed. Production dataset
bootstrap no longer accepts a renderer workspace: the project snapshot is installed first, and the
actor restores the matching persisted ROI view into its canonical workspace before projecting it.
Persisted project masks are installed in the same actor transaction. Production single-view
project attachment and renderer-side dataset switching no longer call the legacy in-process view
restorer; that routine is now compiled only for persistence characterization tests. Remaining
annotation and other heavyweight project resources belong to the shared-resource work in
Milestone 5. The combined viewport value now contains an explicit nested `ViewportRenderState` for
canvas geometry, render generations, prior plane selections, level history, fallback ceilings, and
zoom-out retention. This makes those renderer-only fields mechanically distinct from the actor
projection fields while the two are still carried by one workspace container. Channel selection
drafts, slice drafts, overlay multi-selection, and object-filter caches are likewise isolated in
`ViewportTransientState`, rather than appearing as projected semantic fields. Native channel
visibility, ordering, active stepping, contrast, RGB presets, and channel-group commits likewise
queue actor commands without a projection-readiness fallback; channel projection fields remain in
the renderer only as the current paint/UI input. Frame planning and renderer queries now write back
only camera preview, render history, and transient state through `capture_runtime`; they cannot
recapture channel, layer, object, panel, or other projected semantics. Native-layer topology is
derived explicitly from each actor projection, rather than relying on a prior renderer capture.
Generic native-layer activation, visibility, ordering, offsets, transforms, and property
transactions now obey the same rule. Actor-owned panel toggles, right-tab selection, viewport
rendering preferences, and the native scale-bar menu also remain unchanged locally until their
projection arrives. Left-tab selection now follows the same actor-owned command/projection path,
so all single-view panel settings have crossed this boundary. Mosaic native channel, layer, focus,
layout, camera, panel, tab, object-rendering, and display-preference commits now follow the same
rule: they enqueue typed commands and wait for actor projection, with no projection-readiness or
`control_actor_owned` mutation fallback. Mosaic startup no longer serializes renderer semantics
into `BootstrapMosaic`; RootApp sends the project snapshot first, then only the immutable mosaic
resource, and `MosaicModel` restores persisted mosaic state before publishing its projection. The
test-only mosaic channel/layout/focus command emulators and renderer semantic bootstrap assembler
have been deleted. Annotation ownership subsequently crossed in Milestone 4: the actor now
restores, loads, persists, and projects annotation layers and their shared data resources.

Ownership slices, in order:

1. workspace membership, titles, active viewport, layout, ratio, and links;
2. camera, logical geometry, plane mode, and slice coordinates;
3. channel visibility, active channel, order, groups, colors, contrast, and transforms;
4. user-visible rendering preferences, side panels, and tabs; and
5. per-viewport layer visibility, order, offsets, scale, and rotation.

For each slice:

- prove native and protocol commands produce equivalent actor state and events;
- replace renderer semantic copies with a projection DTO or consumed projection generation;
- retain optimistic local state only for active gestures, tagged with its starting revision;
- prevent stale projections from overwriting a newer local gesture preview;
- submit exactly one typed command at the gesture commit point;
- ensure state queries read the actor rather than reassembling semantic truth from the renderer; and
- delete the obsolete fields and helper methods before beginning the next slice.

Exit criteria:

- viewport and channel queries are authoritative actor queries;
- pausing frames does not affect viewport, camera, plane, channel, or layer command completion;
- linked viewport behavior remains transactional and revision ordered;
- renderer state contains only presentation caches, measured geometry, and transient previews; and
- the two-viewer comparison passes no-frame and restored-projection tests.

## Milestone 4 — remove object, mask, annotation, and selection mirrors

Purpose: establish one semantic owner while preserving renderer-efficient large resources.

Status: complete on 2026-08-23. Native primary and secondary object selection commits now enter targeted typed
actor commands with no renderer-selection fallback. Object analysis editors restore their local
preview after each UI transaction and submit the committed configuration through
`viewer.analysis.set`. Analysis configuration, warmup state, and generations are now target-scoped
in the actor and projected independently for the primary segmentation and each secondary spatial
shape layer. Mask layer identity, polygons, selection,
undo, import, project synchronization, append, and export are actor-only. The renderer retains the
latest projected mask layers plus generation-tagged drawing and move previews; the former local
mask ID allocator, undo stack, project-dirty mirror, filesystem loader/writer, and readiness-based
fallbacks have been deleted. Native mask export now selects a path in the platform UI and performs
the actual I/O on the actor worker service. Structural guards reject reintroduction of these object
selection and mask fallback paths. Annotation layer identity, presentation, source configuration,
readiness, persistence, schema inspection, and Parquet resource installation are now actor-owned
in both single-image and mosaic modes. Native UI, Python, and MCP use the same nine
`viewer.annotations.*` methods. Parquet work runs on the bounded actor worker service, stale
source/document generations are rejected, and immutable datasets cross into the renderer through
shared generation-tagged resources. The renderer retains only paint/UI adapters, GL state, and
pending widget intent; project save paths preserve the actor projection rather than reassembling
annotation truth from those adapters.

Mosaic ROI focus and selection are confirmed projection-only. Mosaic object appearance, legend
configuration, and per-ROI object selections now also commit through typed actor methods; native
hit testing computes a proposed selection without mutating the renderer, and the renderer installs
the resulting actor projection. The final resource adapters are now actor-owned. Single-view
segmentation GeoJSON source identity, readiness, parsing, and immutable polylines live in the actor
and bounded worker service; the renderer retains only line bins and GPU state. Mosaic visible-item
discovery submits explicit item IDs through `mosaic.objects.load`, after which the task progresses
without frames. The former renderer GeoJSON thread, mosaic frame-driven object loader, and
transformed-object loader entry point have been deleted. Every Milestone 4 ownership-ledger row is
now retained and closed.

Work:

- make object presentation, filters, selections, legends, and analysis configuration actor-only;
- retain immutable geometry/property indexes as shared generation-tagged resources;
- make mask layers, polygon identity, selection identity, and undo history actor-only;
- keep in-progress drawing and vertex/move previews transient in the renderer;
- make committed annotation and mask edits actor transactions;
- narrow renderer state to GPU caches, spatial indexes, hit-test caches, and consumed generations;
- verify stale worker results and stale edit commits cannot replace a newer resource generation; and
- verify primary and secondary object targets remain independently addressable.

Exit criteria:

- Python and native edit replay produce equivalent state, events, and persistence output;
- no object, mask, annotation, or selection semantic snapshot is reconstructed from renderer state;
- large resources are not copied on camera or presentation changes;
- no-frame filter, selection, mask I/O, and analysis tests pass; and
- transient drawing and selection behavior remains responsive.

## Milestone 5 — remove project, resource, compute, and application mirrors

Purpose: finish the less frame-sensitive but persistence-critical ownership domains.

Status: complete on 2026-08-23. Histogram/automatic contrast, unified memory pinning,
project/legacy analysis UI, and SpatialData/Xenium resources now have explicit final ownership.
Earlier slices checkpointed settings and mosaic project projections, project object preload,
remote datasets, labels, TIFF planes/tile policy, and threshold analysis. No Milestone 5 ownership
row remains open.

Work:

- narrow project, ROI, saved-view, recent-project, settings, and lifecycle UI models to drafts and
  actor projections;
- remove UI-side auto-loaders superseded by actor-owned retained tasks;
- narrow document, TIFF, SpatialData, Xenium, label, and remote-session UI state to shared resource
  handles, renderer adapters, and dialog drafts;
- remove duplicate memory, threshold, measurement, export, and analysis operation state;
- ensure project save/autosave consumes an immutable actor snapshot at a known revision;
- ensure secrets remain actor-private and projections remain redacted;
- preserve atomic project/deep-link/ROI-open transactions and stale-result rejection; and
- test restart and persistence round trips after each durable-state slice.

Exit criteria:

- application and project queries do not depend on renderer snapshots;
- retained task state is authoritative and observable through the actor;
- project save/load and settings restart tests reproduce actor state;
- all I/O and compute continues without frames on bounded workers; and
- UI structs retain only drafts, status presentation, shared handles, and renderer resources.

### Remaining Milestone 5 slices

Complete these slices in order. Each slice is a separate reviewable commit and closes its ledger
row before the next begins.

#### 5A — image histogram and automatic contrast — complete

Closed row: `viewer.histogram_compute` (12 fields removed or narrowed into the retained
`viewer.histogram_plot_cache` renderer observation row).

- replace the renderer-owned histogram and channel-maximum worker handles with bounded actor work;
- tag every request and result with document, channel, level, region, and operation generations;
- publish immutable intensity/histogram results for plotting without making the plot cache
  authoritative;
- apply automatic contrast in the actor transaction, including automatic contrast requested during
  document opening, so it does not require renderer construction or a future frame;
- make native histogram refresh and automatic-contrast controls submit the same typed command used
  by Python and MCP;
- delete frame-polled histogram/channel-max queues and their local filesystem workers; and
- prove with a no-frame test that a document open followed by automatic contrast reaches its final
  model and projection state while the renderer is paused.

Commit gate: no `hist_loader`, `chanmax_loader`, `spawn_histogram_loader`, or
`spawn_channel_max_loader` remains in the application host; stale intensity results cannot change a
newer document/channel; actor, renderer-projection, and source-organization tests pass.

Checkpoint evidence: on 2026-08-23 the no-frame histogram, explicit automatic-contrast,
on-open automatic-contrast, stale manual-window, stale document-generation, alternate-reader, and
source-organization tests passed. The cumulative Rust library (180), binary (202 passed, 4 ignored
extended fixtures), data-contract (10), and Python SDK (96) suites also passed, as did formatting,
all-target compilation, generated-reference, JSON, application-surface, registry, and ownership
ledger checks.

#### 5B — memory and pinned-level operations — complete

Closed rows: `viewer.memory` (7 fields) and `mosaic.memory` (3 fields), split into retained actor
projection/shared-resource observations, transient confirmation/channel drafts, renderer OS-memory
observations, and the existing projected tile-policy row.

- make pin/unpin/load policy and retained-task status actor-authoritative in both modes;
- retain selected-channel and confirmation-dialog values only as transient UI drafts;
- retain OS memory sampling only as a renderer observation unless a public query requires an actor
  snapshot, and name it accordingly;
- consume pinned levels and tile-policy preferences only from actor projections/shared resources;
- delete readiness-dependent local pin/load fallbacks; and
- test cancellation, supersession, queue bounds, stale completion, and completion with no frames.

Commit gate: no renderer code can start semantic memory loading directly; viewer and mosaic expose
equivalent task state and policy behavior through the actor; both ledger rows are closed.

Checkpoint evidence: on 2026-08-23 the single-view and mosaic no-frame pin workflows, cancellation,
single-view and mosaic stale-result rejection, immutable projection reuse, native-routing source
guard, and exact ownership-ledger coverage passed. The cumulative Rust library (183) and binary
(203 passed, 4 ignored extended fixtures), data-contract (10), and Python SDK (96) suites and
all-target compilation also passed.

#### 5C — project and legacy analysis UI — complete

Closed row: `viewer.project_ui` (4 fields), split into explicit actor projection, renderer
generation observation, ROI UI draft, and renderer-only compatibility-resource rows.

- split `project_space` into an immutable actor projection plus renderer-only panel/draft state, or
  remove the viewer copy where `RootApp` can provide the projection directly;
- rename `project_cfg_seen` to `control_actor_project_config_generation` and classify it as a
  renderer generation observation rather than configuration state;
- replace `roi_selector` semantic behavior with typed project/ROI commands while retaining only its
  selection and validation drafts;
- narrow the legacy `cell_thresholds` object to the renderer-only
  `legacy_cell_threshold_points` compatibility adapter; it has no command or persistence surface,
  while public threshold semantics remain actor-owned; and
- retire production-unreachable project restore/auto-load helpers once characterization tests have
  been moved to the actor model.

Commit gate: project save/open/ROI workflows never reconstruct state from a viewer panel; persistence
round trips use a known actor revision; the project UI row is closed.

#### 5D — SpatialData and Xenium renderer adapters — complete

Closed row: `viewer.external_spatial_layer_adapters` (3 fields), split into retained image,
shape/point, and Xenium renderer/shared-resource adapter rows.

- audit all production construction, tick, retry, and load entry points for image, shape, point,
  cell, and transcript adapters;
- move any remaining metadata/data decoding and retry lifecycle to bounded actor workers;
- cross the boundary using immutable generation-tagged resources and leave only GPU caches, spatial
  indexes, hit testing, tables, and paint adapters in the renderer;
- delete unreachable legacy adapter loaders instead of wrapping them in smaller modules; and
- verify a superseded document or layer generation cannot install a late spatial result.

Commit gate: renderer adapter `tick` methods perform presentation/resource consumption only and no
filesystem/network/decoder work; no-frame SpatialData and Xenium resource tests pass; the final
Milestone 5 row is closed.

Checkpoint evidence: the bounded alternate-document worker prepares raw SpatialData shapes,
sampled points, object resources, secondary images, Xenium cell bins, and Xenium transcripts before
the document becomes resource-ready. The renderer installs immutable payloads and retains only
tile/GPU caches, presentation properties, spatial indexes, hit testing, and feature-selection
drafts. The legacy SpatialData point thread and UI-triggered Xenium transcript reload worker are
deleted; source guards reject their return.

Checkpoint evidence: project save now captures the canonical actor workspace immediately before
building its immutable persistence payload, and a no-frame round trip proves a camera edit is saved
without renderer synchronization. Production panel drawing, deep-link application, and
`take_project_space` no longer reconstruct project state from the renderer. The legacy encoder and
restorer are test-only, ROI actions enter typed commands, and source guards reject renderer reverse
synchronization.

#### 5E — Milestone 5 cumulative gate — complete

- run formatting, all-target compilation, Rust library/bin/data-contract suites, Python SDK tests,
  registry/application-surface audits, generated-reference checks, and the ownership-ledger guard;
- add restart/persistence coverage for any durable state changed in 5A–5D;
- update the ownership inventory counts and milestone narrative to the exact tested commit; and
- record task-queue and projection-allocation regressions before declaring Milestone 5 complete.

Milestone 6 may begin only when the current Milestone 5 row is closed and this cumulative
gate passes.

Checkpoint evidence: formatting and all-target compilation passed; Rust library tests passed
183/183, binary tests passed 205 with 4 fixture-dependent ignores, data-contract tests passed
10/10, and Python SDK tests passed 96/96. Registry, application-surface, ownership-ledger, JSON,
generated Python reference, no-frame, queue-bound, projection-reuse, and local command-latency
checks passed on the same tree.

## Milestone 6 — narrow the application shell and platform boundary

Purpose: leave `RootApp` and the app façade as orchestration boundaries, not alternate model owners.

Work:

- audit all remaining `pending_request`, outbox, snapshot, and synchronization paths;
- remove semantic relay types that are no longer needed;
- keep a narrowly named platform-effect path for focus, close, quit, native dialogs, clipboard, and
  rendered-pixel capture;
- ensure actor construction remains mandatory and precedes optional TCP publication;
- ensure actor failure is fatal rather than selecting a frame-driven fallback;
- make update methods orchestrate projection consumption, input, rendering, and platform effects
  only; and
- reduce façade sizes only where responsibility extraction or ownership deletion makes the new
  boundary clearer.

Exit criteria:

- pausing `RootApp::update` cannot prevent a non-presentation application command from completing;
- `RootApp` cannot directly mutate canonical application state;
- no semantic command is represented as a platform request; and
- application source guards enforce the final boundary.

### Milestone 6 execution slices

#### 6A — classify and narrow host effects

Open rows: `viewer.host_requests`, `mosaic.status_host`, and `root.deep_link_input`.

- enumerate every remaining host request and classify it as a typed semantic command, transient
  dialog action, or unavoidable platform effect;
- move any semantic variant to the command registry and delete its renderer/root relay;
- retain focus, close, quit, native file/folder dialogs, clipboard, URL receipt, and rendered-pixel
  capture behind narrow effect types; and
- make status values either actor task/projection state or explicitly local presentation status.

#### 6B — remove shell projection and mode authority

Open rows: `root.mode`, `root.deferred_projection`, and `mosaic.actor_projection_state`.

- reduce root mode to renderer composition and resource attachment selected by the latest actor
  projection;
- replace deferred semantic projection assembly with latest-generation projection coalescing;
- ensure switching single/project/mosaic presentation cannot create or overwrite actor state; and
- retain only renderer-consumed generation observations in mosaic mode.

#### 6C — narrow native command transport

Open rows: `viewer.native_command_outbox`, `root.native_command_outbox`, and
`mosaic.native_command_outbox`.

- choose one bounded native command ingress owned by the application shell;
- remove per-renderer relays where native controls can submit directly to that ingress;
- preserve ordering and explicit backpressure without making `RootApp::update` responsible for
  semantic execution; and
- test equivalence of native, Python, MCP, menu, startup, and deep-link entry points.

Commit gate for Milestone 6: all nine current rows are closed, source guards reject semantic host
requests and renderer outboxes, and pausing `RootApp::update` cannot block any non-presentation
method.

## Milestone 7 — close completion and presentation semantics

Purpose: make Python waiting behavior precise for both background and pixel-dependent operations.

Work:

- audit every synchronous, asynchronous, retained-task, and presentation method's completion point;
- ensure model/resource completion never waits for an arbitrary repaint;
- expose the exact readiness or presentation phase through task state;
- ensure presentation tasks identify the desired model revision and required resource generations;
- keep the actor mailbox responsive while a presentation task waits;
- test timeout, cancellation, overwrite safety, partial-output cleanup, and stale acknowledgement;
- verify Python synchronous waits block only the calling Python thread;
- verify Python asynchronous waits yield to the Python event loop; and
- synchronize the protocol and Python documentation with the implemented semantics.

Exit criteria:

- every public long-running method has documented completion and cancellation behavior;
- a covered/minimized screenshot reports `waiting_for_presentation` rather than model failure;
- unrelated commands complete while that screenshot waits;
- returning to Odon completes the generation-matched capture; and
- Python sync and async tests cover these distinctions.

### Milestone 7 execution slices

#### 7A — one actor-owned presentation-task model

Open rows: `viewer.screenshots` and `mosaic.screenshots` (13 fields total).

- move screenshot settings, task identity, output lifecycle, cancellation, and error state into a
  shared actor presentation-task model;
- retain only the renderer capture adapter and a generation-specific in-flight acknowledgement at
  the UI boundary;
- use one bounded encoder/writer service for single and mosaic modes;
- clean partial outputs on failure/cancellation and reject stale capture acknowledgements; and
- publish `waiting_for_presentation` with the exact model revision and resource generations when no
  eligible frame exists.

#### 7B — public completion-contract audit

- classify every registered method as immediate semantic, resource-ready, retained background task,
  or presentation-dependent;
- verify its response/task completion point matches that classification;
- add table-driven sync/async Python tests for success, timeout, cancellation, and unrelated-command
  responsiveness; and
- regenerate protocol/Python documentation from the audited registry.

Commit gate for Milestone 7: both screenshot rows are closed, the ledger has no open field, and the
completion contract is generated, tested, and identical across Python and MCP transports.

## Milestone 8 — final verification and sign-off

Purpose: convert the completed design into release-quality evidence.

Automated checks:

```text
cargo fmt --all -- --check
cargo check --all-targets
cargo test --lib
cargo test --bin odon
cargo test --test data_contracts
Python SDK unit suite
Python generated-reference check
application-surface and registry audits
latency, projection-allocation, bounded-queue, and frame-planning checks
```

Native acceptance matrix:

| Platform condition | Semantic/resource work completes in condition | Final projection appears without another command | Presentation wait/cancel verified |
| --- | --- | --- | --- |
| macOS visible | Required | Required | Required |
| macOS fully covered | Required | Required | Required |
| macOS minimized | Required | Required | Required |
| macOS separate Space | Required | Required | Required |
| Windows covered/minimized | Required if supported; otherwise explicit exclusion | Same | Same |
| Linux covered/minimized | Required if supported; otherwise explicit exclusion | Same | Same |

Final documentation work:

- close every row in the ownership ledger;
- update the architecture plan from “Wave 7 in progress” to complete;
- record exact test counts, platform/build information, performance results, and exclusions;
- regenerate the Python API reference;
- document the semantic-versus-presentation completion contract in the protocol reference; and
- archive the verifier JSON and human visual sign-off without modifying generated evidence.

Exit criteria:

- Gates A through D are all satisfied;
- no unresolved compatibility item remains in the ownership inventory;
- the final cumulative suite passes on the exact release candidate;
- real-window evidence covers every supported required condition; and
- the working tree is clean after the completion commits.

## Recommended commit sequence

Use small ownership-complete commits rather than one final cleanup commit:

1. stabilize and checkpoint current actor migration and modularization;
2. add the executable ownership ledger and guards;
3. replace viewport renderer emulators with actor fixtures;
4. remove viewport/workspace/camera/plane mirrors;
5. remove channel/rendering/layer mirrors;
6. remove object/mask/annotation/selection mirrors;
7. remove project/resource/compute/application mirrors;
8. narrow `RootApp` and platform effects;
9. close presentation/completion semantics and documentation; and
10. add final acceptance evidence and release sign-off.

Each commit must leave the relevant suites green and must delete the superseded compatibility path
for its ownership slice. A commit that only relocates an obsolete semantic implementation does not
advance this plan.

## Current execution sequence

From the present checkpoint, work proceeds in this order:

1. **5A histogram/auto-contrast — complete:** image analysis and on-open contrast are actor-driven,
   and the two local worker services are deleted.
2. **5B memory — complete:** viewer and mosaic native controls now share actor-owned bounded work,
   lifecycle projection, cancellation, supersession, and immutable resources.
3. **5C project UI — complete:** project persistence captures the actor workspace at a known
   generation, reverse renderer synchronization is absent from production, and panel/resource
   fields have explicit final roles.
4. **5D external spatial adapters — complete:** document decoders run on the bounded actor worker,
   immutable resources feed renderer adapters, and the 5E cumulative gate passed.
5. **6A–6C application shell — next:** classify platform effects, remove shell semantic authority, and
   consolidate native command ingress; then checkpoint Milestone 6.
6. **7A presentation tasks:** unify screenshot state and generation-specific acknowledgement.
7. **7B completion audit:** test and document every method's exact sync/async completion contract;
   then checkpoint Milestone 7 with zero open ledger fields.
8. **Milestone 8 automation:** run all Rust, Python, generated-contract, performance, allocation,
   queue-bound, and frame-planning checks on the exact release candidate.
9. **Milestone 8 native acceptance:** run and archive visible, covered, minimized, and separate-Space
   evidence on macOS; run supported Windows/Linux covered/minimized checks or record reviewed
   exclusions; restore the window and visually confirm the latest projection.
10. Synchronize all architecture/status documents, record exact counts/build identifiers/results,
    and make the final sign-off commit with no unexplained tracked or untracked output.

No later step may be used to declare an earlier ownership gate complete. In particular, passing
the no-frame API suite does not prove that renderer semantic mirrors have been removed, and a clean
ownership audit does not replace real covered/minimized window evidence.

### Definition of done for each implementation commit

Every slice commit must:

- remove or narrow the superseded path in the same change as its replacement;
- update the executable ledger and add a structural guard where regression is mechanically
  detectable;
- include actor-model behavior tests, stale-generation tests for worker work, and renderer tests
  only for projection/resource consumption or transient UI behavior;
- pass formatting, all-target compilation, and the focused Rust tests for the touched domains;
- leave unrelated user files and local acceptance artifacts untouched; and
- state which ownership rows and field count were closed.

At milestone boundaries, run the cumulative suites instead of relying on the focused commit checks.
Do not use lower line count, a smaller `actor.rs`, or a smaller `app.rs` as completion evidence; the
relevant evidence is ownership, frame independence, bounded work, completion semantics, and the
recorded release matrix.

## Progress reporting

Report these figures separately after every milestone:

- registered route coverage and any mode/target variants;
- number of ownership-ledger items retained, narrowed, replaced, deleted, and still open;
- number of renderer semantic emulators remaining;
- no-frame test coverage by method family;
- native acceptance conditions passed, provisional, excluded, or not run;
- cumulative Rust and Python test results; and
- performance results and regressions.

Do not use reduced file size, total actor line count, or the number of modules as a proxy for
architectural completion.
