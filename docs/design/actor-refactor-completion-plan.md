# Control Actor Refactor Completion Plan

Status: in progress — Milestones 0 through 2 complete; Milestone 3 topology ownership in progress

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

- 263 of 263 registered application methods are actor-owned in every supported mode and target;
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

The stabilized migration base, executable ownership ledger, and renderer-emulator retirement are
now separately checkpointed milestones. The next work is removal of viewport and presentation
mirrors, starting with workspace topology and navigation state.

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

Current progress: native viewport chrome always emits typed topology commands, including before
the first actor projection, and has no renderer-mutation fallback for clone, remove, activate,
rename, layout, ratio, swap, or links. Periodic renderer feedback now uses a renderer-only
observation payload; actor-owned workspace, navigation, channel, layer, panel, and mask semantics
are absent from that path. Renderer frames no longer advance navigation/presentation revisions or
propagate linked camera/plane state, and native camera-fit and plane controls have no direct
semantic fallback. Project compatibility restoration and the mutable per-frame viewport state
container remain to be narrowed before the first ownership slice is closed.

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
