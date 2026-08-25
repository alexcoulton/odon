# Python-controlled application shell: remaining work

Status: implementation in progress, audited 2026-08-25

## Short answer: what remains

The actor-controlled, recursive single-window layout is working. Python can already discover
components, replace and patch the tree, mount extensions, round-trip native split/tab/collapse/focus
interactions, import and export versioned layouts, and save named session/application/project
profiles.

The remaining work falls into three completion levels:

1. **Maintain the completed persistence baseline.** Add compatibility migrations only when a
   future schema requires them; checked-in fixtures now preserve every currently supported shape
   and its failure/recovery behavior.
2. **Complete control of the existing native window.** The legacy extension hosts and shell
   inspector are actor-tree mounts, and the macOS platform menu now follows an actor-owned
   declarative command presentation. Complete cross-platform shortcut realization and
   visual/interaction/performance tests.
3. **Qualify declarative behavior within that window.** Actor-resource bindings over the
   contextual command-state model are implemented without Python frame callbacks; capture their
   rendered behavior in the cross-platform release suite.

Multi-window work—including docking, floating regions, detached native windows, monitor placement,
and moving canvases between windows—is explicitly out of scope for this initiative. It is a
significant architectural change requiring its own design review and proposal, not a condition of
completing Python control of Odon's existing application shell.

### Latest implementation checkpoint

Extension-owned default templates are now wired through
`ui.extensions.layouts.register/list/remove` and typed synchronous/asynchronous Python handles.
Registration normalizes compatible documents to v1, validates extension-mount ownership, requires
`ui.panels`, enforces a 64-template quota, and follows the extension's remove/disable/retain
disconnect lifecycle. Applying a template remains an explicit revision-guarded shell import.
Application settings can select one startup application profile per mode. The actor restores each
mode once on first activation and installs the protected recovery layout if the configured profile
is absent, malformed, incompatible, or targets another mode. Profile listings and
`app.settings.get` expose the exact validation/recovery diagnostics.

Boundary stress tests now validate a complete 256-node desired tree and reject node 257, accept a
128-item toolbar and reject item 129, exercise the accepted predicate node/depth limits, and
reconcile 512 rapidly changing selection projections. Local timing gates cover those maximum
inputs and sustained slow-subscriber pressure; results and reproduction commands are recorded in
[Python-controlled shell performance evidence](python-shell-performance.md). Cross-platform
rendering, GPU timing, and platform TCP pressure remain release work.

Shell snapshots now carry actor-owned `active_region_id` and `focused_node_id`. Revision-guarded
patches set them or explicitly clear focus; native tab, collapse, leaf activation, and completed
split-drag interactions use the same command path. `ui.shell.changed` reports precise per-node
visibility, order, selection, size, split, collapse, activation, and focus changes. Split drag
frames remain renderer-local and only a changed final ratio is committed. Actor-direct and
transport mutations both enforce extension-node ownership, and ownership/conflict failures carry
actionable resolution metadata.

Disconnect cleanup now reconciles actor-owned focus after extension mounts become unavailable. If
the active or focused node is the disconnected contribution—or an application-owned default host
that the disconnect made empty—the bridge commits one revision-guarded native patch selecting the
mode's required workspace/canvas. The change uses the ordinary actor and event path rather than
mutating transport-local state.

The first dataset-specific shell capture now drives the checked-in five-channel OME-Zarr fixture
through `examples/python_shell_control.py`. On macOS it renders the 23-node review tree and mounted
four-command toolbar, including checked scale-bar and disabled mask-export state. Captured actor
output and paired frames prove the scale-bar command also hides its bound right inspector region
and releases that space to the canvas without changing the persisted layout revision or invoking a
Python frame callback; the example restores the prior shell and toolbar on exit.

`examples/python_extension_host_control.py` now qualifies an actor-owned default extension host
with a separate Python extension session. The checked-in macOS sequence renders its panel in the
left-sections host, retains the same mount with an explicit disconnected state when that session
closes, and reclaims it from a compatible replacement session without changing shell revision 3.
The capture found a passive compatibility bug that replaced the desired extension tab with the
legacy Layers tab. Passive legacy projection no longer rewrites the declarative tree; explicit
legacy tab commands still update it, with a regression test covering both behaviors.

`examples/python_shell_interaction_control.py` now supplies a third repeatable macOS qualification
workflow. Four checked-in frames and actor output cover the 23-node baseline at split ratio 0.24,
an atomic resize to 0.36 with Project selected and focused, collapse with focus transferred to the
canvas, and protected recovery invoked through the shared command path. The state advances through
revisions 2 to 5 and the final tree is the two-node recovery canvas. Python patches make the
capture deterministic; native interaction tests prove the corresponding gestures use the same
actor patch and semantic-event path.

`examples/python_shell_startup_mode_control.py` now proves startup restore across a real isolated
process restart. A setup process saves and selects a 23-node application profile; the evidence
process restores it on first single-view activation, transitions to the default project tree, and
returns to the same single-view tree with Project active/focused at revision 3. Three checked-in
macOS frames accompany exact restore diagnostics. The new `ODON_SETTINGS_PATH` launch override
keeps the two-process proof separate from normal user settings and rejects an empty override.

Project, single-viewer, and mosaic built-in top bars now render as ordinary desired-tree mounts.
The application root gives menu, toolbar, content, and status hosts vertical placement semantics
with intrinsic chrome heights. Reordering or hiding the top-bar node therefore changes the actual
window, while the old `TopBottomPanel` path exists only when no desired-layout projection is
available.

Application commands are now actor-owned descriptors independent of presentation. A separate
revisioned, bounded menu tree references those commands through stable IDs; Python has typed sync
and async builders for nested menus, command items, and separators. `ui.menus.replace` requires the
chrome grant, preserves protected close/quit/recovery presentations, correlates changes, and emits
`ui.menus.changed`. macOS rebuilds its actual native menu from the projection rather than a
hard-coded action list.

Extensions that declare `ui.actions` now register and remove namespaced event commands through
typed synchronous/asynchronous Python handles. The actor enforces extension ownership,
`ui.shell.shortcuts`, revision guards, and overlapping-mode shortcut conflicts. Native menu or
Python invocation publishes the declared extension event without a GUI-thread callback. Explicit
readiness, remove/disable/retain disconnect policy, compatible reconnect, and version mismatch all
reconcile the retained descriptor; native menu items follow application mode and command
readiness. The real TCP lifecycle test covers disconnect and reconnect of both a focused retained
mount and its retained command.

Menus, mounted toolbars, the native command palette, shortcuts, and Python now invoke commands
through `ui.commands.execute`. The actor resolves native, control-method, and extension-event
handlers, applies mode/readiness and capability checks, publishes `ui.commands.executed`, and emits
typed platform effects. Python can revision-guard and correlate palette replacement through typed
sync/async resources; Rust validates its title, prompt, shortcut, description visibility, and
bounded result count. The egui palette searches the shared descriptor catalogue and filters it by
active mode and readiness, so it does not introduce a second command registry.

Commands now carry optional bounded `visible`, `enabled`, and `checked` predicates. Extension
authors build typed capability, actor-state, boolean-composition, and negation predicates in Python;
Rust limits their depth/node count and restricts state access to a published resource, selection,
mode, GPU, panel, and scale-bar context. The actor evaluates one state record for requesting-session
command lists, native projections, and execution. Menus, toolbars, and the palette consume it;
direct execution rejects unavailable predicates with reasons and missing-capability data. Toolbar
items now support icon, tooltip, label-visibility, checked selection, and disabled-reason rendering.

Desired mount nodes also carry versioned `configuration` objects. `ui.shell.patch_layout` updates
them through its `configurations` map and emits a precise `configuration` change. The three native
top bars publish non-empty JSON Schemas for their optional control groups; Rust rejects unknown or
wrongly typed properties before committing and the mount dispatcher renders the accepted values.
Configuration retention is bounded per node and per tree, with depth, value-count, key, and string
limits exposed through `ui.shell.describe_schema`.

The registry no longer describes every shell mutation as `ui.shell.write`. Existing methods now
advertise `ui.shell.compose`, `ui.shell.extension_place`, `ui.shell.persistence`, or
`ui.shell.recovery`; `ui.shell.chrome`, `ui.shell.window_control`, and `ui.shell.shortcuts` are
reserved for their corresponding surfaces. Actor preflight confines extension sessions to their
own extension-mount nodes, including configuration and selection targets, and reports the precise
owner/capability needed for application or foreign-extension nodes.

`system.hello` now negotiates explicit requested/granted capabilities. Transport dispatch checks
the shell method boundary, actor preflight checks application/chrome/recovery/extension ownership,
and authentication alone grants no shell mutation authority. The Python client requests the full
current application-controller set by default but accepts a narrower set for extension or
read-only sessions.

The ownership, focus, extension-command, command-menu, toolbar, and semantic-interaction checkpoint
passes `cargo check --all-targets`, formatting, source-organization, generated-reference,
JSON-manifest, and diff checks. The complete regression run records 493 passing Rust tests
(4 ignored) and 131 passing Python tests.

## Target

Python should be able to describe, inspect, and update the complete Odon application shell. Rust
should continue to own egui, the event loop, rendering and GPU resources, platform integration,
validation, bounded queues, and recovery UI.

The important boundary is declarative control: Python says what should exist and how it should be
arranged; Rust safely realizes that desired state.

## What is implemented now

The actor-owned shell foundation is in place:

- `ui.shell.get`, `ui.shell.patch`, `ui.shell.patch_layout`, `ui.shell.replace_layout`, and
  `ui.shell.reset` expose revision-guarded, atomic shell mutations.
- `ui.shell.describe_schema` publishes formal JSON Schemas and
  `ui.shell.components.list` publishes typed built-in component descriptors.
- Matching synchronous and asynchronous Python APIs expose typed snapshots, component descriptors,
  layout nodes, sizes, splits, changes, and builders.
- A desired tree can contain keyed applications, rows, columns, splits, tabs, panels,
  collapsibles, toolbar/status/menu hosts, canvas slots, built-in mounts, and extension mounts.
- Validation covers topology, reachability, cycles, identity, node/depth quotas, child cardinality,
  tab selection, size values, split ratios, mode-specific mounts, singleton built-ins, and a usable
  required canvas/workspace.
- Complete tree replacement and interactive layout patches are atomic. Interactive patches cover
  visibility, tab selection, size, split state, collapse state, active region, and focus.
- Versioned layout documents support deterministic export, atomic import, v0-to-v1 migration, and
  rejection of unsupported future versions without changing actor state.
- Named layouts can be retained in actor-session scope, persisted atomically in application
  settings, or stored in canonical project state. A protected minimal recovery layout is
  available for every mode.
- Stable node keys retain renderer-local state across projections.
- Single-viewer, mosaic, and project content use a recursive renderer for nested rows, columns,
  splits, tabs, panels, collapsibles, canvases, built-in mounts, and extension mounts.
- Built-in project, viewer, and mosaic top bars use that same mount dispatcher. Application roots
  stack declarative menu/toolbar/content/status children vertically with intrinsic chrome sizing.
- Desired mount nodes carry atomically validated per-instance configuration. Native top bars
  publish and honor discoverable non-empty schemas; Python exposes the configuration on typed
  layout nodes and revision-guarded patches.
- Desired, minimum, maximum, and flex sizing drive geometry. Nested split handles, tab selection,
  and collapse controls commit through `ui.shell.patch_layout`.
- Active region and optional focus live in the actor snapshot. Native interaction commits and
  Python patches produce the same precise `ui.shell.changed` property changes; no-op split frames
  are suppressed and split ratios commit when dragging stops.
- Built-in validation is driven from the published component catalogue, including modes, legal
  parents, singleton rules, component kind, and intrinsic minimum size.
- Registered extension contributions receive stable shell mount IDs, appear in the component
  catalogue, render beside built-ins, preserve native bindings, and support safe
  disconnect/reconnect/missing placeholders.
- Version-1 visibility, order, and selection operations remain compatible with the desired tree.

The complete verification pass on 2026-08-25 records 493 passing Rust tests (4 ignored) and all
131 Python tests passing.

## The central remaining limitation

The recursive single-window shell and its actor-owned command surfaces are functionally in place.
The principal remaining gap is release evidence across the supported desktop platforms, not a
second competing shell architecture.

In particular:

- extension default placements are actor-tree mounts, while the extension diagnostics window
  remains a protected native recovery/operability surface;
- the macOS native menu, extension command lifecycle, rich toolbar model, native searchable palette,
  contextual predicate evaluation, built-in/extension dispatch, and Windows/Linux descriptor
  shortcuts are declarative, while cross-platform rendered evidence remains incomplete;
- actor-direct and transport mutations enforce explicit session grants and extension-node
  ownership; protocol v1 deliberately keeps the existing bounded `system.batch` non-atomic until a
  concrete cross-resource invariant requires another transaction primitive.

Cross-platform qualification of the completed single-window control path is the next release seam.

## Remaining work by priority

### P0 — make the declared single-window tree real

#### 1. Add a recursive layout renderer — substantially complete

Implemented on 2026-08-24 for project, single-viewer, and mosaic content trees. Stable keyed
geometry, nested rows/columns/splits, tabs, panels, collapsibles, canvases, built-ins, extension
mounts, sizing constraints, flex allocation, and safe canvas fallback are active.

Required work:

- repeat the macOS ready/disconnected/reconnected left-sections host evidence for representative
  remaining host locations on Windows and Linux;
- add cross-platform screenshot coverage for deeply nested representative trees.

Acceptance criteria:

- two trees with different nesting but the same mount order produce visibly different layouts;
- a nested split can be resized and its committed ratio or size is visible through
  `ui.shell.get`;
- untouched keyed nodes retain selection, scroll, and renderer-local state after reconciliation;
- minimum and maximum constraints are enforced during native resizing; and
- invalid or impossible geometry falls back safely without corrupting actor state.

#### 2. Complete native interaction round-tripping — complete

Tab selection, split ratios, collapse, activation, and focus now round-trip in one actor patch.
Renderer-local interaction state is keyed by node ID and revision. Split drags commit on release
and unchanged results are suppressed. Events expose precise per-node property changes.

End-to-end TCP lifecycle coverage now focuses a retained extension mount, disconnects its owning
session, observes focus transfer to the required workspace, reconnects the compatible extension,
and observes its retained mount return to `ready`. Mode-transition coverage proves each mode keeps
its own active/focused state without cross-mode leakage. A revision-race stress test commits 48
rapid native gestures, races 12 clients from one shell revision, verifies exactly one guarded
winner and actionable conflicts, and completes a refetch/retry.

#### 3. Finish built-in component composition — substantially complete

Catalogue-driven validation, shared viewer/mosaic dispatchers, intrinsic sizing, recursive project
workspace and top-bar mounting, and unavailable-component placeholders are implemented. Channels,
single-view viewport controls, the complete documentation browser, protected layout recovery,
shell diagnostics, and command toolbars are independently mountable. Stable typed Python mount IDs
cover each surface. A catalogue-to-renderer conformance test compares every advertised built-in in
every mode with the native dispatch sources.

Future demand-driven extensions:

- add non-empty configuration schemas for other components when they gain per-instance behavior;
- add actor-derived unavailable/not-ready presentation when a future model-dependent mount can be
  legal in its mode but temporarily unusable; and
- split further composite controls only when a concrete layout needs a smaller independent mount.

Acceptance criteria:

- every catalogue entry can be mounted in every advertised legal parent and mode;
- every rejected mount explains which catalogue rule failed; and
- the catalogue cannot advertise a placement or configuration that the renderer ignores.

#### 4. Make extension contributions first-class shell nodes — substantially complete

Registration now returns a stable `shell_mount`; Python has a typed `Contribution.mount()`
builder; registered contributions appear in `ui.shell.components.list`; and the recursive
renderer places them in compatible containers without duplicating legacy-host rendering.

Compatibility locations now resolve to application-owned `builtin:extension-host.*` components
inside each mode's default desired tree. Empty hosts release geometry without changing declared
visibility, and explicitly mounted contributions are excluded from their host copy. No extension
location is rendered through an independent egui side panel, top/bottom panel, area, or project
window.

Retained extension contributions/templates now require an exact registering-version match on
reconnect. Extensions publish ready/not-ready state and an optional reason; snapshots and native
placeholders distinguish ready, not-ready, disconnected, incompatible, and missing mounts.

Ownership metadata is now present on legacy nodes, desired-layout nodes, and component
descriptors. Application roots and required workspaces are marked protected. A session that owns
registered extensions is confined to its own extension nodes; a foreign-node mutation is rejected
inside the actor with the node, mount, owner, required capability, and resolution in the error.
The native recovery authority and a session explicitly granted `ui.shell.application_control` can
compose the complete registered shell. Authentication alone grants no shell mutation authority.

Acceptance criteria:

- an extension panel can be placed in the same tabs container as a built-in panel;
- disconnecting its Python process cannot destabilize the shell; and
- reconnecting with the same compatible identity restores the retained placement and state.

### P1 — make layouts durable, safe, and operable

#### 5. Persist, migrate, import, and recover layouts — complete for schema v1

Portable v1 layout documents, v0-to-v1 migration, atomic import/export, named session profiles,
durable application profiles, project profiles, missing-extension placeholders, deterministic
reset, an explicit protected recovery layout, and extension-owned default templates are
implemented. Application profiles use the existing atomic settings worker and survive restart.
Project profiles participate in canonical project dirty/save/open transactions. Extension
templates are normalized to v1, ownership checked, and retained according to the extension's
disconnect policy. Application settings select a startup profile per mode; the actor restores each
mode once and falls back to protected recovery with typed diagnostics. Profile lists perform full
document validation and report the exact error kind, message, and recovery method.

Checked-in compatibility fixtures cover the supported v0 migration input, canonical v1 project and
viewer documents, retained missing/version-incompatible extension mounts, corrupt v1 trees,
unsupported future schemas, startup restore, and protected recovery. Tests prove that invalid input
does not advance the shell revision. The fixture policy requires a preserved input for every
readable schema version when a future migration is added.

The storage policy is settled for the current desktop architecture: `application` profiles live in
Odon's ordinary per-OS-user configuration directory and are therefore user-specific despite the
API scope name. `project` profiles remain the sharing mechanism. A machine-global or shared-service
profile store would require a separate authorization/storage design and is outside this work.

Future compatibility maintenance:

- add an explicit migration and immutable fixture before accepting any schema newer than v1.

#### 6. Add ownership-specific permissions and concurrency rules

Method introspection now separates inspection, composition, extension placement, persistence, and
protected recovery. It also reserves explicit application-chrome, window-control, shortcut, and
application-controller capabilities for the surfaces that need them. Actor-owned node metadata and
cross-extension mutation enforcement are implemented: extension sessions may mutate only their own
`extension_mount` nodes, while ownership errors identify the owner and capability required. Stale
shell revisions return the expected/current revisions, snapshot method, conflicting domain, and
`refetch_merge_retry` strategy. Caller-supplied transaction IDs correlate one atomic shell
mutation across its response and semantic event.

Implemented policy:

- `system.hello` negotiates requested and granted capabilities per authenticated session;
- method dispatch enforces shell read/composition/extension-placement/persistence/recovery grants;
- the actor enforces application-controller, chrome, recovery, and extension ownership boundaries;
  and
- the Python client requests the complete current controller grant set by default while allowing a
  caller to request a narrower set.

Protocol-v1 decision:

- do not add a general atomic multi-method batch without a concrete cross-resource invariant that
  cannot be expressed by an existing atomic mutation; the existing bounded `system.batch` remains
  explicitly ordered and non-atomic; and
- retain equivalent byte/count quotas for any future shell state or contribution payloads; layout
  nodes, configuration, and extension-template counts are already bounded.

#### 7. Add shell tooling and broader verification

Implemented tooling:

- `builtin:shell-inspector` is a native, catalogue-described mount available in every mode. It
  displays the actor revision, active/focused nodes, tree depth, node IDs/types/mounts,
  ownership/protection, required mutation capability, readiness problems, and presented geometry.
- `odon.layouts` supplies validated review, analysis, comparison, mosaic-triage, and presentation
  layouts. A worked guide demonstrates revision-guarded installation and explicit extension-panel
  placement.
- `examples/python_shell_control.py` supplies a plan-only-testable and live dataset workflow. Its
  paired macOS capture records the nested shell, mounted toolbar, checked/disabled command state,
  and both rendered outcomes of command-state-driven region visibility.
- `examples/python_extension_host_control.py` supplies a plan-only-testable retained extension
  lifecycle. Its macOS capture records an actor-owned default host ready, disconnected, and
  reclaimed by a compatible replacement session while preserving selection and shell revision.
- `examples/python_shell_interaction_control.py` supplies a plan-only-testable interaction and
  recovery workflow. Its four-frame macOS capture records a 0.24-to-0.36 split change, Project
  selection/focus, collapse and canvas focus transfer, and protected recovery through the shared
  command path. Actor output records the corresponding revisions and exact state IDs. The scripted
  Python patches are paired with native gesture-path tests rather than represented as pointer
  automation.
- `examples/python_shell_startup_mode_control.py` supplies a plan-only-testable two-process startup
  and mode workflow. It uses an isolated settings-path override, records successful v1 application
  profile restoration, transitions through the default project tree, and proves the same
  single-view layout plus active/focused IDs survive reactivation. Three macOS frames accompany the
  actor output.

Required work:

- finish the macOS baseline with menu/toolbar/palette invocation, predicate-hidden state, and
  enablement;
- run the representative nested-layout resize, selection/focus, collapse, and protected-recovery
  suite on Windows and Linux;
- repeat the local maximum-size and slow-subscriber timing gates on Windows and Linux, and extend
  them with platform TCP-disconnect and GPU/render timing evidence; and
- add further end-to-end dataset-specific examples and rendered outputs beyond the checked-in
  synthetic single-view workflow and existing two-view comparison.

### P2 — complete declarative behavior in the existing window

#### 8. Make commands, menus, toolbars, palettes, and shortcuts declarative — functionally complete

Implemented:

- actor-owned stable command descriptors are separate from presentation and publish handlers,
  modes, protection, icons, and shortcuts;
- a bounded recursive menu tree supports nested menus, command items, separators, and presentation
  label/icon overrides;
- `ui.commands.describe_schema/list` and revision-guarded `ui.menus.get/replace` are exposed through
  typed synchronous/asynchronous Python resources;
- menu mutation requires `ui.shell.chrome`, reports correlated `ui.menus.changed` events and
  actionable conflicts, and cannot remove protected close, quit, or recovery presentations; and
- macOS rebuilds its actual native menu from each changed actor projection, including accelerators
  and scale-bar checked state;
- extensions register/remove namespaced event commands under actor-enforced ownership and
  `ui.shell.shortcuts`; overlapping-mode shortcut conflicts are rejected with actionable data;
  invocation publishes the declared extension event without a GUI-thread callback; and
- remove/disable/retain disconnect policy, compatible reconnect, version mismatch, and explicit
  readiness changes reconcile retained command state and native mode/readiness enablement.
- a bounded revisioned toolbar presentation, typed synchronous/asynchronous Python API, and native
  `builtin:command-toolbar` mount render and invoke built-in native/control commands and extension
  event commands from those same descriptors; and
- menu, toolbar, settings-shortcut, and Python invocation all submit `ui.commands.execute`. The
  actor resolves and authorizes the handler, publishes `ui.commands.executed`, and emits typed
  native/control platform effects or the declared extension event; and
- a searchable native palette consumes the same descriptors and dispatch path. Its actor-owned,
  bounded presentation and shortcut are revision guarded through `ui.palette.get/replace`, exposed
  through typed sync/async Python resources, and conflict checked against command shortcuts;
- bounded visible/enabled/checked predicates evaluate capability and published actor state through
  one session-aware command-state projection shared by all presentations and direct execution; and
- boundary tests exercise the complete predicate node/depth allowance and rapidly changing actor
  selection state, while toolbar validation is tested at its maximum item count;
- toolbar items render label/icon/tooltip overrides, actor-derived checked state, and disabled
  reasons, and submit toggled check state through `ui.commands.execute`;
- mounted toolbar buttons publish native accessibility roles, checked/toggled state, and tooltip or
  disabled-reason descriptions; AccessKit clicks and keyboard Tab/Enter activation use the same
  toggled invocation, and a changed actor projection reconciles enabled state on the next frame;
- command toolbars are intentionally opt-in: the actor starts with an empty toolbar presentation
  and default shell layouts do not reserve an empty toolbar region; a layout/profile that wants one
  mounts `builtin:command-toolbar` explicitly;
- Windows and Linux resolve all eligible descriptor shortcuts from the current actor projection,
  consume exact key events, toggle checked state, and submit the shared execution method without a
  detached registration that can become stale; and
- native interaction tests prove text-focus suppression and immediate replacement of changed
  shortcut projections, while menu, shortcut, toolbar, and palette entry points construct one
  common actor execution intent;
- every built-in native command has an explicit, conformance-tested mode set matching its Rust
  realizer. Viewer-only and single-viewer-only commands are disabled before an unsupported native
  fallback can run;
- protected close, quit, and recovery resolve to identical actor results, semantic events, and
  platform effects from native presentation intents and direct Python execution; and
- shortcut conflicts use platform-effective aliases, while schema introspection publishes neutral
  labels and platform-specific support and unsupported-modifier diagnostics.

Required work:

- capture native toolbar and shortcut realization evidence on Windows and Linux, including
  protected close, quit, and recovery smoke runs.

#### 9. Complete declarative state bindings — functionally complete

Implemented:

- extension control values bind to bounded native viewer/channel/layer targets and remain
  responsive without Python;
- `Component.when()` plus `command_state()` binds component `visible` and `enabled` properties, and
  `ShellLayoutNode.state_bindings` binds shell-node visibility, to actor-evaluated command
  visible/enabled/checked state, reusing the bounded command predicate projection instead of
  introducing arbitrary state paths or callbacks;
- shell bindings round-trip through export/import and profiles while their effective visibility is
  derived in the active actor projection rather than overwriting desired persisted state;
- missing commands resolve bound properties to false and changed projections reconcile before the
  next render;
- immediate, throttle, and debounce policies coalesce high-frequency component interactions per
  stable component key; event queues remain bounded and slow subscribers cannot block publishers;
  and
- Python cannot synchronously intercept or cancel native interactions. It observes semantic events
  and can submit a later revision-guarded mutation; and
- a source conformance test compares every actor-declared native command action with every Rust
  realization arm, while shell gesture tests require semantic patch/event commits.

Release evidence:

- paired rendered visibility evidence and both actor results are captured on macOS; add the
  equivalent enablement and predicate-hidden cases and repeat the state matrix on Windows and
  Linux. Capability denial remains a requesting-session API property and should be captured from a
  narrow Python session rather than attributed to native chrome's trusted evaluation context.

Additional bounded binding targets are added only for concrete workflows; arbitrary actor paths and
Python predicates remain intentionally unsupported.

## Recommended remaining execution order

1. Extend the paired macOS baseline with the remaining presentation-invocation, predicate-hidden,
   enablement, and representative-host cases. Resize, selection/focus, collapse, protected
   recovery, isolated startup restore, mode transition, and retained reconnect are captured.
2. Run the same toolbar, shortcut, protected-command, recovery, binding, and interaction suite on
   Windows and Linux.
3. Repeat the local timing gates and add platform TCP-disconnect plus GPU/render timing evidence.
4. Extend the workflow guide with dataset-specific shell examples and captured outputs.
5. Add migrations, component schemas, or binding targets only when a concrete future schema or
   workflow requires them.

These are qualification and demand-driven extension tasks for the completed single-window model;
they do not expand the initiative into multi-window architecture.

## Definition of complete Python GUI control

The goal is complete when a Python client can:

1. discover every available built-in and extension component and its constraints;
2. submit the complete existing-window application layout as one revision-guarded transaction;
3. configure, move, show, hide, and remove every non-protected GUI region;
4. arrange native canvases, native components, and extension components in nested containers,
   tabs, and splits within the existing window;
5. define menus, toolbars, palettes, and shortcuts through validated command descriptors;
6. observe all semantic interactions without executing Python on the GUI thread;
7. persist, migrate, share, reset, and safely recover layouts; and
8. receive the same actor-owned state whether a change originates in Python or native UI.

Rust should retain unconditional ownership of egui objects, rendering, GPU and platform resources,
the event loop, safety validation, bounded queues, protected recovery controls, and failure
containment.

This completion definition deliberately excludes creating or arranging additional native windows,
docking across windows, monitor placement, and cross-window canvas transfer.

## Related documents

- [Concise remaining-work checklist](python-controlled-application-shell-remaining-work.md)
- [Python-controlled shell performance evidence](python-shell-performance.md)
- [Python API contracts](../reference/python-api-contracts.md)
- [Current Python API limitations](../advanced/python-api-limitations.md)
- [Application state ownership inventory](app-state-ownership-inventory.md)
- [Complete application Python API plan](complete-application-python-api-plan.md)
