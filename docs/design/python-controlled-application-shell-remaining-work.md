# Python-controlled application shell: what remains

Status: audited 2026-08-25

This is the concise execution checklist. The detailed architecture and implementation history are
in the [application-shell roadmap](python-controlled-application-shell-roadmap.md).

## Current boundary

Python now has broad declarative control of Odon's existing native window. It can replace and patch
the recursive shell tree, arrange built-in and extension mounts, observe native layout interactions,
manage layouts and recovery profiles, define extension commands, reorganize menus and toolbars, and
configure the searchable command palette.

This initiative now ends at complete declarative control of that existing native window.
Multi-window architecture—including docking, floating regions, detached native windows, monitor
placement, and moving canvases between windows—is explicitly out of scope and is not counted as
remaining work. It would require a separate architectural proposal and design review.

Menus, mounted toolbars, the palette, supported shortcuts, and Python all invoke the same
actor-resolved `ui.commands.execute` path. The actor validates mode, readiness, ownership, and
capability, then resolves native, control-method, or extension-event handlers without executing
Python on the GUI thread.

Commands also expose one actor-evaluated state across every presentation. Bounded `visible`,
`enabled`, and `checked` predicates cover session capabilities and a fixed resource, selection,
mode, GPU, panel, and scale-bar context. Toolbar items render icon/tooltip/label overrides,
checked selection, and disabled reasons.

The native palette milestone is complete. `ui.palette.get/replace` exposes a revisioned,
chrome-capability-guarded presentation with configurable title, placeholder, shortcut, description
visibility, and bounded result count. Rust conflict-checks its shortcut, searches the shared command
catalogue, filters by active mode and readiness, and dispatches the selected command through the
shared path. Typed synchronous and asynchronous Python resources cover the same contract.

The verified checkpoint passes 493 Rust tests (4 ignored) and 134 Python tests. Formatting,
`cargo check --all-targets`, generated-reference validation, JSON-manifest validation, and diff
checks also pass.

## P0 — finish control of the existing native window

### 1. Finish contextual-predicate coverage and evidence

The bounded predicate vocabulary, typed sync/async Python builders, actor evaluation, session-aware
command listing, direct-execution enforcement, and shared menu/toolbar/palette state are implemented.
Scale-bar checked state now uses the same actor context instead of a menu-specific synchronization
path. Boundary tests accept exactly the advertised 32 predicate nodes and depth 8, reject one more,
and repeatedly reconcile selection-dependent command state across 512 changing projections.

The built-in audit is complete and enforced by a conformance test that gives every native command
an explicit mode set. Screenshot settings are viewer-only; ROI information, annotations,
segmentation loading, mask export, and scale-bar controls are single-viewer-only. Commands that
cannot be realized in the active mode are therefore disabled by the actor before any presentation
can invoke their defensive native fallback.

Required work:

- capture the multiplex-review cockpit's predicate-hidden command and separate enablement binding
  on macOS, then repeat the rendered state matrix on Windows and Linux. Its live output already
  retains the structured narrow-session capability denial as API evidence. Native chrome evaluates
  in Odon's trusted native context, so it intentionally cannot render a denial caused by one remote
  session's missing capability.

New fixed context paths remain demand-driven protocol extensions rather than unfinished work.

Done when every presentation reports and enforces the same actor-derived enabled and checked state.

### 2. Finish rich command-toolbar presentation

The toolbar resource, validation, typed Python builders, native mount, shared command dispatch,
label/icon/tooltip overrides, checked state, and disabled-reason rendering are implemented. Native
tests exercise AccessKit click dispatch, button role, toggled state, accessible descriptions,
Tab/Enter activation, and disabled-to-enabled projection reconciliation.

The default-policy decision is also complete: command toolbars are opt-in. The actor's default
presentation is empty and the default shell trees do not consume vertical space for it. A Python
layout or saved profile mounts `builtin:command-toolbar` when it has a workflow-specific toolbar to
show.

Required work:

- capture Windows and Linux rendering evidence.

Done when one Python-defined toolbar works consistently for native, control, and extension-event
commands in every advertised mode.

### 3. Finish cross-platform shortcut and native-chrome evidence

macOS menu accelerators and the palette shortcut work. Windows and Linux now resolve every eligible
descriptor shortcut from the current actor projection, consume its exact event, toggle checked
state, and submit `ui.commands.execute`. Changed command and palette projections take effect without
an OS-side registration that can become stale. Platform-effective aliases participate in conflict
checks, and schema introspection publishes neutral labels, current mappings, supported modifiers,
and actionable unsupported-modifier diagnostics.

Native interaction tests now prove focused text input suppresses descriptor shortcuts, checkable
commands submit the toggled state, and a changed actor projection immediately retires the old
shortcut without stale registration. Menu, shortcut, toolbar, and palette entry points also build
one common `ui.commands.execute` intent. Actor tests prove protected close, quit, and recovery
produce identical results, events, and platform effects for that native intent and direct Python
execution.

Required work:

- capture rendered Windows and Linux evidence for built-in and extension descriptor shortcuts;
- include protected close, quit, and recovery smoke runs in that rendered platform evidence.

Done when an available command has the same validation, execution, event, and recovery behavior
regardless of presentation or desktop platform.

### 4. Close built-in component-catalogue gaps

Channels, single-view viewport controls, the documentation browser, protected recovery controls,
shell diagnostics, and the command toolbar are independently catalogue-described and mountable.
Python exposes stable typed IDs for them. A conformance test compares every advertised component in
each mode with the corresponding native renderer dispatch source.

Required work:

- add actor-derived unavailable/not-ready placeholders if a future model-dependent component is
  valid in the active mode but temporarily unusable;
- publish non-empty per-instance schemas whenever a built-in gains configurable behavior;
- extend the checked-in macOS ready/disconnected/reconnected left-sections host evidence to other
  representative legal parents and to Windows and Linux; and
- split further composite controls only when a concrete layout needs a smaller independent mount.

Done when Python can rearrange every non-protected region in the existing Odon window and every
catalogue claim is enforced by the native renderer.

## P1 — harden and release the single-window shell

### 5. Maintain the completed persistence baseline

Version 1 documents, deterministic export/import, v0-to-v1 migration, session/application/project
profiles, startup restore, and protected recovery are implemented. Checked-in fixtures now cover
v0 migration, canonical v1 project/viewer documents, corrupt v1 input, an unsupported future
schema, startup activation, protected recovery, and missing/disconnected/version-incompatible
extension mounts. Invalid imports are verified not to advance the shell revision.

The storage-scope decision is complete. `application` profiles use Odon's existing settings file in
the current OS user's configuration directory, so they are already user-specific. `project`
profiles are the portable/shared scope. A machine-global or shared-service profile store is a
separate authorization and deployment feature, not part of this shell work.

Future maintenance:

- add an explicit migration and preserve a fixture when schema v2 is designed; and
- apply equivalent retained-state quotas to new bindings or command-surface payloads.

The schema-v2 migration is intentionally conditional; it is not a reason to invent v2 early.

### 6. Add cross-platform rendered, interaction, and performance evidence

A paired macOS dataset-specific capture is checked in with the
[Python shell workflow](../examples/python-shell-workflows.md). It opens the synthetic five-channel
OME-Zarr fixture, renders a 23-node nested review layout, mounts a four-command toolbar, and shows
checked and disabled actor-derived command state. Paired frames from one live run show the bound
right panel changing from visible to hidden and the canvas taking its space when the checked
scale-bar command changes. The actor reports the effective visibility change without advancing the
persisted shell-document revision or invoking a Python frame callback. A plan-only regression test
validates the exact layout, toolbar, protected command, disabled command, and binding used by the
example.

A second repeatable macOS workflow registers a Python panel into the actor-owned default
left-sections host from an independent extension session. Checked-in frames and actor output cover
ready, retained-disconnected, and compatible-reconnected states without replacing the desired tree
or advancing its revision. The live run exposed and fixed passive legacy tab synchronization
overwriting an extension-host selection; a regression test now proves passive compatibility
projection preserves the desired selection while an explicit legacy tab command may still change
it.

A third repeatable macOS workflow installs a 23-node review tree with a collapsible left region.
Four checked-in frames plus actor output show ratio 0.24 with Layers selected; an atomic change to
ratio 0.36 with Project selected and focused; the left region collapsed with focus transferred to
the canvas; and protected `app.shell.recover` dispatched through `ui.commands.execute` replacing
the tree with its two-node recovery canvas. Revisions advance 2 through 5 and the output records
every split, selection, collapse, active/focused, handler, root, and mount assertion. The example
uses Python patches for reproducibility; native interaction tests prove pointer/keyboard gestures
commit the same actor patch and semantic-event path.

A fourth repeatable macOS workflow performs a real isolated process restart. A setup process saves
and selects a distinctive 23-node application profile through the public Python API; the evidence
process reports that profile restored at first single-view activation with schema v1, ratio 0.34,
and Project selected. It then records the default project tree and returns to the same single-view
tree with Project active/focused at unchanged revision 3. `ODON_SETTINGS_PATH` provides an explicit
per-process settings file for this evidence, and the workflow removes its temporary directory
without reading or changing the normal user profile.

Required work:

- capture native macOS interaction evidence for menu/toolbar/palette invocation. The checked-in
  multiplex-review cockpit now implements the shared command routes, predicate-hidden and disabled
  states, bound Python controls, extension events, and a structured restricted-session denial;
- run the same nested-layout resize/selection/focus/collapse/recovery suite and retained extension
  lifecycle on Windows and Linux;
- repeat the locally timed 256-node tree, 128-item toolbar, maximum-predicate, and sustained
  slow-subscriber gates on Windows and Linux, and add platform TCP-disconnect plus GPU/render
  timing evidence; and
- add rendered output for the checked-in multiplex-review cockpit and further dataset-specific
  worked Python examples beyond it and the existing two-view comparison.

The local CPU/backpressure baseline, reproduction commands, broad non-blocking budgets, and
supporting queue/projection tests are recorded in
[Python-controlled shell performance evidence](python-shell-performance.md).

### Active next tranche: command-surface qualification

The dataset-specific implementation and deterministic API evidence are now checked in as
`examples/python_multiplex_review_cockpit.py`. Native pointer interaction and rendered evidence
remain a bounded follow-up:

1. Capture the disabled/hidden multiplex-review controls and their enabled/visible counterparts
   after loading mask and object resources.
2. Invoke the same extension command through the native macOS menu, mounted toolbar, and command
   palette. Assert that all three entry points produce the same `ui.commands.execute` result and
   the same namespaced extension event; do not substitute direct Python invocation for the native
   interaction evidence.
3. Retain screenshots, checksums, and the live output beside the existing macOS evidence. Native
   chrome uses Odon's trusted native capability context, so the checked-in narrow-session denial
   must not be presented as native-menu state.
4. Restore the original shell, menu, toolbar, palette, and scale-bar state, unregister the
   extension, and terminate only an application process launched by the workflow.
5. Add the resulting checksums, assertions, and screenshots to the workflow guide and API surface
   manifest, then run the complete Rust/Python/reference/JSON/format/diff verification suite.

This tranche remains single-window work. It does not reopen docking, floating panels, detached
windows, or any other multi-window architecture.

Done when supported behavior has repeatable visual, interaction, recovery, and performance evidence
on every supported desktop platform.

### 7. Multi-method transaction decision — complete

Layout replacement and patches are already atomic, revision guarded, and transaction correlated.
No current shell workflow requires a transaction spanning unrelated methods, so protocol v1 will
not add a general atomic batch primitive. The existing `system.batch` remains a bounded, ordered,
explicitly non-atomic command convenience. Each mutable resource retains its own atomic replacement/patch,
revision guard, and transaction correlation. A future batch requires a concrete cross-resource
invariant that cannot be represented as one existing mutation; it is not remaining work here.

## P2 — complete declarative behavior in the existing window

### 8. Declarative actor-resource bindings — functionally complete

Extension component values already bind to bounded native viewer/channel/layer targets through
`ui.bind(...)`. Components can now also bind `visible` and `enabled`, and shell nodes can bind
`visible`, to the evaluated visible/enabled/checked state of any actor-owned command with
`command_state(...)`. Missing commands safely resolve false, bindings persist in layout documents
and profiles, predicates stay inside the bounded actor vocabulary, and reconciliation occurs from
the actor projection without a Python frame callback.

High-frequency component interactions are bounded per component. Immediate events are limited to
the render cadence, throttle/debounce policies are clamped and coalesce to the newest value in one
retained map entry, and slow event subscribers drop bounded queued events rather than blocking the
publisher. Python cannot synchronously intercept or cancel a native Odon interaction; it observes
the semantic result and may submit a later revision-guarded command.

A conformance test compares the complete actor-declared native action set with the Rust realizer's
match arms, so a descriptor cannot silently fall through as unsupported and an undeclared native
application action cannot be added there. Existing shell interaction tests require tabs, collapse,
focus, activation, and split commits to use `ui.shell.patch_layout` and `ui.shell.changed`.

Release evidence retained under item 6:

- the paired macOS before/after view proves command-state-driven visibility and actor output; add
  equivalent enablement evidence and repeat both cases on Windows and Linux.

New binding targets remain demand-driven protocol extensions, not an incomplete generic-expression
system.

Done when dynamic GUI state follows application data without polling or executing Python on the GUI
thread.

## Recommended execution order

1. Extend the paired macOS baseline with extension hosts beyond left sections,
   menu/toolbar/palette invocation, predicate-hidden state, and enablement. Resize, selection/focus,
   collapse, recovery, isolated startup restore, mode transition, and the retained
   disconnect/reconnect lifecycle are now captured. Keep capability denial as narrow-session
   structured API evidence because native chrome uses the trusted native capability context.
2. Run the same toolbar, shortcut, protected-command, recovery, binding, and interaction suite on
   Windows and Linux.
3. Repeat the timing gates and add platform TCP-disconnect plus GPU/render timing evidence.
4. Add dataset-specific shell workflows with captured actor and rendered outputs.
5. Maintain fixtures and add component schemas or binding targets only when future concrete
   workflows require them.

These steps qualify the implemented Python control of Odon's existing native window; they do not
expand the native-window architecture.

## Boundary that should remain in Rust

Complete Python control should remain declarative. Rust should continue to own egui objects, the
event loop, rendering and GPU resources, native window handles, platform effects, validation,
bounded queues, protected recovery controls, and failure containment. Python should specify desired
state and observe semantic outcomes; it should not execute callbacks on the GUI thread.
