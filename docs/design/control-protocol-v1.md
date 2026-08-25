# Odon Control Protocol v1

Status: experimental implementation; no stable compatibility promise yet

## Transport and framing

Version 1 uses UTF-8 JSON-RPC 2.0 messages over a persistent loopback TCP
connection. Each message is one JSON object followed by `\n`. Authenticated
connections use a bounded worker pool, requests may execute concurrently, and
responses are correlated by JSON-RPC request ID rather than response order.

Each GUI binds an ephemeral loopback endpoint and atomically publishes a
user-private runtime manifest. The manifest contains an instance ID, endpoint,
protocol versions, PID, and a random 256-bit bearer token. Clients remove stale
manifests only after checking that their endpoint is unreachable.

## Handshake

The first JSON-RPC request on every connection must be:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "system.hello",
  "params": {
    "token": "token-from-private-runtime-manifest",
    "client": {"name": "odon-client", "version": "0.1.0"},
    "protocol_versions": [1]
  }
}
```

The response selects a protocol version and returns instance/session IDs,
application and control API versions, capabilities, permission policy, and the
maximum inline payload. Requests made before a successful handshake fail with
`HANDSHAKE_REQUIRED`; invalid authentication closes the connection.

## Events, revisions, and tasks

`events.subscribe` accepts exact names and suffix-wildcard patterns. Odon emits
bounded JSON-RPC notifications using method `events.event`; each carries a
global sequence, state revision, source ID, payload, and optional initiating
session/request IDs. Slow subscribers drop events rather than block egui and can
inspect drop counts with `events.get_status`.

Successful mutations increment the shared revision. Mutating application
commands accept `if_revision` and return `CONFLICT` if it is stale.

`tasks.start` submits an existing application command and immediately returns a
retained task snapshot. `tasks.get`, `tasks.list`, `tasks.cancel`, and
`tasks.forget` manage it. Open operations settle when the viewer is ready;
screenshot operations settle when output is written. Progress and terminal
states are emitted as `tasks.*` events. Cancellation is cooperative: queued work
can be cancelled, but an indivisible operation already running on the UI thread
cannot be forcibly interrupted.

## Referenced data, layers, and UI

Large data is registered by URI through `data.resources.*`, with explicit
format, axes, units, scale, translation, ownership, metadata, and provenance.
`viewer.layers.*` manages stable external layer IDs, style, order, visibility,
and lifecycle without putting arrays in JSON. These resource and layer methods
execute in the control actor's serialized command stream. Consequently, a
following `project.save` observes every successful preceding project-owned
resource/layer mutation even when no GUI frame occurs.

`ui.extensions.*` and `ui.contributions.*` retain bounded, versioned component
trees. Rust renders the supported controls through egui and emits extension
events. Native command actions and supported layer bindings execute locally.
Contributions receive stable first-class shell mounts. Right tabs, left
sections, top-bar actions, a status bar, canvas controls, and project cards
remain supported legacy/default-placement hosts.
`ui.shell.export_layout` and `ui.shell.import_layout` exchange versioned layout
documents. Named `ui.shell.profiles.*` resources use actor-session, durable
application-settings, or canonical project-state scope, and `ui.shell.recover` installs the protected
minimal workspace after an incompatible restore.
`app.settings.set.shell_layout_startup_profiles` selects an application profile
for each mode. On the next process start the actor attempts each configured
restore once, at first mode activation; any missing, malformed, incompatible,
or wrong-mode document is replaced by protected recovery and diagnosed by
`app.settings.get`.
Owned `ui.extensions.layouts.register/list/remove` resources let an extension
publish bounded default layout templates. Registration validates mount
ownership and normalizes documents to v1; templates follow the extension's
remove/disable/retain connection lifecycle and are applied explicitly through
`ui.shell.import_layout`.
`ui.commands.describe_schema/list` expose stable actor-owned command descriptors independently of
their visual presentation. `ui.menus.get/replace` expose a separate revision domain for the
bounded recursive platform-menu tree. Replacement requires `ui.shell.chrome`, validates every
command reference and node ID, preserves protected close/quit/recovery presentations, and emits
`ui.menus.changed` with the caller's optional transaction ID. Stale guards report
`conflicting_domain=application_command_surface` and direct clients to `ui.menus.get`. On macOS,
the projected tree is realized as the actual native application menu, including nested submenus,
separators, accelerators, and scale-bar checked state.
`ui.toolbars.get/replace` and `ui.palette.get/replace` share that command-surface revision and
chrome capability. The palette presentation controls its title, search prompt, shortcut,
description visibility, and a bounded result count. Its native egui realization searches the same
descriptor catalogue and filters by active mode and readiness; its shortcut is conflict checked
against command shortcuts.
Commands may also carry bounded `visible`, `enabled`, and `checked` predicates. The vocabulary is
limited to `always`, session `capability`, published actor `state`, `all`, `any`, and `not`, with
node/depth quotas and a fixed state-path catalogue covering resources, object/mosaic selection,
panel visibility, scale-bar state, mode, and GPU readiness. Each actor projection evaluates the
same state record consumed by the macOS menu, mounted toolbar, searchable palette, and direct
execution. `ui.commands.list` evaluates capability predicates for the requesting session; native
projection uses trusted native authority. Failed execution returns the evaluated reasons and any
missing capabilities instead of allowing presentation state to bypass actor enforcement.
The component catalogue also exposes independently mountable channels, single-view viewport
controls, documentation, protected layout recovery, shell diagnostics, and command-toolbar
surfaces. Typed mount IDs are stable in Python. A native conformance test compares the complete
mode-scoped catalogue with project, single-viewer, and mosaic dispatch coverage so an advertised
built-in cannot silently degrade into an ignored mount.
Owned extensions can add namespaced event commands with `ui.commands.register` and remove them
with `ui.commands.remove`; these mutations require `ui.shell.shortcuts`, the owning session, and
an extension declaration containing `ui.actions`. Shortcut collisions are rejected when modes
overlap, including platform-effective aliases (`primary`/Command on macOS and `primary`/Ctrl on
Windows and Linux). The schema publishes neutral display labels, the current platform mapping, and
supported modifiers. A non-realizable modifier is rejected with `UNSUPPORTED`, platform, and
resolution data. On Windows and Linux, the native shell resolves eligible descriptors directly
from the latest actor projection each frame, consumes the exact key event, and submits
`ui.commands.execute`; replacing a command or palette shortcut therefore leaves no stale native
registration. `ui.commands.execute` resolves every ready descriptor in the actor: extension handlers
publish their declared event, while authorized native and control handlers become typed platform
effects for the Rust application shell. Menu items, toolbar buttons, the command palette, native
shortcuts, and Python therefore share one mode/readiness/permission-checked dispatch path and publish
`ui.commands.executed`. An optional boolean `checked` value carries check-menu intent without
letting callers replace handler parameters. Native disconnect cleanup removes or disables
descriptors and their menu/toolbar presentations according to the extension policy, and compatible
reconnects reconcile retained commands.
Toolbar items may override label, icon, tooltip, and label visibility. Native buttons render
checked state as selected, include actor-derived unavailable reasons in their tooltip, and submit
the toggled boolean through the same command dispatcher. They publish native accessibility role,
toggled state, and description metadata; AccessKit clicks and keyboard activation use that same
dispatcher. The actor's default toolbar is empty and default shell layouts do not mount
`builtin:command-toolbar`; workflow layouts and profiles opt in explicitly.
Retained contributions and templates record their registering extension version. Reconnects with
a different version are incompatible until the resource is registered again.
Shell-layout compatibility fixtures preserve every readable schema shape and exercise corruption,
unsupported versions, startup restore/recovery, and missing or incompatible retained mounts.
Application-scoped layout profiles use the current OS user's Odon settings directory; project
profiles remain the portable sharing scope.
Declarative extension component values use bounded `ui.bind` targets. A component's `state_bindings`
may additionally map `visible` or `enabled` to a `command_state` descriptor containing a stable
command ID, one of `visible`, `enabled`, or `checked`, and an optional expected boolean. Odon reads
the evaluated actor command projection; a missing command resolves false, and Python is never called
from the render thread. Desired shell nodes accept the same descriptor for `visible`; it persists in
layout documents/profiles while the effective value is derived in the active projection. Immediate
component interactions are render-cadence limited; throttle and
debounce policies coalesce the newest value per stable component key. Python observes these
semantic events but cannot synchronously intercept or cancel native interactions.
The actor-declared native action set is source-conformance checked against the Rust realizer arms;
shell layout gestures commit through the registered semantic patch/event path.
`ui.extensions.set_readiness` lets the owning session publish ready/not-ready state and a reason;
desired mount snapshots distinguish ready, not-ready, disconnected, incompatible, and missing.

Shell snapshots and component descriptors identify application- and extension-owned nodes and
mark protected application roots/workspaces. Ownership checks execute in the actor as well as the
transport preflight, so an extension session cannot mutate another extension's retained node by
bypassing the TCP service. Permission errors carry the affected node/mount and owner; shell
revision conflicts carry the current revision and an explicit refetch/merge/retry contract.
Shell snapshots also own active-region and optional focus IDs. The revision-guarded layout patch
sets either ID or clears focus explicitly. Native interactions use the same transaction and emit
precise per-node visibility, order, selection, size, split, collapse, activation, and focus
changes. Split resizing commits once when dragging stops; preview frames and unchanged final
ratios do not enter the actor queue.
The desired-tree application root is a vertical menu/toolbar/content/status stack. Built-in
project, viewer, and mosaic top bars render through their catalogue mount IDs with intrinsic
chrome sizing; fixed native top panels are a no-projection compatibility fallback.
Mount nodes carry configuration objects. `ui.shell.patch_layout.configurations` updates them under
the shell revision guard, and built-in component descriptors publish the validating schema. Native
top-bar schemas currently expose boolean visibility groups; unknown or wrongly typed properties
reject the entire shell transaction.
Shell method metadata uses distinct read, composition, extension-placement, persistence, and
protected-recovery capabilities. Chrome, platform-window, and shortcut capability names are
reserved for those forthcoming surfaces. Extension sessions are actor-confined to their own
extension-mount nodes; application-owned or foreign nodes return the precise required capability.
`system.hello` accepts `requested_capabilities` and returns `granted_capabilities`; authentication
alone grants no shell mutation authority. The standard Python client explicitly requests the
current application-controller set by default, while callers can request a narrower set.

Session-owned descriptors are cleaned when their authenticated connection
closes. Project-owned data-resource and layer descriptors persist in project
JSON and are restored when that project opens. A project-owned layer cannot
reference a session-temporary resource.

## Introspection

- `system.get_capabilities` returns the negotiated protocol version and
  capability names.
- `system.list_methods` and `system.describe_methods` return method metadata,
  request schemas, execution class, readiness requirements, completion contract,
  exact completion point, and cancellation policy.
- `system.get_diagnostics` reports actor health, bounded-work counters,
  per-method `actor`/`hybrid`/`legacy_ui`/`control_service` routing, and
  queue/model/reply/presentation timing independently.
- `system.describe_events` describes the event envelope and event families.
- `ui.describe_schema` returns the native declarative UI vocabulary and limits.
- `ui.commands.describe_schema` returns command/menu/toolbar/palette vocabulary, quotas, shortcut
  modifiers, and protected-presentation policy.

Application methods are admitted and validated through the central Rust control
registry. The Python SDK and MCP adapter both use this boundary; MCP retains
only its client-specific tool descriptions and result formatting. Methods are
currently provisional.

Every canonical method is classified by the same registry value returned to
both transports:

| Completion contract | Response/task completion point |
| --- | --- |
| `immediate_semantic` | The serialized actor model commit has completed. |
| `resource_ready` | Required resources or filesystem output are installed and generation-current. |
| `retained_background` | The initial call returns a retained task; its terminal event carries the result. |
| `presentation_dependent` | A matching projection has been presented and output commits atomically. |

Task methods advertise cooperative cancellation. Non-task methods report
`not_applicable`; timing out a client-side wait never implies server-side cancellation.

An `actor` route is a no-frame execution guarantee, not merely a preferred
dispatcher. If its model/resource prerequisites are unavailable during a
dataset transition, the request returns `NOT_READY` with readiness detail; it
must not fall through to `legacy_ui`. Resource-class actor methods run blocking
I/O on bounded workers and return or install generation-checked results.
Project open installs and validates the persisted resource/layer manifest in
the actor transaction before replying; project file I/O likewise runs on the
worker pool. Samplesheet inspection, validation, import, export, and recursive
OME-Zarr discovery also use those workers; mutating results are installed only
if their project generation is still current. A delayed renderer frame therefore cannot restore an older
manifest over newer Python commands.
The UI consumes latest-value actor projections and reports the consumed
projection revision. Renderer observations are compatibility-only and cannot
write migrated semantic fields back over newer actor state. Explicit per-viewport object
appearance, fill-property, and legend overrides are also actor-owned; a requested property is
retained declaratively while its shared object column is still materializing.
Primary object imports run on the same bounded worker pool and return an immutable semantic index
with a shared renderer-native preload. `viewer.objects.source.load` replies only after that index
is installed (`model_ready` and `resources_ready`), while pixel presentation remains asynchronous.
Per-viewport object filter evaluation also runs on workers, commits with document/resource/
presentation generation checks, and returns authoritative visible/hidden counts without waiting
for egui or the operating-system window compositor.
Mask CRUD, selection, undo, and project synchronization commit in the actor as well. GeoJSON
import/export uses the bounded worker pool; import results carry document and mask-generation
checks, exports report `output_ready`, and native committed edits use an atomic
`viewer.masks.state.replace` transaction with `expected_generation`. The renderer receives a
versioned mask projection and is never part of semantic completion.
Primary-object selection is likewise canonical actor state. World-coordinate rectangle/lasso
queries, logical-viewport queries, ID selection, focus, and selection clearing execute without a
frame. Standalone filter selection uses the bounded worker pool and commits only if its document,
object-resource, operation, and selection generations are still current. Native primary-object
selection commits through `viewer.objects.selection.state.replace`; the renderer receives only
selected indices, primary index, and generation, while immutable geometry remains in the shared
resource. Explicit active/spatial-shape targets and screen-coordinate rectangles remain a
parameter-scoped compatibility route pending spatial-layer and exact-canvas-origin migration.
Single-image `viewer.native_layers.*` and `viewer.viewports.layers.*` methods execute in the actor.
The canonical model retains stable descriptors, stack order, active identity, visibility,
presentation, offsets, and loaded-offset baselines while the renderer retains actual image,
geometry, and GPU resources. Native UI layer edits use
`viewer.viewports.layers.state.replace` with `if_presentation_revision`; stale native frames
therefore conflict instead of overwriting newer remote layer state.
Local `viewer.labels.*` discovery, load, visibility, and unload also execute without a frame.
Label metadata is opened on the bounded worker pool and committed with document/label generation
checks; the renderer receives the shared label resource and creates GPU/tile-loader state only
when presentation resumes.
Saved-view capture serializes actor-owned viewport state directly. Saved-view apply also executes
in the actor when referenced segmentation resources are already ready; presets requiring a
missing object or label load use the documented hybrid compatibility route.

## Errors

Application failures use JSON-RPC error responses:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "error": {
    "code": -32602,
    "message": "zoom must be finite and greater than zero",
    "data": {"kind": "INVALID_PARAMS", "method": "set_camera"}
  }
}
```

Defined initial kinds include `PARSE_ERROR`, `INVALID_REQUEST`,
`METHOD_NOT_FOUND`, `INVALID_PARAMS`, `APPLICATION`, `AUTHENTICATION_FAILED`,
`HANDSHAKE_REQUIRED`, `INCOMPATIBLE_PROTOCOL`, `NOT_READY`, `UNSUPPORTED`,
`CONFLICT`, `CANCELLED`, `RESOURCE_NOT_FOUND`, `PERMISSION_DENIED`, `WRONG_MODE`,
`RESOURCE_LIMIT`, `TIMEOUT`, and `INTERNAL`.

## Compatibility bridge

Explicit development/test listeners may accept the previous envelope:

```json
{"method": "get_camera", "params": {}}
```

and return `{"ok": true, "result": ...}`. Published authenticated instances
reject legacy envelopes. The compatibility path will not gain v1 features.
