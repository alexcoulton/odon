# odon-client

`odon-client` is the pure-Python SDK for controlling and extending a separately
running Odon microscopy viewer. Odon does not bundle Python, and the SDK does
not bundle Odon.

```python
import odon

with odon.connect() as app:
    print(app.application.get_state())
    app.channels.set_visible(["DAPI", "CD3"])
    app.viewer.set_camera(center=(500, 700), zoom=0.5)
```

Each Odon process publishes a private authenticated discovery manifest for its
dynamic loopback endpoint. Use `odon.list_instances()` or
`odon.connect(instance="…")` when several instances are open. You can also
start an installed executable with `odon.launch(path)` or
`await odon.launch_async(path)`.

Async calls share a persistent connection and do not block the Python event
loop:

```python
import asyncio
import odon

async def main():
    async with odon.connect_async() as app:
        camera, channels = await asyncio.gather(
            app.viewer.get_camera(), app.channels.list()
        )
        await app.viewer.set_camera(zoom=1.0)
        print(camera, channels)

asyncio.run(main())
```

Long operations return retained task handles. Sync code uses `task.wait()`;
async tasks are directly awaitable. Completion and progress are pushed by Odon,
not polled:

```python
task = app.projects.open("experiment.odon")
task.wait(timeout=120)

async_task = await app.screenshots.capture("view.png")
await async_task
```

External analysis sends large results by reference:

```python
space = odon.CoordinateSpace(axes=("y", "x"), scale=(0.5, 0.5))
labels = app.data.register(
    "file:///data/cellpose.zarr",
    format="ome-zarr",
    coordinate_space=space,
)
layer = app.layers.add(labels, name="Cellpose", kind="labels")
```

Single-image mode also supports a linked two-viewport comparison without
duplicating the open document or raw caches:

```python
views = app.viewer.viewports.compare(titles=("A", "B"), ratio=0.55)
views.left.set_visible_channels(["DAPI", "CD3"])
views.right.set_visible_channels(["DAPI", "PanCK"])
views.left.objects.set_style(fill_cells=True, color_property="marker_a")
views.right.objects.set_style(fill_cells=True, color_property="marker_b")
views.left.set_rendering(smooth_pixels=False)
views.right.set_rendering(smooth_pixels=True)
app.viewer.viewport_links.update(fields=["plane", "selection"])
app.screenshots.capture_workspace("comparison.png").wait()
```

Install the `arrays` extra to use sync or async `register_numpy(...)`. It writes
a managed temporary Zarr resource and cleans it with the session. Project-owned
resource and layer descriptors persist in Odon project JSON.

Declarative components in `odon.ui` are validated and rendered as native egui
controls by Rust. They support Rust-local Odon commands and state bindings as
well as events delivered to Python. `odon.run_extension(...)` provides a
reconnecting lifecycle runner for separately packaged extensions.
Use `component.when(visible=ui.command_state(...), enabled=ui.command_state(...))` to bind those
properties to evaluated actor-owned command state without polling or Python render callbacks.
`ShellLayoutNode(..., state_bindings={"visible": ui.command_state(...)})` applies the same bounded
model to a shell region and persists with the layout/profile.
Component events support immediate, throttled, debounced, and commit policies; deferred changes
coalesce to the latest value per stable component ID.

`app.ui.shell` exposes the versioned native application shell. It can inspect
stable built-in/extension-host node IDs and atomically show, hide, reorder, or
select supported chrome in the active project, single-viewer, or mosaic mode.
`app.ui.shell.replace_layout(...)` accepts a typed, complete keyed tree of rows,
columns, splits, tabs, panels, canvas slots, and native/extension mounts. Odon
validates and commits the whole tree atomically; Rust retains the event loop,
native rendering, platform resources, and failure recovery.
Use `app.ui.shell.list_components()` to discover supported built-in mounts and
registered extension contributions with their legal parents, modes, sizing,
commands, events, and persistence rules. `extension.register(...)` returns a
contribution whose `mount(node_id)` helper creates its keyed shell node.
Typed independently mountable native surfaces include channels, single-view viewport controls,
documentation, protected recovery, shell diagnostics, and command toolbars through
`ShellMountId.CHANNELS`, `VIEWER_VIEWPORT_CONTROLS`, `HELP`, `RECOVERY_CONTROLS`,
`SHELL_INSPECTOR`, and `COMMAND_TOOLBAR`.
The command-toolbar mount is opt-in: Odon's default toolbar presentation is empty and the default
shell layouts do not reserve space for it. Configure `app.ui.toolbars`, then include
`ShellMountId.COMMAND_TOOLBAR` in a workflow layout or saved profile. Its native buttons expose
checked and unavailable state to assistive technology and use the same command dispatcher for
mouse, keyboard, and accessibility activation.
Compatibility locations such as `right.tabs`, `top_bar.actions`, and `project.cards` are rendered
by application-owned `builtin:extension-host.*` mounts inside the same desired tree. Empty hosts
release their space, and explicitly mounted contributions are not duplicated in their default
host. `ShellMountId` exposes stable typed IDs for built-ins and these host components.
Mount `ShellMountId.SHELL_INSPECTOR` to diagnose revision, ownership, readiness, and geometry
inside Odon. `from odon import layouts` provides ready-to-apply `review()`, `analysis()`,
`comparison()`, `mosaic_triage()`, and `presentation()` trees.
Shell snapshots and component descriptors expose typed `.ownership` metadata. Application roots
and required workspaces are protected; extension sessions can mutate their own mounted nodes but
receive `PermissionDeniedError` with owner and resolution details for another extension's node.
The same enforcement now rejects application-owned topology/chrome targets from extension
sessions. Method introspection distinguishes compose, extension-placement, persistence, recovery,
chrome, window, and shortcut capability classes. `odon.connect()` and `connect_async()` explicitly
request the complete current shell-controller set by default; pass `requested_capabilities=()` or
a narrower tuple for a least-privilege/read-only or extension session. Authentication alone grants
no shell mutation authority, and the negotiated set is exposed as
`client.hello.granted_capabilities`.
Stale shell revisions report the current revision and `ui.shell.get` refetch/merge/retry strategy.
Snapshots expose actor-owned `active_region_id` and `focused_node_id`. Use
`patch_layout(active_region_id=..., focused_node_id=...)` to set them and
`patch_layout(clear_focus=True)` to clear focus. Native interaction commits produce the same
revisioned semantic changes as Python patches; split drag previews are not sent through Python.
Built-in project, viewer, and mosaic top bars are mounted inside the same tree. Reorder or hide
their nodes like other non-protected regions; menu/toolbar/status hosts receive compact intrinsic
heights unless an explicit `ShellSize` overrides them.
Each `ShellLayoutNode` also has a `configuration` mapping. Discover its schema with
`list_components()`, set it during layout construction, or update it with
`patch_layout(configurations={"layout:top": {"show_title": False}})`. Native top-bar options are
validated in Rust and omitted flags remain enabled.
Layouts can be exported/imported with `export_layout()` and `import_layout()`,
saved under session, durable application, or project-owned names with `save_profile()`, and
restored safely with `recover()`.
Application profiles are durable per OS user because they use that user's Odon configuration
directory; use project profiles for portable layouts that travel with a project.
An extension can publish a validated default with
`template = extension.register_layout("Review", document)`, rediscover it with
`extension.list_layouts()`, and apply it atomically with
`template.apply(app.ui.shell)`. These templates follow the extension's
disconnect policy and never replace the active shell implicitly.
To restore saved application layouts on the next process start, select the
profile for each desired mode with
`app.application.update_settings(shell_layout_startup_profiles={"single": "Review"})`.
Invalid or incompatible startup profiles fall back to the protected recovery
tree and are diagnosed by `app.application.get_settings()`.

Application commands are separate from their native-menu presentation. Inspect stable handlers and
availability with `app.ui.commands.list()`, then atomically reorganize the real macOS menu with
`app.ui.menus.get()` and `app.ui.menus.replace(...)`. Build nested trees with
`ui.CommandMenuNode.menu_bar()`, `.menu()`, `.command()`, and `.separator()`; always pass the
snapshot revision when collaborating with another controller. Odon preserves protected close,
quit, and recovery commands and validates the complete tree before rebuilding platform items.

Extensions that declare `ui.actions` can also own commands:

```python
extension = app.ui.register_extension(
    id="org.example.analysis",
    name="Analysis",
    version="1.0.0",
    capabilities=("ui.actions",),
    disconnect_policy="retain",
)
measure = extension.register_command(
    "measure.cells",
    "Measure Cells",
    "Measure the current selection.",
    "measure",
    modes=("single", "mosaic"),
    shortcut={"key": "m", "modifiers": ["primary", "shift"]},
    predicates=ui.CommandPredicates.predicates(
        visible=ui.CommandPredicate.capability("viewer.read"),
        enabled=ui.CommandPredicate.state(
            "selection.objects.count",
            operator="greater_than",
            value=0,
            reason="Select at least one object.",
        ),
    ),
)
app.events.subscribe("ui.extension:org.example.analysis.measure", print)
```

The returned ID is `extension:org.example.analysis/measure.cells`. It can be placed in a menu or
toolbar with `CommandMenuNode.command(...)` or `CommandToolbarItem.command(...)`, or invoked with
`app.ui.commands.execute(measure)`. Built-in native/control commands use the same method. Odon
checks mode, readiness, permission, and shortcut conflicts and owns reconnect, removal, semantic
events, and native dispatch without running Python on the GUI thread.
`primary` is portable (Command on macOS and Ctrl on Windows/Linux); effective aliases are
conflict-checked. `describe_schema()` reports the active mapping and rejects a non-realizable
platform modifier with diagnostic resolution data. Windows and Linux resolve eligible descriptors
from the latest actor projection, so changing a command or palette shortcut cannot leave a stale
OS registration.
The predicate vocabulary and allowed actor-state paths are bounded and discoverable through
`app.ui.commands.describe_schema()`. `app.ui.commands.list()` returns each command's evaluated
visibility, enablement, optional checked state, unavailable reasons, and missing capabilities for
the calling session.

The native searchable palette uses that same command catalogue and dispatcher. Its presentation is
revisioned and configurable from Python:

```python
current = app.ui.palette.get()
app.ui.palette.replace(
    ui.CommandPalette.palette(
        "palette:analysis",
        title="Analysis commands",
        placeholder="Find a command…",
        shortcut={"key": "k", "modifiers": ["primary"]},
        max_results=12,
    ),
    if_revision=current.revision,
    transaction_id="configure-analysis-palette",
)
```

Odon conflict-checks the palette shortcut and filters results by query, active mode, and extension
readiness before invoking the selected command through `ui.commands.execute`.

The control API is currently experimental. See
`docs/reference/python-api.md` for the guided introduction,
`docs/reference/python-api-reference.md` for every sync and async member,
`docs/reference/python-api-contracts.md` for behavioural guarantees, and
`docs/advanced/python-api-limitations.md` for current boundaries. The control
transport itself is documented in `docs/design/control-protocol-v1.md`.

The complete semantic surface includes projects/ROIs and saved views, image
planes and channels, native layers and NGFF labels, objects/masks/thresholds,
analysis/measurements/exports, mosaics, RAM pinning and tile policy,
screenshots, persistent settings, recent projects, and guarded application
lifecycle requests. Query `app.application.get_application_surface()` for the
machine-readable native/control/sync/async parity map.
