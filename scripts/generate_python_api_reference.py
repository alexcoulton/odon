#!/usr/bin/env python3
"""Generate the exhaustive Odon Python SDK member index.

The approachable guide and behavioural contracts remain hand-written. This
index is mechanical: it comes from the SDK signatures, central Rust command
registry, and application-surface parity manifest so public members cannot
silently drift away from the documentation.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python" / "src"
REGISTRY_CATALOG = ROOT / "src" / "control" / "registry" / "catalog.rs"
REGISTRY_SHELL_CATALOG = ROOT / "src" / "control" / "registry" / "shell_catalog.rs"
PROTOCOL_CATALOG = ROOT / "src" / "control" / "registry" / "protocol_catalog.rs"
SURFACE = ROOT / "api" / "application-surface.json"
OUTPUT = ROOT / "docs" / "reference" / "python-api-reference.md"

sys.path.insert(0, str(PYTHON_SOURCE))


@dataclass(frozen=True)
class ClassSpec:
    heading: str
    reference: str
    access: str
    purpose: str


@dataclass(frozen=True)
class FunctionSpec:
    reference: str
    purpose: str


CLASS_SPECS = (
    ClassSpec("Client", "odon.client:Client", "`odon.connect()`", "Synchronous connection and root resource container."),
    ClassSpec("Application", "odon.resources:Application", "`app.application`", "Application state, settings, navigation, lifecycle, and introspection."),
    ClassSpec("Datasets", "odon.resources:Datasets", "`app.datasets`", "Dataset inspection and opening."),
    ClassSpec("S3 datasets", "odon.resources:S3Datasets", "`app.datasets.s3`", "Authenticated S3 session, listing, and opening."),
    ClassSpec("Deep links", "odon.resources:DeepLinks", "`app.deep_links`", "Parse, generate, resolve, and apply Odon deep links."),
    ClassSpec("Viewer", "odon.resources:Viewer", "`app.viewer`", "Camera, rendering, panels, scale bar, and viewer UI state."),
    ClassSpec("Viewport workspace", "odon.resources:ViewportWorkspace", "`app.viewer.workspace`", "Native viewport layout and link configuration."),
    ClassSpec("Viewport links", "odon.resources:ViewportLinks", "`app.viewer.viewport_links`", "Canonical comparison navigation link-group resource."),
    ClassSpec("Viewports", "odon.resources:Viewports", "`app.viewer.viewports`", "Create, inspect, and address native viewports."),
    ClassSpec("Viewport handle", "odon.resources:Viewport", "returned by `app.viewer.viewports`", "Stable handle for one viewport's navigation and presentation."),
    ClassSpec("Viewport objects", "odon.resources:ViewportObjects", "`viewport.objects`", "Object presentation and filter operations bound to a stable viewport."),
    ClassSpec("Viewport comparison", "odon.resources:ViewportComparison", "returned by `app.viewer.viewports.compare()`", "Paired left and right viewport handles."),
    ClassSpec("Channels", "odon.resources:Channels", "`app.channels` or `app.viewer.channels`", "Channel visibility, presentation, contrast, transforms, and groups."),
    ClassSpec("Planes", "odon.resources:Planes", "`app.planes` or `app.viewer.planes`", "Multidimensional orientation and slice navigation."),
    ClassSpec("Native layers", "odon.resources:NativeLayers", "`app.native_layers` or `app.viewer.native_layers`", "Odon-owned image, channel, object, and mask layer controls."),
    ClassSpec("Projects", "odon.resources:Projects", "`app.projects`", "Project creation, persistence, metadata, and ROI opening."),
    ClassSpec("Project samplesheets", "odon.resources:ProjectSamplesheets", "`app.projects.samplesheets`", "Samplesheet inspection, validation, import, and export."),
    ClassSpec("Project discovery", "odon.resources:ProjectDiscovery", "`app.projects.discovery`", "Project search-root discovery."),
    ClassSpec("Project object preload", "odon.resources:ProjectObjects", "`app.projects.objects`", "Project-wide object preload lifecycle."),
    ClassSpec("Project ROIs", "odon.resources:ProjectRois", "`app.projects.rois`", "ROI CRUD, ordering, selection, focus, and opening."),
    ClassSpec("Project views", "odon.resources:ProjectViews", "`app.projects.views`", "Saved view creation, capture, mutation, and application."),
    ClassSpec("Screenshots", "odon.resources:Screenshots", "`app.screenshots`", "Viewer, window, and project screenshots plus retained settings."),
    ClassSpec("Labels", "odon.resources:Labels", "`app.labels` or `app.viewer.labels`", "OME-NGFF label discovery, loading, and visibility."),
    ClassSpec("Memory", "odon.resources:Memory", "`app.memory` or `app.viewer.memory`", "RAM pinning estimates and tile-loading policy."),
    ClassSpec("Objects", "odon.resources:Objects", "`app.objects` or `app.viewer.objects`", "Object source, styling, properties, filters, spatial queries, selection, and focus."),
    ClassSpec("Annotations", "odon.resources:Annotations", "`app.annotations` or `app.viewer.annotations`", "Actor-owned point annotation layers, Parquet sources, schema inspection, and styling."),
    ClassSpec("Masks", "odon.resources:Masks", "`app.masks` or `app.viewer.masks`", "Mask layer and polygon editing, selection, history, persistence, and GeoJSON interchange."),
    ClassSpec("Thresholds", "odon.resources:Thresholds", "`app.thresholds` or `app.viewer.thresholds`", "Threshold preview configuration and polygon creation."),
    ClassSpec("Analysis", "odon.resources:Analysis", "`app.analysis` or `app.viewer.analysis`", "Object analysis state, histograms, suggestions, presets, and warmup."),
    ClassSpec("Measurements", "odon.resources:Measurements", "`app.measurements` or `app.viewer.measurements`", "Polygon intensity measurement configuration and execution."),
    ClassSpec("Object exports", "odon.resources:ObjectExports", "`app.object_exports` or `app.objects.exports`", "Column discovery and scoped object exports."),
    ClassSpec("Mosaic", "odon.resources:Mosaic", "`app.mosaic`", "Mosaic items, layout, selection, focus, object loading, and UI state."),
    ClassSpec("Data resources", "odon.data:DataResources", "`app.data`", "Register and manage large external data by reference."),
    ClassSpec("Data resource handle", "odon.data:DataResource", "returned by `app.data`", "Mutable handle for one registered external data descriptor."),
    ClassSpec("Coordinate space", "odon.data:CoordinateSpace", "construct directly", "Validated pixel/world coordinate metadata and conversion."),
    ClassSpec("External layers", "odon.layers:Layers", "`app.layers` or `app.viewer.layers`", "Add and manage stable external viewer layers."),
    ClassSpec("External layer handle", "odon.layers:Layer", "returned by `app.layers`", "Mutable handle for one external viewer layer."),
    ClassSpec("Tasks", "odon.tasks:Tasks", "`app.tasks`", "Start, inspect, list, cancel, and forget retained work."),
    ClassSpec("Task handle", "odon.tasks:Task", "returned by task-starting methods", "Wait for and manage one retained operation."),
    ClassSpec("Events", "odon.events:Events", "`app.events`", "Synchronous subscriptions, callbacks, and queued event consumption."),
    ClassSpec("Declarative UI registry", "odon.ui:Ui", "`app.ui`", "Register extensions, inspect schemas, and list component contributions."),
    ClassSpec("Application shell", "odon.ui:Shell", "`app.ui.shell`", "Inspect, reorder, select, show, hide, and reset native shell nodes."),
    ClassSpec("Application commands", "odon.ui:Commands", "`app.ui.commands`", "Discover stable application command descriptors independently of their presentations."),
    ClassSpec("Application command", "odon.ui:ApplicationCommand", "returned by `app.ui.commands.list`", "Typed command identity, handler, availability, protection, icon, and shortcut metadata."),
    ClassSpec("Command predicate", "odon.ui:CommandPredicate", "configure extension commands", "A bounded actor-evaluated capability or application-state condition."),
    ClassSpec("Command predicate slots", "odon.ui:CommandPredicates", "configure extension commands", "Declarative visible, enabled, and checked conditions shared by every command presentation."),
    ClassSpec("Platform menus", "odon.ui:Menus", "`app.ui.menus`", "Inspect and revision-guard the declarative platform application menu."),
    ClassSpec("Platform menu snapshot", "odon.ui:CommandMenuSnapshot", "returned by `app.ui.menus`", "Typed revisioned platform-menu presentation snapshot."),
    ClassSpec("Platform menu node", "odon.ui:CommandMenuNode", "build platform menu trees", "A menu bar, nested menu, command presentation, or separator."),
    ClassSpec("Command toolbars", "odon.ui:Toolbars", "`app.ui.toolbars`", "Inspect and revision-guard the declarative application command toolbar."),
    ClassSpec("Command toolbar snapshot", "odon.ui:CommandToolbarSnapshot", "returned by `app.ui.toolbars`", "Typed revisioned command-toolbar presentation snapshot."),
    ClassSpec("Command toolbar", "odon.ui:CommandToolbar", "build a command toolbar", "A bounded toolbar presentation containing ordered groups."),
    ClassSpec("Command toolbar group", "odon.ui:CommandToolbarGroup", "contained by `CommandToolbar`", "One labelled or unlabelled group of command presentations."),
    ClassSpec("Command toolbar item", "odon.ui:CommandToolbarItem", "contained by `CommandToolbarGroup`", "A toolbar presentation referencing one stable command ID."),
    ClassSpec("Command palette resource", "odon.ui:Palette", "`app.ui.palette`", "Inspect and revision-guard the searchable command-palette presentation."),
    ClassSpec("Command palette snapshot", "odon.ui:CommandPaletteSnapshot", "returned by `app.ui.palette`", "Typed revisioned command-palette presentation snapshot."),
    ClassSpec("Command palette", "odon.ui:CommandPalette", "configure `app.ui.palette`", "Palette title, prompt, shortcut, description visibility, and bounded result count."),
    ClassSpec("Application shell snapshot", "odon.ui:ShellSnapshot", "returned by `app.ui.shell`", "Typed, mapping-compatible versioned shell snapshot."),
    ClassSpec("Application shell node", "odon.ui:ShellNode", "contained by `ShellSnapshot`", "Typed native or extension-host shell node."),
    ClassSpec("Application shell mutability", "odon.ui:ShellMutability", "`ShellNode.mutable`", "Per-property native shell mutation capabilities."),
    ClassSpec("Application shell change", "odon.ui:ShellChange", "`ShellSnapshot.change`", "Old/new revisions and property-level mutation results."),
    ClassSpec("Application shell property change", "odon.ui:ShellPropertyChange", "contained by `ShellChange`", "One changed shell node property with before/after values."),
    ClassSpec("Application shell IDs", "odon.ui:ShellId", "use in shell patches", "Stable schema-version-1 built-in and extension-host IDs."),
    ClassSpec("Application shell mount IDs", "odon.ui:ShellMountId", "use in `ShellLayoutNode` builders", "Stable built-in component and application-owned extension-host mount IDs."),
    ClassSpec("Application shell desired layout", "odon.ui:ShellLayout", "submit to `app.ui.shell.replace_layout`", "A complete validated keyed application layout tree."),
    ClassSpec("Application shell layout document", "odon.ui:ShellLayoutDocument", "returned by `app.ui.shell.export_layout`", "A portable versioned layout document for import, migration, and recovery workflows."),
    ClassSpec("Application shell layout profile", "odon.ui:ShellLayoutProfile", "returned by `app.ui.shell.list_profiles`", "Metadata for one named session, application, or project layout."),
    ClassSpec("Extension layout template", "odon.ui:ExtensionLayoutTemplate", "returned by `extension.register_layout()`", "A canonical version-1 default layout owned by one extension."),
    ClassSpec("Application shell layout node", "odon.ui:ShellLayoutNode", "contained by `ShellLayout`", "A row, column, split, tabs, panel, canvas, built-in, or extension mount node."),
    ClassSpec("Application shell layout node types", "odon.ui:ShellLayoutType", "use in `ShellLayoutNode`", "Supported desired-layout node kinds."),
    ClassSpec("Application shell ownership", "odon.ui:ShellOwnership", "available on shell nodes and component descriptors", "Server-derived application/extension owner identity, session identity, and protected status."),
    ClassSpec("Application shell mount readiness", "odon.ui:ShellMountReadiness", "available on extension `ShellLayoutNode` instances", "Ready, not-ready, disconnected, incompatible, or missing retained extension-mount state."),
    ClassSpec("Application shell size", "odon.ui:ShellSize", "use in `ShellLayoutNode`", "Advisory desired, minimum, maximum, and flex sizing."),
    ClassSpec("Application shell split", "odon.ui:ShellSplit", "use in split layout nodes", "Validated split ratio and native-resize behavior."),
    ClassSpec("Application shell component descriptor", "odon.ui:ShellComponentDescriptor", "returned by `app.ui.shell.list_components`", "Introspected built-in mount compatibility, sizing, commands, events, and persistence."),
    ClassSpec("UI extension handle", "odon.ui:Extension", "returned by `app.ui.register_extension()`", "Register component trees owned by one extension."),
    ClassSpec("UI contribution handle", "odon.ui:Contribution", "returned by `extension.register()`", "Patch or remove one retained component tree."),
    ClassSpec("UI component base", "odon.ui:Component", "construct through component subclasses", "Serializable native-egui component contract."),
    ClassSpec("UI event policy", "odon.ui:EventPolicy", "`Immediate`, `OnCommit`, `Throttle`, or `Debounce`", "Controls native widget event delivery."),
    ClassSpec("Extension lifecycle protocol", "odon.extensions:Extension", "implemented by packaged extensions", "Optional cleanup protocol used by the reconnecting extension runner."),
    ClassSpec("Instance", "odon.discovery:Instance", "returned by discovery functions", "Discovered Odon endpoint and authentication metadata."),
    ClassSpec("Hello", "odon.models:Hello", "`app.hello`", "Negotiated connection and capability metadata."),
    ClassSpec("Event", "odon.models:Event", "returned by `app.events`", "Ordered semantic event envelope."),
    ClassSpec("Task snapshot", "odon.models:TaskSnapshot", "`task.snapshot`", "Immutable retained-task state."),
)

ASYNC_CLASS_SPECS = (
    ClassSpec("Async client", "odon.async_client:AsyncClient", "`await odon.connect_async()`", "Async connection and root resource container."),
    ClassSpec("Async application", "odon.async_resources:AsyncApplication", "`app.application`", "Async application resource."),
    ClassSpec("Async datasets", "odon.async_resources:AsyncDatasets", "`app.datasets`", "Async dataset resource."),
    ClassSpec("Async S3 datasets", "odon.async_resources:AsyncS3Datasets", "`app.datasets.s3`", "Async S3 dataset resource."),
    ClassSpec("Async deep links", "odon.async_resources:AsyncDeepLinks", "`app.deep_links`", "Async deep-link resource."),
    ClassSpec("Async viewer", "odon.async_resources:AsyncViewer", "`app.viewer`", "Async viewer resource."),
    ClassSpec("Async viewport workspace", "odon.async_resources:AsyncViewportWorkspace", "`app.viewer.workspace`", "Async native viewport layout and link resource."),
    ClassSpec("Async viewport links", "odon.async_resources:AsyncViewportLinks", "`app.viewer.viewport_links`", "Async comparison navigation link-group resource."),
    ClassSpec("Async viewports", "odon.async_resources:AsyncViewports", "`app.viewer.viewports`", "Async native viewport collection."),
    ClassSpec("Async viewport handle", "odon.async_resources:AsyncViewport", "returned by `app.viewer.viewports`", "Stable asynchronous viewport handle."),
    ClassSpec("Async viewport objects", "odon.async_resources:AsyncViewportObjects", "`viewport.objects`", "Async object presentation and filter operations bound to a stable viewport."),
    ClassSpec("Async viewport comparison", "odon.async_resources:AsyncViewportComparison", "returned by `app.viewer.viewports.compare()`", "Paired asynchronous viewport handles."),
    ClassSpec("Async channels", "odon.async_resources:AsyncChannels", "`app.channels`", "Async channel resource."),
    ClassSpec("Async planes", "odon.async_resources:AsyncPlanes", "`app.planes`", "Async plane resource."),
    ClassSpec("Async native layers", "odon.async_resources:AsyncNativeLayers", "`app.native_layers`", "Async native-layer resource."),
    ClassSpec("Async projects", "odon.async_resources:AsyncProjects", "`app.projects`", "Async project resource."),
    ClassSpec("Async project samplesheets", "odon.async_resources:AsyncProjectSamplesheets", "`app.projects.samplesheets`", "Async samplesheet resource."),
    ClassSpec("Async project discovery", "odon.async_resources:AsyncProjectDiscovery", "`app.projects.discovery`", "Async project discovery resource."),
    ClassSpec("Async project object preload", "odon.async_resources:AsyncProjectObjects", "`app.projects.objects`", "Async object preload resource."),
    ClassSpec("Async project ROIs", "odon.async_resources:AsyncProjectRois", "`app.projects.rois`", "Async ROI resource."),
    ClassSpec("Async project views", "odon.async_resources:AsyncProjectViews", "`app.projects.views`", "Async saved-view resource."),
    ClassSpec("Async screenshots", "odon.async_resources:AsyncScreenshots", "`app.screenshots`", "Async screenshot resource."),
    ClassSpec("Async labels", "odon.async_resources:AsyncLabels", "`app.labels`", "Async label resource."),
    ClassSpec("Async memory", "odon.async_resources:AsyncMemory", "`app.memory`", "Async memory resource."),
    ClassSpec("Async objects", "odon.async_resources:AsyncObjects", "`app.objects`", "Async object resource."),
    ClassSpec("Async annotations", "odon.async_resources:AsyncAnnotations", "`app.annotations`", "Async point annotation resource."),
    ClassSpec("Async masks", "odon.async_resources:AsyncMasks", "`app.masks`", "Async mask resource."),
    ClassSpec("Async thresholds", "odon.async_resources:AsyncThresholds", "`app.thresholds`", "Async threshold resource."),
    ClassSpec("Async analysis", "odon.async_resources:AsyncAnalysis", "`app.analysis`", "Async analysis resource."),
    ClassSpec("Async measurements", "odon.async_resources:AsyncMeasurements", "`app.measurements`", "Async measurement resource."),
    ClassSpec("Async object exports", "odon.async_resources:AsyncObjectExports", "`app.object_exports`", "Async object-export resource."),
    ClassSpec("Async mosaic", "odon.async_resources:AsyncMosaic", "`app.mosaic`", "Async mosaic resource."),
    ClassSpec("Async data resources", "odon.async_data:AsyncDataResources", "`app.data`", "Async external-data resource."),
    ClassSpec("Async data handle", "odon.async_data:AsyncDataResource", "returned by `app.data`", "Async external-data handle."),
    ClassSpec("Async external layers", "odon.async_layers:AsyncLayers", "`app.layers`", "Async external-layer resource."),
    ClassSpec("Async external layer handle", "odon.async_layers:AsyncLayer", "returned by `app.layers`", "Async external-layer handle."),
    ClassSpec("Async tasks", "odon.async_tasks:AsyncTasks", "`app.tasks`", "Async retained-task collection."),
    ClassSpec("Async task handle", "odon.async_tasks:AsyncTask", "returned by task-starting methods", "Awaitable retained-task handle."),
    ClassSpec("Async events", "odon.async_events:AsyncEvents", "`app.events`", "Async subscriptions and iteration."),
    ClassSpec("Async UI registry", "odon.async_ui:AsyncUi", "`app.ui`", "Async declarative UI registry."),
    ClassSpec("Async application shell", "odon.async_ui:AsyncShell", "`app.ui.shell`", "Async native application-shell composition."),
    ClassSpec("Async application commands", "odon.async_ui:AsyncCommands", "`app.ui.commands`", "Async command discovery."),
    ClassSpec("Async platform menus", "odon.async_ui:AsyncMenus", "`app.ui.menus`", "Async revision-guarded platform-menu composition."),
    ClassSpec("Async command toolbars", "odon.async_ui:AsyncToolbars", "`app.ui.toolbars`", "Async revision-guarded command-toolbar composition."),
    ClassSpec("Async command palette", "odon.async_ui:AsyncPalette", "`app.ui.palette`", "Async revision-guarded command-palette composition."),
    ClassSpec("Async UI extension", "odon.async_ui:AsyncExtension", "returned by `app.ui.register_extension()`", "Async UI extension handle."),
    ClassSpec("Async UI contribution", "odon.async_ui:AsyncContribution", "returned by `extension.register()`", "Async UI contribution handle."),
)

FUNCTION_SPECS = (
    FunctionSpec("odon.client:connect", "Connect synchronously to a running Odon instance."),
    FunctionSpec("odon.async_client:connect_async", "Create an asynchronous Odon connection context."),
    FunctionSpec("odon.discovery:list_instances", "List discoverable authenticated local Odon instances."),
    FunctionSpec("odon.discovery:select_instance", "Resolve an instance selector or require an unambiguous instance."),
    FunctionSpec("odon.launch:launch", "Launch an installed Odon executable and connect synchronously."),
    FunctionSpec("odon.launch:launch_async", "Launch an installed Odon executable and connect asynchronously."),
    FunctionSpec("odon.extensions:run", "Run and optionally reconnect a packaged Python extension."),
    FunctionSpec("odon.ui:emit", "Create an action that emits an extension event to Python."),
    FunctionSpec("odon.ui:command", "Create an action that invokes a validated native command."),
    FunctionSpec("odon.ui:bind", "Create a native state binding action."),
    FunctionSpec("odon.layouts:review", "Build a reusable single-view review shell."),
    FunctionSpec("odon.layouts:analysis", "Build a reusable single-view analysis shell."),
    FunctionSpec("odon.layouts:comparison", "Build a shell around the native comparison workspace."),
    FunctionSpec("odon.layouts:mosaic_triage", "Build a reusable mosaic-triage shell."),
    FunctionSpec("odon.layouts:presentation", "Build a canvas-first presentation shell."),
)

UI_COMPONENTS = (
    "Container", "Panel", "Column", "Row", "Grid", "Tabs", "Scroll", "Group", "Collapsible",
    "Text", "Markdown", "Status", "Warning", "Error", "Spinner", "Button", "Toggle",
    "Checkbox", "Slider", "Number", "Integer", "TextInput", "Select", "Radio",
    "MultiSelect", "Color", "Progress", "Separator", "Spacer", "Immediate", "OnCommit",
    "Throttle", "Debounce",
)

ERROR_NAMES = (
    "OdonError", "ConnectionClosedError", "ProtocolError", "RequestTimeoutError",
    "TaskCancelledError", "TaskFailedError", "InstanceNotFoundError",
    "MultipleInstancesError", "RemoteError", "AuthenticationError",
    "ProtocolVersionError", "InvalidParametersError", "ResourceNotFoundError",
    "NotReadyError", "UnsupportedCapabilityError", "ConflictError",
    "PermissionDeniedError", "WrongModeError", "ResourceLimitError",
)


def completion_contract(name: str, starts_task: bool) -> str:
    """Mirror the central Rust registry's public completion classification."""

    if name in {
        "viewer.screenshot.capture",
        "viewer.workspace.screenshot.capture",
        "app.screenshot.capture",
        "project.screenshot.capture",
        "exports.canvas.capture",
    }:
        return "presentation_dependent"
    if starts_task:
        return "retained_background"
    if (
        name == "viewer.screenshot.settings.set"
        or name.startswith("data.resources.")
        or name.startswith("viewer.layers.")
        or name.startswith("datasets.open_")
        or name in {"project.open", "project.save", "project.save_as"}
        or name.endswith(".load")
        or name.endswith(".reload")
        or "preload" in name
        or name.startswith("exports.")
        or name.startswith("viewer.measurements.")
        or name.startswith("viewer.analysis.warmup.")
        or name.startswith("memory.pin")
    ):
        return "resource_ready"
    return "immediate_semantic"

DOCUMENTED_CLASS_MODULES = (
    "odon.client", "odon.async_client", "odon.resources", "odon.async_resources",
    "odon.data", "odon.async_data", "odon.layers", "odon.async_layers", "odon.tasks",
    "odon.async_tasks", "odon.events", "odon.async_events", "odon.ui", "odon.async_ui",
    "odon.discovery", "odon.extensions", "odon.models", "odon.errors",
)


def resolve(reference: str) -> Any:
    module_name, name = reference.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for part in name.split("."):
        value = getattr(value, part)
    return value


def signature(value: Any, *, drop_self: bool = False) -> str:
    result = str(inspect.signature(value))
    if drop_self:
        result = re.sub(r"^\((?:self|cls)(?:, )?", "(", result)
    return result


def markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def first_line(value: Any) -> str | None:
    doc = inspect.getdoc(value)
    return doc.splitlines()[0].strip() if doc else None


def direct_methods(value: Any) -> tuple[str, ...]:
    """Return literal protocol methods called directly by a Python wrapper."""

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
    except (OSError, TypeError, SyntaxError):
        return ()
    methods: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr not in {"call", "start"}:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            if "." in argument.value and argument.value not in methods:
                methods.append(argument.value)
    return tuple(methods)


def registry_methods() -> dict[str, dict[str, Any]]:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REGISTRY_CATALOG, REGISTRY_SHELL_CATALOG)
    )
    protocol_source = PROTOCOL_CATALOG.read_text(encoding="utf-8")
    modes = {
        "ALL_MODES": "project, single, mosaic, transition",
        "READY_MODES": "project, single, mosaic",
        "VIEWER_MODES": "single, mosaic",
        "SINGLE_MODE": "single",
        "MOSAIC_MODE": "mosaic",
    }
    result: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r'method!\(\s*"(?P<name>[^"]+)",\s*"(?P<summary>[^"]+)"'
        r',\s*"(?P<capability>[^"]+)"\s*,\s*(?P<mutates>true|false)'
        r'\s*,\s*(?P<task>true|false)\s*,\s*(?P<event>None|Some\("[^"]+"\))'
        r'\s*,\s*(?P<modes>[A-Z_]+)\s*,',
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        event_value = match.group("event")
        result[match.group("name")] = {
            "summary": match.group("summary"),
            "capability": match.group("capability"),
            "mutates": match.group("mutates") == "true",
            "task": match.group("task") == "true",
            "event": None if event_value == "None" else event_value[6:-2],
            "modes": modes.get(match.group("modes"), match.group("modes")),
        }

    custom = re.compile(
        r'MethodDescriptor \{\s*name: "(?P<name>[^"]+)"'
        r',\s*summary: "(?P<summary>[^"]+)"'
        r',\s*capability: "(?P<capability>[^"]+)"'
        r',\s*mutates: (?P<mutates>true|false)'
        r',\s*starts_task: (?P<task>true|false).*?'
        r'event: (?P<event>None|Some\("[^"]+"\))'
        r',\s*available_in: (?P<modes>[A-Z_]+)',
        re.DOTALL,
    )
    for match in custom.finditer(source):
        event_value = match.group("event")
        result[match.group("name")] = {
            "summary": match.group("summary"),
            "capability": match.group("capability"),
            "mutates": match.group("mutates") == "true",
            "task": match.group("task") == "true",
            "event": None if event_value == "None" else event_value[6:-2],
            "modes": modes.get(match.group("modes"), match.group("modes")),
        }

    protocol_block = protocol_source.split("pub static PROTOCOL_METHODS", 1)[1].split("];", 1)[0]
    protocol_pattern = re.compile(
        r'\(\s*"(?P<name>[^"]+)",\s*"(?P<summary>[^"]+)"'
        r',\s*"(?P<capability>[^"]+)"\s*,\s*(?P<mutates>true|false)'
        r'\s*,\s*(?P<task>true|false)\s*,?\s*\)',
        re.DOTALL,
    )
    for match in protocol_pattern.finditer(protocol_block):
        result[match.group("name")] = {
            "summary": match.group("summary"),
            "capability": match.group("capability"),
            "mutates": match.group("mutates") == "true",
            "task": match.group("task") == "true",
            "event": None,
            "modes": "protocol",
        }
    for name, descriptor in result.items():
        descriptor["completion"] = completion_contract(name, descriptor["task"])
        descriptor["cancellation"] = (
            "cooperative" if descriptor["task"] else "not_applicable"
        )
    return result


def surface_references() -> dict[str, dict[str, Any]]:
    manifest = json.loads(SURFACE.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for entry in manifest["entries"]:
        for key in ("python_sync", "python_async"):
            references = entry.get(key, [])
            methods = entry.get("methods", [])
            for index, reference in enumerate(references):
                method = methods[index] if len(methods) == len(references) else None
                result[reference] = {
                    "title": entry["title"],
                    "method": method,
                    "events": entry.get("events", []),
                    "permissions": entry.get("permissions", []),
                    "status": entry["status"],
                }
    return result


def public_members(cls: type[Any]) -> Iterable[tuple[str, Any]]:
    for name, raw in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(raw, (staticmethod, classmethod)):
            value = raw.__func__
        else:
            value = raw
        if callable(value) or isinstance(value, property):
            yield name, value


def validate_coverage(registry: dict[str, dict[str, Any]]) -> None:
    documented = {spec.reference for spec in CLASS_SPECS + ASYNC_CLASS_SPECS}
    documented.update(f"odon.ui:{name}" for name in UI_COMPONENTS)
    documented.update(f"odon.errors:{name}" for name in ERROR_NAMES)
    missing: list[str] = []
    for module_name in DOCUMENTED_CLASS_MODULES:
        module = importlib.import_module(module_name)
        for name, value in inspect.getmembers(module, inspect.isclass):
            if value.__module__ != module_name or name.startswith("_"):
                continue
            reference = f"{module_name}:{name}"
            if reference not in documented:
                missing.append(reference)
    if missing:
        raise RuntimeError("undocumented public SDK classes: " + ", ".join(missing))

    unknown: dict[str, tuple[str, ...]] = {}
    for spec in CLASS_SPECS + ASYNC_CLASS_SPECS:
        cls = resolve(spec.reference)
        for name, value in public_members(cls):
            if isinstance(value, property):
                continue
            methods = tuple(method for method in direct_methods(value) if method not in registry)
            if methods:
                unknown[f"{cls.__module__}:{cls.__name__}.{name}"] = methods
    if unknown:
        details = ", ".join(f"{member} -> {methods}" for member, methods in unknown.items())
        raise RuntimeError("Python wrappers use unregistered control methods: " + details)


def method_row(
    cls: type[Any],
    name: str,
    value: Any,
    registry: dict[str, dict[str, Any]],
    surface: dict[str, dict[str, Any]],
) -> str:
    qualified = f"{cls.__module__}:{cls.__name__}.{name}"
    parity = surface.get(qualified, {})
    if isinstance(value, property):
        member_signature = " (property)"
        protocols: tuple[str, ...] = ()
        description = first_line(value.fget) if value.fget is not None else None
    else:
        member_signature = signature(value, drop_self=True)
        protocols = direct_methods(value)
        description = first_line(value)
    if not protocols and parity.get("method"):
        protocols = (parity["method"],)
    descriptors = [registry[method] for method in protocols if method in registry]
    if description is None and len(descriptors) == 1:
        description = descriptors[0]["summary"]
    if description is None:
        description = parity.get("title")
    if description is None:
        description = f"{name.replace('_', ' ').capitalize()}."
    rpc = ", ".join(f"`{method}`" for method in protocols) or "SDK-local/delegated"
    modes = "; ".join(dict.fromkeys(item["modes"] for item in descriptors)) or "Inherited from delegated operation"
    flags: list[str] = []
    if any(item["mutates"] for item in descriptors):
        flags.append("mutates")
    if any(item["task"] for item in descriptors):
        flags.append("task")
    completions = list(dict.fromkeys(item["completion"] for item in descriptors))
    if completions:
        flags.append("completion: " + ", ".join(completions))
    if any(item["cancellation"] == "cooperative" for item in descriptors):
        flags.append("cancellation: cooperative")
    events = [item["event"] for item in descriptors if item.get("event")]
    if events:
        flags.append("event: " + ", ".join(events))
    detail = description + (f" ({'; '.join(flags)})" if flags else "")
    return (
        f"| `{name}{markdown(member_signature)}` | {markdown(rpc)} | "
        f"{markdown(modes)} | {markdown(detail)} |"
    )


def render_class(
    spec: ClassSpec,
    registry: dict[str, dict[str, Any]],
    surface: dict[str, dict[str, Any]],
) -> list[str]:
    cls = resolve(spec.reference)
    lines = [
        f"### {spec.heading}",
        "",
        f"Access: {spec.access}. {spec.purpose}",
        "",
    ]
    if "construct directly" in spec.access or spec.reference.startswith("odon.ui:"):
        lines.extend([f"Constructor: `{cls.__name__}{markdown(signature(cls))}`", ""])
    members = list(public_members(cls))
    if not members:
        lines.extend(["This value type has no public methods beyond its fields.", ""])
        return lines
    lines.extend([
        "| Member | Control method | Modes | Contract |",
        "| --- | --- | --- | --- |",
    ])
    lines.extend(method_row(cls, name, value, registry, surface) for name, value in members)
    lines.append("")
    return lines


def generate() -> str:
    registry = registry_methods()
    validate_coverage(registry)
    surface = surface_references()
    lines = [
        "# Python API member reference",
        "",
        "<!-- Generated by scripts/generate_python_api_reference.py. Do not edit manually. -->",
        "",
        "This is the exhaustive member index for the checked-in Python SDK. It is generated",
        "from Python signatures, the Rust control registry, and the application-surface",
        "manifest. See [Python API](python-api.md) for the guided introduction and",
        "[Python API contracts](python-api-contracts.md) for behavioural and response",
        "contracts.",
        "",
        "Signatures omit `self`. `task` means the operation returns a retained `Task` or",
        "`AsyncTask`; it does not block Odon's GUI. `protocol` mode denotes transport and",
        "extension-registry operations that are not tied to the project/single/mosaic view",
        "mode. Availability is still capability- and readiness-checked at runtime.",
        "",
        "## Module functions",
        "",
        "| Function | Contract |",
        "| --- | --- |",
    ]
    for spec in FUNCTION_SPECS:
        function = resolve(spec.reference)
        module, name = spec.reference.split(":", 1)
        lines.append(
            f"| `{module}.{name}{markdown(signature(function))}` | {markdown(spec.purpose)} |"
        )
    lines.extend(["", "## Synchronous API", ""])
    for spec in CLASS_SPECS:
        lines.extend(render_class(spec, registry, surface))

    lines.extend([
        "## UI component constructors",
        "",
        "All components inherit `Component.to_dict()` and are validated by Rust before",
        "registration. Exact constructor signatures are listed here because these classes",
        "are normally instantiated directly.",
        "",
        "| Component | Constructor |",
        "| --- | --- |",
    ])
    ui_module = importlib.import_module("odon.ui")
    for name in UI_COMPONENTS:
        component = getattr(ui_module, name)
        lines.append(f"| `{name}` | `{name}{markdown(signature(component))}` |")
    lines.extend(["", "## Asynchronous API", ""])
    lines.extend([
        "Async resources have the same semantic contracts as their synchronous peers.",
        "Network methods are coroutines and therefore require `await`; retained",
        "`AsyncTask` handles are themselves awaitable.",
        "",
    ])
    for spec in ASYNC_CLASS_SPECS:
        lines.extend(render_class(spec, registry, surface))

    lines.extend([
        "## Error classes",
        "",
        "All SDK errors derive from `odon.OdonError`.",
        "",
        "| Error | Meaning |",
        "| --- | --- |",
    ])
    errors = importlib.import_module("odon.errors")
    for name in ERROR_NAMES:
        value = getattr(errors, name)
        description = first_line(value) or name.replace("Error", " error").replace("Task", "Task ")
        lines.append(f"| `{name}` | {markdown(description)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the generated file is stale")
    args = parser.parse_args()
    content = generate()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != content:
            print(f"{OUTPUT.relative_to(ROOT)} is stale; run {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
