use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stability {
    Experimental,
    Provisional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionClass {
    Model,
    Geometry,
    Resource,
    Presentation,
    ExternalCompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOwner {
    Actor,
    LegacyUi,
    ControlService,
    Unavailable,
}

impl ExecutionOwner {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Actor => "actor",
            Self::LegacyUi => "legacy_ui",
            Self::ControlService => "control_service",
            Self::Unavailable => "unavailable",
        }
    }
}

impl ExecutionClass {
    pub fn readiness_requirements(self) -> &'static [&'static str] {
        match self {
            Self::Model => &["model"],
            Self::Geometry => &["model", "geometry"],
            Self::Resource => &["model", "resources"],
            Self::Presentation => &["model", "presentation", "output"],
            Self::ExternalCompute => &["model", "resources"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestShape {
    Empty,
    SetSidePanels,
    SetSmoothPixels,
    SetVisibleChannels,
    SetCamera,
    CaptureScreenshot,
    AppSettings,
    LifecycleRequest,
    SetScaleBar,
    ScreenshotSettings,
    TileLoading,
    MemoryPin,
    MemoryUnpin,
    LabelLoad,
    LabelVisibility,
    ChannelPresentation,
    MethodAvailability,
    SetPlane,
    StepPlane,
    SetChannelColor,
    SetChannelNote,
    SetChannelTransform,
    NativeLayerSelector,
    NativeLayerVisibility,
    NativeLayerOrder,
    NativeLayerOffset,
    ProjectViewSelector,
    ProjectViewCreate,
    ProjectViewCapture,
    ProjectViewRename,
    ProjectCreate,
    Path,
    ProjectMetadata,
    SamplesheetInspect,
    SamplesheetExport,
    SpatialDataOpen,
    XeniumOpen,
    HttpOpen,
    S3Session,
    S3Prefix,
    TiffOpen,
    ProjectRoiId,
    ProjectRoiAdd,
    ProjectRoiUpdate,
    ProjectRoiOrder,
    ProjectRoiSelect,
    MosaicFocus,
    MosaicItems,
    MosaicSelect,
    MosaicLayout,
    ObjectPreloadStart,
    DeepLinkUri,
    DeepLinkGenerate,
    DeepLinkApply,
    Object,
}

#[derive(Debug, Clone, Copy)]
pub struct MethodDescriptor {
    pub name: &'static str,
    pub summary: &'static str,
    pub capability: &'static str,
    pub mutates: bool,
    pub starts_task: bool,
    pub mcp_exposed: bool,
    pub stability: Stability,
    pub request_shape: RequestShape,
    pub event: Option<&'static str>,
    pub available_in: &'static [&'static str],
    pub since: &'static str,
    pub execution_class: ExecutionClass,
}

pub fn execution_owner(
    descriptor: &MethodDescriptor,
    mode: &str,
    _params: &Value,
    _project_view_requires_resource_load: bool,
) -> ExecutionOwner {
    if !descriptor.available_in.contains(&mode) {
        return ExecutionOwner::Unavailable;
    }
    if !ACTOR_CAPABLE_METHODS.contains(&descriptor.name) {
        return ExecutionOwner::LegacyUi;
    }
    if mode == "mosaic"
        && (descriptor.name.starts_with("viewer.") || descriptor.name.starts_with("memory."))
        && !is_actor_owned_mosaic_shared_method(descriptor.name)
    {
        return ExecutionOwner::LegacyUi;
    }
    ExecutionOwner::Actor
}

pub fn execution_route_summary(descriptor: &MethodDescriptor) -> &'static str {
    let actor_capable = ACTOR_CAPABLE_METHODS.contains(&descriptor.name);
    if !actor_capable {
        return "legacy_ui";
    }
    if (descriptor.name.starts_with("viewer.") || descriptor.name.starts_with("memory."))
        && descriptor.available_in.contains(&"mosaic")
        && !is_actor_owned_mosaic_shared_method(descriptor.name)
    {
        "hybrid"
    } else {
        "actor"
    }
}

pub fn execution_route_json(descriptor: &MethodDescriptor) -> Value {
    let actor_capable = ACTOR_CAPABLE_METHODS.contains(&descriptor.name);
    let variants = json!([]);
    let by_mode = ["project", "single", "mosaic", "transition"]
        .into_iter()
        .map(|mode| {
            let owner = if !descriptor.available_in.contains(&mode) {
                ExecutionOwner::Unavailable
            } else if !actor_capable
                || (mode == "mosaic"
                    && (descriptor.name.starts_with("viewer.")
                        || descriptor.name.starts_with("memory."))
                    && !is_actor_owned_mosaic_shared_method(descriptor.name))
            {
                ExecutionOwner::LegacyUi
            } else {
                ExecutionOwner::Actor
            };
            (
                mode.to_string(),
                json!({
                    "default_owner":owner,
                    "conditional": false,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    json!({
        "summary":execution_route_summary(descriptor),
        "by_mode":by_mode,
        "variants":variants,
    })
}

pub fn is_actor_owned_mosaic_shared_method(method: &str) -> bool {
    matches!(
        method,
        "app.get_state"
            | "viewer.channels.list"
            | "viewer.channels.list_visible"
            | "viewer.channels.get_active"
            | "viewer.channels.set_active"
            | "viewer.channels.set_visible"
            | "viewer.channels.get_contrast"
            | "viewer.channels.set_contrast"
            | "viewer.channels.set_color"
            | "viewer.channels.set_note"
            | "viewer.channels.set_order"
            | "viewer.channels.presentation.get"
            | "viewer.channels.presentation.set"
            | "viewer.channels.list_groups"
            | "viewer.channels.set_group"
            | "viewer.native_layers.list"
            | "viewer.native_layers.get"
            | "viewer.native_layers.set_active"
            | "viewer.native_layers.set_visibility"
            | "viewer.native_layers.set_order"
            | "viewer.camera.get"
            | "viewer.camera.set"
            | "viewer.camera.zoom_in"
            | "viewer.camera.zoom_out"
            | "viewer.camera.fit"
            | "viewer.panels.get"
            | "viewer.panels.set"
            | "viewer.rendering.get_smooth_pixels"
            | "viewer.rendering.set_smooth_pixels"
            | "viewer.rendering.get_state"
            | "viewer.objects.get_visibility"
            | "viewer.objects.set_visibility"
            | "viewer.objects.rendering.get_fast"
            | "viewer.objects.rendering.set_fast"
            | "viewer.screenshot.settings.get"
            | "viewer.screenshot.settings.set"
            | "viewer.screenshot.capture"
            | "memory.get"
            | "memory.pin"
            | "memory.unpin"
            | "memory.unpin_all"
    )
}

fn mcp_exposed(name: &str) -> bool {
    matches!(
        name,
        "app.get_state"
            | "app.get_loading_state"
            | "app.navigation.show_project"
            | "project.rois.list"
            | "project.rois.open"
            | "project.open"
            | "project.save"
            | "datasets.open_ome_zarr"
            | "datasets.open_tiff"
            | "datasets.open_mosaic_samplesheet"
            | "viewer.channels.list"
            | "viewer.channels.list_visible"
            | "viewer.channels.get_active"
            | "viewer.channels.set_active"
            | "viewer.channels.set_visible"
            | "viewer.channels.get_contrast"
            | "viewer.channels.set_contrast"
            | "viewer.channels.intensity_stats"
            | "viewer.channels.set_order"
            | "viewer.channels.list_groups"
            | "viewer.channels.set_group"
            | "viewer.panels.get"
            | "viewer.panels.set"
            | "viewer.rendering.get_smooth_pixels"
            | "viewer.rendering.set_smooth_pixels"
            | "viewer.camera.get"
            | "viewer.camera.set"
            | "viewer.camera.fit"
            | "viewer.camera.zoom_in"
            | "viewer.camera.zoom_out"
            | "viewer.objects.get_visibility"
            | "viewer.objects.set_visibility"
            | "viewer.objects.get_selection"
            | "viewer.objects.query_rect"
            | "viewer.objects.query_view"
            | "viewer.objects.select_rect"
            | "viewer.objects.clear_selection"
            | "viewer.objects.get_filter"
            | "viewer.objects.set_filter"
            | "viewer.objects.clear_filter"
            | "viewer.ui.set_left_tab"
            | "viewer.ui.set_right_tab"
            | "mosaic.ui.set_right_tab"
            | "mosaic.layout.configure"
            | "viewer.screenshot.capture"
            | "app.screenshot.capture"
            | "project.screenshot.capture"
    )
}

fn execution_class(name: &str, starts_task: bool) -> ExecutionClass {
    if name.contains("screenshot") || name == "exports.canvas.capture" {
        return ExecutionClass::Presentation;
    }
    if matches!(
        name,
        "viewer.camera.fit"
            | "viewer.viewports.camera.fit"
            | "viewer.objects.query_view"
            | "viewer.objects.query_region"
    ) {
        return ExecutionClass::Geometry;
    }
    if name.starts_with("data.resources.") || name.starts_with("viewer.layers.") {
        return ExecutionClass::ExternalCompute;
    }
    if starts_task
        || name.starts_with("datasets.open_")
        || matches!(name, "project.open" | "project.save" | "project.save_as")
        || name.ends_with(".load")
        || name.ends_with(".reload")
        || name.contains("preload")
        || name.starts_with("exports.")
        || name.starts_with("viewer.measurements.")
        || name.starts_with("viewer.analysis.warmup.")
        || name.starts_with("memory.pin")
    {
        return ExecutionClass::Resource;
    }
    ExecutionClass::Model
}

macro_rules! method {
    (
        $name:literal, $summary:literal, $capability:literal,
        $mutates:expr, $starts_task:expr, $event:expr, $available_in:expr, $shape:ident
    ) => {
        MethodDescriptor {
            name: $name,
            summary: $summary,
            capability: $capability,
            mutates: $mutates,
            starts_task: $starts_task,
            mcp_exposed: mcp_exposed($name),
            stability: Stability::Provisional,
            request_shape: RequestShape::$shape,
            event: $event,
            available_in: $available_in,
            since: "0.2.0",
            execution_class: execution_class($name, $starts_task),
        }
    };
}

const ALL_MODES: &[&str] = &["project", "single", "mosaic", "transition"];
const READY_MODES: &[&str] = &["project", "single", "mosaic"];
const VIEWER_MODES: &[&str] = &["single", "mosaic"];
const SINGLE_MODE: &[&str] = &["single"];
const MOSAIC_MODE: &[&str] = &["mosaic"];

mod actor_methods;
mod catalog;
mod protocol_catalog;
mod schema;

pub use actor_methods::ACTOR_CAPABLE_METHODS;
pub use catalog::METHODS;
pub use protocol_catalog::{METHOD_ALIASES, PROTOCOL_METHODS};
use schema::request_schema_for;

pub fn method(name: &str) -> Option<&'static MethodDescriptor> {
    let canonical = canonical_method(name);
    METHODS
        .iter()
        .find(|descriptor| descriptor.name == canonical)
}

pub fn canonical_method(name: &str) -> &str {
    METHOD_ALIASES
        .iter()
        .find_map(|(alias, canonical)| (*alias == name).then_some(*canonical))
        .unwrap_or(name)
}

pub fn capabilities() -> Vec<String> {
    METHODS
        .iter()
        .map(|descriptor| descriptor.capability)
        .chain(PROTOCOL_METHODS.iter().map(|method| method.2))
        .chain(["system.introspect"])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub fn catalog_json() -> Value {
    Value::Array(
        METHODS
            .iter()
            .map(|descriptor| {
                json!({
                    "name": descriptor.name,
                    "summary": descriptor.summary,
                    "capability": descriptor.capability,
                    "mutates": descriptor.mutates,
                    "starts_task": descriptor.starts_task,
                    "mcp_exposed": descriptor.mcp_exposed,
                    "stability": descriptor.stability,
                    "since": descriptor.since,
                    "event": descriptor.event,
                    "available_in": descriptor.available_in,
                    "execution_class": descriptor.execution_class,
                    "readiness_requirements": descriptor.execution_class.readiness_requirements(),
                    "execution_route": execution_route_json(descriptor),
                    "aliases": aliases_for(descriptor.name),
                    "request_schema": request_schema_for(descriptor),
                    "result_schema": {"type": "object"},
                })
            })
            .chain(PROTOCOL_METHODS.iter().map(
                |(name, summary, capability, mutates, starts_task)| {
                    let execution_class = execution_class(name, *starts_task);
                    json!({
                        "name": name,
                        "summary": summary,
                        "capability": capability,
                        "mutates": mutates,
                        "starts_task": starts_task,
                        "mcp_exposed": false,
                        "stability": Stability::Experimental,
                        "execution_class": execution_class,
                        "readiness_requirements": execution_class.readiness_requirements(),
                        "execution_route": {
                            "summary":ExecutionOwner::ControlService,
                            "by_mode":{
                                "project":{"default_owner":ExecutionOwner::ControlService,"conditional":false},
                                "single":{"default_owner":ExecutionOwner::ControlService,"conditional":false},
                                "mosaic":{"default_owner":ExecutionOwner::ControlService,"conditional":false},
                                "transition":{"default_owner":ExecutionOwner::ControlService,"conditional":false},
                            },
                            "variants":[],
                        },
                        "request_schema": {"type": "object"},
                    })
                },
            ))
            .chain(METHOD_ALIASES.iter().filter_map(|(alias, canonical)| {
                method(canonical).map(|descriptor| {
                    json!({
                        "name": alias,
                        "canonical_name": canonical,
                        "summary": descriptor.summary,
                        "capability": descriptor.capability,
                        "mutates": descriptor.mutates,
                        "starts_task": descriptor.starts_task,
                        "mcp_exposed": false,
                        "stability": Stability::Experimental,
                        "deprecated": true,
                        "since": descriptor.since,
                        "event": descriptor.event,
                        "available_in": descriptor.available_in,
                        "execution_class": descriptor.execution_class,
                        "readiness_requirements": descriptor.execution_class.readiness_requirements(),
                        "execution_route": execution_route_json(descriptor),
                        "request_schema": request_schema_for(descriptor),
                        "result_schema": {"type": "object"},
                    })
                })
            }))
            .collect(),
    )
}

pub fn aliases_for(canonical: &str) -> Vec<&'static str> {
    METHOD_ALIASES
        .iter()
        .filter_map(|(alias, target)| (*target == canonical).then_some(*alias))
        .collect()
}

pub fn availability_catalog(mode: &str, requested: Option<&[String]>) -> Value {
    let requested = requested.map(|methods| {
        methods
            .iter()
            .filter_map(|name| method(name))
            .map(|descriptor| descriptor.name)
            .collect::<BTreeSet<_>>()
    });
    let methods = METHODS
        .iter()
        .filter(|descriptor| {
            requested
                .as_ref()
                .is_none_or(|requested| requested.contains(descriptor.name))
        })
        .map(|descriptor| {
            let available = descriptor.available_in.contains(&mode);
            json!({
                "method": descriptor.name,
                "available": available,
                "mode": mode,
                "available_in": descriptor.available_in,
                "capability": descriptor.capability,
                "reason": if available {
                    Value::Null
                } else if mode == "transition" {
                    json!("not_ready")
                } else {
                    json!("wrong_mode")
                },
            })
        })
        .collect::<Vec<_>>();
    json!({"mode": mode, "methods": methods})
}

#[cfg(test)]
#[path = "registry/tests.rs"]
mod tests;
