use std::collections::BTreeSet;
use std::sync::LazyLock;

use serde::Serialize;
use serde_json::{Value, json};

mod actor_methods;

pub use actor_methods::ACTOR_CAPABLE_METHODS;

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
    params: &Value,
    project_view_requires_resource_load: bool,
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
    if is_parameter_routed_primary_object_method(descriptor.name)
        && (matches!(
            params.get("target").and_then(Value::as_str),
            Some("active" | "spatial_shape")
        ) || params.get("screen_rect").is_some())
    {
        return ExecutionOwner::LegacyUi;
    }
    if descriptor.name == "project.views.apply" && project_view_requires_resource_load {
        return ExecutionOwner::LegacyUi;
    }
    ExecutionOwner::Actor
}

pub fn execution_route_summary(descriptor: &MethodDescriptor) -> &'static str {
    let actor_capable = ACTOR_CAPABLE_METHODS.contains(&descriptor.name);
    if !actor_capable {
        return "legacy_ui";
    }
    if is_parameter_routed_primary_object_method(descriptor.name)
        || descriptor.name == "project.views.apply"
        || ((descriptor.name.starts_with("viewer.") || descriptor.name.starts_with("memory."))
            && descriptor.available_in.contains(&"mosaic")
            && !is_actor_owned_mosaic_shared_method(descriptor.name))
    {
        "hybrid"
    } else {
        "actor"
    }
}

pub fn execution_route_json(descriptor: &MethodDescriptor) -> Value {
    let actor_capable = ACTOR_CAPABLE_METHODS.contains(&descriptor.name);
    let variants = if is_parameter_routed_primary_object_method(descriptor.name) {
        json!([
            {
                "when":{"target":["active","spatial_shape"]},
                "owner":"legacy_ui",
                "reason":"renderer-owned object target"
            },
            {
                "when":{"screen_rect":"present"},
                "owner":"legacy_ui",
                "reason":"screen-space query requires renderer state"
            }
        ])
    } else if descriptor.name == "project.views.apply" {
        json!([{
            "when":{"required_project_resources":"not_loaded"},
            "owner":"legacy_ui",
            "reason":"saved-view resource loading has not yet migrated"
        }])
    } else {
        json!([])
    };
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
                    "conditional": owner == ExecutionOwner::Actor && variants.as_array().is_some_and(|items| !items.is_empty()),
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

pub fn is_parameter_routed_primary_object_method(method: &str) -> bool {
    matches!(
        method,
        "viewer.objects.get_selection"
            | "viewer.objects.query_rect"
            | "viewer.objects.query_view"
            | "viewer.objects.query_lasso"
            | "viewer.objects.select_rect"
            | "viewer.objects.select_lasso"
            | "viewer.objects.clear_selection"
            | "viewer.objects.selection.select_ids"
            | "viewer.objects.selection.select_filtered"
            | "viewer.objects.focus.set"
            | "viewer.objects.focus.clear"
            | "viewer.objects.style.get"
            | "viewer.objects.style.set"
            | "viewer.objects.legend.set"
            | "viewer.objects.rendering.get_fast"
            | "viewer.objects.rendering.set_fast"
            | "viewer.objects.get_filter"
            | "viewer.objects.set_filter"
            | "viewer.objects.clear_filter"
            | "viewer.objects.filters.set_model"
            | "viewer.objects.filters.get_revision"
            | "viewer.analysis.get"
            | "viewer.analysis.set"
            | "viewer.analysis.histogram"
            | "viewer.analysis.suggest_thresholds"
            | "viewer.analysis.warmup.get"
            | "viewer.analysis.warmup.start"
            | "viewer.analysis.presets.import"
            | "viewer.analysis.presets.export"
            | "viewer.measurements.get"
            | "viewer.measurements.configure"
            | "viewer.measurements.start"
            | "viewer.measurements.cancel"
            | "viewer.measurements.properties.list"
            | "exports.objects.columns"
            | "exports.objects.get_state"
            | "exports.objects.start"
            | "exports.objects.export_csv"
            | "exports.objects.export_geoparquet"
    )
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

pub static METHODS: LazyLock<Vec<MethodDescriptor>> = LazyLock::new(|| {
    vec![
        method!(
            "app.get_state",
            "Get current application and viewer state.",
            "viewer.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.settings.get",
            "Inspect persistent application preferences.",
            "application.settings.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.settings.set",
            "Validate, persist, and apply application preferences.",
            "application.settings.write",
            true,
            false,
            Some("application.settings.changed"),
            ALL_MODES,
            AppSettings
        ),
        method!(
            "app.recent_projects.list",
            "List recently opened project files.",
            "application.settings.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.recent_projects.forget",
            "Forget one recently opened project path.",
            "application.settings.write",
            true,
            false,
            Some("application.recent_projects.changed"),
            ALL_MODES,
            Path
        ),
        method!(
            "app.recent_projects.clear",
            "Clear the recent-project list.",
            "application.settings.write",
            true,
            false,
            Some("application.recent_projects.changed"),
            ALL_MODES,
            Empty
        ),
        method!(
            "app.lifecycle.get",
            "Inspect dirty state and safe close options.",
            "application.lifecycle.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.lifecycle.request_close",
            "Request that the Odon window close with an explicit save decision.",
            "application.close",
            true,
            false,
            Some("application.close.requested"),
            READY_MODES,
            LifecycleRequest
        ),
        method!(
            "app.lifecycle.request_quit",
            "Request that Odon quit with an explicit save decision.",
            "application.quit",
            true,
            false,
            Some("application.quit.requested"),
            READY_MODES,
            LifecycleRequest
        ),
        MethodDescriptor {
            name: "app.get_method_availability",
            summary: "Describe whether control methods are available in the current mode.",
            capability: "system.introspect",
            mutates: false,
            starts_task: false,
            mcp_exposed: false,
            stability: Stability::Provisional,
            request_shape: RequestShape::MethodAvailability,
            event: None,
            available_in: ALL_MODES,
            since: "0.2.0",
            execution_class: execution_class("app.get_method_availability", false),
        },
        method!(
            "project.rois.list",
            "List project ROIs.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.get",
            "Get project metadata and lifecycle state.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.create",
            "Create a new empty project workspace.",
            "project.write",
            true,
            false,
            Some("application.mode.changed"),
            READY_MODES,
            ProjectCreate
        ),
        method!(
            "project.save_as",
            "Save the active project to an explicit path.",
            "project.write",
            true,
            false,
            Some("project.saved"),
            READY_MODES,
            Path
        ),
        method!(
            "project.update_metadata",
            "Update supported project metadata and search roots.",
            "project.write",
            true,
            false,
            Some("project.changed"),
            READY_MODES,
            ProjectMetadata
        ),
        method!(
            "project.samplesheets.inspect",
            "Parse and validate a samplesheet without changing the active project.",
            "project.read",
            false,
            false,
            None,
            ALL_MODES,
            SamplesheetInspect
        ),
        method!(
            "project.samplesheets.validate",
            "Validate samplesheet identity, paths, and metadata without changing the project.",
            "project.read",
            false,
            false,
            None,
            ALL_MODES,
            SamplesheetInspect
        ),
        method!(
            "project.samplesheets.import",
            "Replace project ROIs from a validated samplesheet.",
            "project.write",
            true,
            true,
            Some("project.rois.changed"),
            READY_MODES,
            Path
        ),
        method!(
            "project.samplesheets.export",
            "Export local project ROIs and metadata to a samplesheet.",
            "project.export",
            true,
            false,
            Some("project.samplesheet.exported"),
            READY_MODES,
            SamplesheetExport
        ),
        method!(
            "project.discovery.add_root",
            "Discover OME-Zarr datasets recursively and add them as project ROIs.",
            "project.write",
            true,
            true,
            Some("project.rois.changed"),
            READY_MODES,
            Path
        ),
        method!(
            "project.objects.preload.get",
            "Inspect available and cached project object segmentations.",
            "project.objects.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.objects.preload.list_sources",
            "List preload-eligible project segmentation sources.",
            "project.objects.read",
            false,
            false,
            None,
            READY_MODES,
            MosaicItems
        ),
        method!(
            "project.objects.preload.start",
            "Preload project object geometry or centroids and wait for completion.",
            "project.objects.write",
            true,
            true,
            Some("project.objects.preload.changed"),
            READY_MODES,
            ObjectPreloadStart
        ),
        method!(
            "project.objects.preload.clear",
            "Clear preloaded project objects from memory.",
            "project.objects.write",
            true,
            false,
            Some("project.objects.preload.changed"),
            READY_MODES,
            Empty
        ),
        method!(
            "project.rois.get",
            "Get one project ROI by stable ID.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            ProjectRoiId
        ),
        method!(
            "project.rois.add",
            "Add a project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiAdd
        ),
        method!(
            "project.rois.update",
            "Update a project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiUpdate
        ),
        method!(
            "project.rois.remove",
            "Remove a project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiId
        ),
        method!(
            "project.rois.reorder",
            "Set the exact project ROI order.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiOrder
        ),
        method!(
            "project.rois.get_selection",
            "Get focused and selected project ROIs.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.rois.select",
            "Select project ROIs by stable ID.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            ProjectRoiSelect
        ),
        method!(
            "project.rois.focus",
            "Focus a project ROI by stable ID.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            ProjectRoiId
        ),
        method!(
            "project.rois.next",
            "Focus the next project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            StepPlane
        ),
        method!(
            "project.rois.previous",
            "Focus the previous project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            StepPlane
        ),
        method!(
            "project.rois.open_selected_mosaic",
            "Open selected project ROIs as a mosaic.",
            "project.write",
            true,
            true,
            Some("application.mode.changed"),
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.channels.list",
            "List channels.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.channels.list_visible",
            "List visible channels.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.planes.get",
            "Get the active view plane, slice, extent, and supported orientations.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.planes.set",
            "Set the active view orientation and/or slice.",
            "viewer.write",
            true,
            false,
            Some("viewer.planes.changed"),
            SINGLE_MODE,
            SetPlane
        ),
        method!(
            "viewer.planes.next",
            "Move forward through slices in the active view orientation.",
            "viewer.write",
            true,
            false,
            Some("viewer.planes.changed"),
            SINGLE_MODE,
            StepPlane
        ),
        method!(
            "viewer.planes.previous",
            "Move backward through slices in the active view orientation.",
            "viewer.write",
            true,
            false,
            Some("viewer.planes.changed"),
            SINGLE_MODE,
            StepPlane
        ),
        method!(
            "viewer.planes.operation_availability",
            "Describe XY-only operation safeguards for the active multidimensional view plane.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.panels.get",
            "Get side-panel visibility.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.panels.set",
            "Set side-panel visibility.",
            "viewer.write",
            true,
            false,
            Some("viewer.panels.changed"),
            VIEWER_MODES,
            SetSidePanels
        ),
        method!(
            "viewer.rendering.get_smooth_pixels",
            "Get image interpolation state.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.rendering.set_smooth_pixels",
            "Set image interpolation state.",
            "viewer.write",
            true,
            false,
            Some("viewer.rendering.changed"),
            VIEWER_MODES,
            SetSmoothPixels
        ),
        method!(
            "viewer.rendering.get_state",
            "Inspect renderer, additive compositing, interpolation, and deterministic-capture readiness.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "app.get_loading_state",
            "Get loading diagnostics.",
            "viewer.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "viewer.channels.get_active",
            "Get the active channel.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Object
        ),
        method!(
            "viewer.channels.set_active",
            "Set the active channel.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.channels.set_visible",
            "Set channel visibility.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            SetVisibleChannels
        ),
        method!(
            "viewer.channels.set_color",
            "Set a channel's additive-compositing colour.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            SetChannelColor
        ),
        method!(
            "viewer.channels.set_note",
            "Set a channel note.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            SetChannelNote
        ),
        method!(
            "viewer.channels.get_transform",
            "Get a channel's translation, scale, and rotation.",
            "viewer.channels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.channels.set_transform",
            "Set a channel's translation, scale, and rotation.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            SINGLE_MODE,
            SetChannelTransform
        ),
        method!(
            "viewer.channels.reset_transform",
            "Reset a channel transform to identity.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "project.open",
            "Open a project.",
            "project.write",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "datasets.open_ome_zarr",
            "Open an OME-Zarr dataset.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "datasets.inspect",
            "Inspect a local dataset source and discover supported elements without opening it.",
            "datasets.read",
            false,
            false,
            None,
            ALL_MODES,
            Path
        ),
        method!(
            "datasets.open_spatialdata",
            "Open a selected SpatialData image with typed image, label, shape, and point elements.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            SpatialDataOpen
        ),
        method!(
            "datasets.open_xenium",
            "Open a Xenium experiment with explicit imagery and overlay choices.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            XeniumOpen
        ),
        method!(
            "datasets.open_http",
            "Open a remote HTTP(S) OME-Zarr source.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            HttpOpen
        ),
        method!(
            "datasets.s3.get_session",
            "Inspect redacted session-only S3 connection metadata.",
            "datasets.credentials.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "datasets.s3.configure_session",
            "Configure session-only S3 credentials without persisting or returning secrets.",
            "datasets.credentials.write",
            true,
            false,
            Some("datasets.credentials.changed"),
            ALL_MODES,
            S3Session
        ),
        method!(
            "datasets.s3.clear_session",
            "Remove session-only S3 credentials from Odon memory.",
            "datasets.credentials.write",
            true,
            false,
            Some("datasets.credentials.changed"),
            ALL_MODES,
            Empty
        ),
        method!(
            "datasets.s3.list",
            "List one S3 prefix using the configured session credentials.",
            "datasets.remote.read",
            false,
            true,
            None,
            ALL_MODES,
            S3Prefix
        ),
        method!(
            "datasets.open_s3",
            "Open an S3 OME-Zarr prefix using session credentials.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            S3Prefix
        ),
        method!(
            "deep_links.parse",
            "Parse and validate an Odon deep link into its structured public model.",
            "deep_links.read",
            false,
            false,
            None,
            ALL_MODES,
            DeepLinkUri
        ),
        method!(
            "deep_links.resolve",
            "Resolve a deep link against its project and return an unambiguous ROI without changing application state.",
            "deep_links.read",
            false,
            false,
            None,
            READY_MODES,
            DeepLinkApply
        ),
        method!(
            "deep_links.filters.get",
            "Extract the typed object-filter state carried by a deep link without applying it.",
            "deep_links.read",
            false,
            false,
            None,
            ALL_MODES,
            DeepLinkApply
        ),
        method!(
            "deep_links.generate",
            "Generate a canonical Odon deep link from structured or current viewer state.",
            "deep_links.read",
            false,
            false,
            None,
            READY_MODES,
            DeepLinkGenerate
        ),
        method!(
            "deep_links.apply",
            "Apply a validated deep link as an atomic actor transaction and settle after its model and resources are ready.",
            "application.write",
            true,
            true,
            Some("application.state.changed"),
            ALL_MODES,
            DeepLinkApply
        ),
        method!(
            "datasets.open_tiff",
            "Open a TIFF dataset.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            TiffOpen
        ),
        method!(
            "datasets.open_mosaic_samplesheet",
            "Open a mosaic samplesheet.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "project.rois.open",
            "Open a project ROI.",
            "project.write",
            true,
            true,
            Some("project.active_roi.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "project.save",
            "Save the active project.",
            "project.write",
            true,
            false,
            Some("project.saved"),
            READY_MODES,
            Empty
        ),
        method!(
            "project.views.list",
            "List saved project view presets.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.views.get",
            "Get a saved project view preset.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            ProjectViewSelector
        ),
        method!(
            "project.views.create",
            "Create or replace a saved project view preset from a specification.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            READY_MODES,
            ProjectViewCreate
        ),
        method!(
            "project.views.capture",
            "Capture the current single-image viewer as a saved project view preset.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            SINGLE_MODE,
            ProjectViewCapture
        ),
        method!(
            "project.views.rename",
            "Rename a saved project view preset.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            READY_MODES,
            ProjectViewRename
        ),
        method!(
            "project.views.delete",
            "Delete a saved project view preset.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            READY_MODES,
            ProjectViewSelector
        ),
        method!(
            "project.views.apply",
            "Apply a saved project view preset to the current single-image viewer.",
            "project.write",
            true,
            false,
            Some("project.views.applied"),
            SINGLE_MODE,
            ProjectViewSelector
        ),
        method!(
            "viewer.channels.get_contrast",
            "Get channel contrast.",
            "viewer.channels.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.channels.set_contrast",
            "Set channel contrast.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.get_visibility",
            "Get object overlay visibility.",
            "viewer.layers.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.native_layers.list",
            "List Odon-native layers in their channel and overlay stacks.",
            "viewer.layers.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.native_layers.get",
            "Get one Odon-native layer.",
            "viewer.layers.read",
            false,
            false,
            None,
            VIEWER_MODES,
            NativeLayerSelector
        ),
        method!(
            "viewer.native_layers.set_active",
            "Set the active Odon-native layer.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            NativeLayerSelector
        ),
        method!(
            "viewer.native_layers.set_visibility",
            "Set an Odon-native layer's visibility.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            NativeLayerVisibility
        ),
        method!(
            "viewer.native_layers.set_order",
            "Set the exact order of the native channel or overlay stack.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            NativeLayerOrder
        ),
        method!(
            "viewer.native_layers.set_offset",
            "Set an Odon-native layer's world translation.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            NativeLayerOffset
        ),
        method!(
            "viewer.native_layers.reset_offset",
            "Reset an Odon-native layer's world translation to its loaded baseline.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            NativeLayerSelector
        ),
        method!(
            "viewer.objects.set_visibility",
            "Set object overlay visibility.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.get_state",
            "Get bounded object source, loading, rendering, styling, filter, and selection state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.source.load",
            "Load a CSV, GeoJSON, Parquet, or GeoParquet object source and settle when parsing finishes.",
            "viewer.objects.write",
            true,
            true,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.source.reload",
            "Reload the current object source and settle when parsing finishes.",
            "viewer.objects.write",
            true,
            true,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.objects.source.clear",
            "Clear the current object source and all derived object state.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.objects.source.cancel_load",
            "Cooperatively cancel the current object-source load.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.objects.style.get",
            "Get complete object appearance, color-property, and bounded legend state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.style.set",
            "Set object visibility, stroke, fill, selection overlay, and color-property appearance.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.style.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.legend.set",
            "Set visibility and color overrides for object color-property legend values.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.style.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.rendering.get_fast",
            "Get fast object-rendering mode.",
            "viewer.objects.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.rendering.set_fast",
            "Set fast object-rendering mode.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.rendering.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.properties.list",
            "List the object property schema with bounded pagination and lazy-load state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.properties.load",
            "Load one lazy object property column and settle when its values are available.",
            "viewer.objects.write",
            true,
            true,
            Some("viewer.objects.properties.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.properties.values",
            "Read a bounded page of typed values for one loaded object property.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.get_selection",
            "Get selected objects.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.query_rect",
            "Query objects in a rectangle.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.query_view",
            "Query objects in the viewport.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.query_lasso",
            "Query objects intersecting a world-coordinate lasso with bounded results.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.select_rect",
            "Select objects in a rectangle.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.select_lasso",
            "Select objects intersecting a world-coordinate lasso with explicit set semantics.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.clear_selection",
            "Clear object selection.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.selection.select_ids",
            "Select objects by stable IDs with replace, add, remove, or toggle semantics.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.selection.select_filtered",
            "Apply an explicitly sourced viewport filter or standalone query to selection.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.focus.set",
            "Focus an object by stable ID or index and optionally fit it in the viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.focus.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.focus.clear",
            "Clear primary object focus without clearing the selection set.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.focus.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.selection.state.replace",
            "Atomically replace committed primary object selection with generation checking.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.get_filter",
            "Get object filter state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.set_filter",
            "Set an object filter query.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.clear_filter",
            "Clear object filtering.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.filters.set_model",
            "Set the complete typed simple-clause or boolean-query object filter model.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.filter.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.filters.get_revision",
            "Get the monotonic object-filter revision and bounded visible/hidden counts shared by downstream consumers.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.list",
            "List editable and read-only mask layers with complete presentation state.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.layers.get",
            "Get one mask layer by stable ID.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.create",
            "Create an editable mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.update",
            "Update mask layer name, presentation, editability, or offset.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.delete",
            "Delete a mask layer and its polygons.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.list",
            "List a bounded page of mask polygons in local and world coordinates.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.add",
            "Add a closed polygon to an editable mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.update",
            "Replace the vertices of one editable mask polygon.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.remove",
            "Remove one polygon from an editable mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.selection.get",
            "Get the selected mask polygon and optional selected vertex.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.selection.set",
            "Select one mask polygon and optional vertex by layer ID and index.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.selection.clear",
            "Clear the selected mask polygon and vertex.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.selection.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.undo",
            "Undo the most recent mask or mask-offset edit.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.state.replace",
            "Atomically replace committed mask state with optional generation conflict checking.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.import_geojson",
            "Import GeoJSON polygon or line geometry as a mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.export_geojson",
            "Export one mask layer or all mask layers as GeoJSON.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.persistence.get",
            "Inspect mask persistence state for the current dataset and project.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.persistence.sync",
            "Synchronize live mask layers into the current project in memory.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.labels.list",
            "List discovered NGFF label groups and current render state.",
            "viewer.labels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.labels.get",
            "Inspect current NGFF label selection, loading, visibility, and alignment state.",
            "viewer.labels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.labels.load",
            "Load one discovered NGFF label group into the shared label renderer.",
            "viewer.labels.write",
            true,
            false,
            Some("viewer.labels.changed"),
            SINGLE_MODE,
            LabelLoad
        ),
        method!(
            "viewer.labels.unload",
            "Unload the active NGFF label group and release its loader state.",
            "viewer.labels.write",
            true,
            false,
            Some("viewer.labels.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.labels.set_visibility",
            "Set NGFF label visibility, loading the selected group when necessary.",
            "viewer.labels.write",
            true,
            false,
            Some("viewer.labels.changed"),
            SINGLE_MODE,
            LabelVisibility
        ),
        method!(
            "viewer.thresholds.levels.list",
            "List image levels and whole-image threshold safety limits.",
            "viewer.thresholds.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.screenshot.settings.get",
            "Inspect canvas screenshot overlay, scaling, quick-save, and readiness settings.",
            "viewer.screenshot",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.scale_bar.get",
            "Inspect canvas scale-bar visibility and availability.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.scale_bar.set",
            "Set canvas scale-bar visibility.",
            "viewer.write",
            true,
            false,
            Some("viewer.scale_bar.changed"),
            SINGLE_MODE,
            SetScaleBar
        ),
        method!(
            "viewer.screenshot.settings.set",
            "Set canvas screenshot overlay, scaling, and quick-save folder options.",
            "viewer.screenshot",
            true,
            false,
            Some("viewer.screenshot.settings.changed"),
            VIEWER_MODES,
            ScreenshotSettings
        ),
        method!(
            "memory.tiles.get",
            "Inspect tile workers, cache occupancy, target level, and prefetch policy.",
            "memory.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "memory.tiles.set",
            "Set tile worker count, prefetch policy, and pinned-level fallback.",
            "memory.write",
            true,
            false,
            Some("memory.tiles.changed"),
            SINGLE_MODE,
            TileLoading
        ),
        method!(
            "memory.get",
            "Inspect system RAM, selected channel estimates, and pinned-level lifecycle.",
            "memory.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "memory.pin",
            "Load selected channels from one pyramid level into CPU RAM.",
            "memory.write",
            true,
            true,
            Some("memory.changed"),
            VIEWER_MODES,
            MemoryPin
        ),
        method!(
            "memory.unpin",
            "Unload one pinned pyramid level from CPU RAM.",
            "memory.write",
            true,
            false,
            Some("memory.changed"),
            VIEWER_MODES,
            MemoryUnpin
        ),
        method!(
            "memory.unpin_all",
            "Unload all pinned pyramid levels from CPU RAM.",
            "memory.write",
            true,
            false,
            Some("memory.changed"),
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.get",
            "Get threshold-preview configuration, source extent, and bounded summary statistics.",
            "viewer.thresholds.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.configure",
            "Configure threshold scope, level, channel, value, and component filtering.",
            "viewer.thresholds.write",
            true,
            false,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.thresholds.preview.start",
            "Read the selected channel region and start an interactive threshold preview.",
            "viewer.thresholds.write",
            true,
            true,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.thresholds.preview.refresh",
            "Reload source pixels for the active threshold preview.",
            "viewer.thresholds.write",
            true,
            true,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.apply",
            "Filter components, polygonize the preview, and create an editable mask layer.",
            "viewer.thresholds.write",
            true,
            true,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.cancel",
            "Cancel and clear the active threshold preview.",
            "viewer.thresholds.write",
            true,
            false,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.analysis.get",
            "Get persisted calls, named selections, channel mappings, and analysis readiness.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.set",
            "Atomically replace calls, named selections, mappings, and live-analysis options.",
            "viewer.analysis.write",
            true,
            false,
            Some("viewer.analysis.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.histogram",
            "Compute a bounded histogram for a numeric property over the active filtered set.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.suggest_thresholds",
            "Suggest quantile or one-dimensional K-means thresholds for a numeric property.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.warmup.get",
            "Inspect project-linked property-analysis cache warmup progress.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.warmup.start",
            "Start project-linked property-analysis cache warmup.",
            "viewer.analysis.write",
            true,
            true,
            Some("viewer.analysis.warmup.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.presets.import",
            "Import a call preset JSON file.",
            "viewer.analysis.write",
            true,
            false,
            Some("viewer.analysis.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.presets.export",
            "Export calls as a reusable preset JSON file.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.get",
            "Inspect polygon intensity measurement configuration and progress.",
            "viewer.measurements.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.configure",
            "Configure metric, image level, filtered scope, concurrency, and output prefix.",
            "viewer.measurements.write",
            true,
            false,
            Some("viewer.measurements.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.start",
            "Start background mean or exact-median polygon intensity measurement.",
            "viewer.measurements.write",
            true,
            true,
            Some("viewer.measurements.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.cancel",
            "Cooperatively cancel the active polygon intensity measurement.",
            "viewer.measurements.write",
            true,
            false,
            Some("viewer.measurements.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.properties.list",
            "List numeric properties generated by the configured measurement prefix.",
            "viewer.measurements.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.columns",
            "List source, geometry, measurement, call, and named-selection export columns.",
            "exports.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.get_state",
            "Inspect enriched object export progress and status.",
            "exports.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.start",
            "Export all, filtered, or selected objects to enriched CSV or GeoParquet.",
            "exports.objects.write",
            true,
            true,
            Some("exports.objects.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.export_csv",
            "Export all, filtered, or selected objects and derived columns to CSV.",
            "exports.objects.write",
            true,
            true,
            Some("exports.objects.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.export_geoparquet",
            "Export all, filtered, or selected objects with WKB geometry and GeoParquet metadata.",
            "exports.objects.write",
            true,
            true,
            Some("exports.objects.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.channels.intensity_stats",
            "Get channel intensity statistics.",
            "viewer.analysis.read",
            false,
            true,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.channels.set_order",
            "Set channel ordering.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.channels.presentation.get",
            "Inspect channel-list search, sort, and effective ordering.",
            "viewer.channels.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.channels.presentation.set",
            "Set channel-list search and sort presentation state.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            ChannelPresentation
        ),
        method!(
            "viewer.channels.list_groups",
            "List channel groups.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.channels.set_group",
            "Set channel grouping.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.camera.get",
            "Get camera state.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.camera.set",
            "Set camera state.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            SetCamera
        ),
        method!(
            "viewer.camera.zoom_in",
            "Zoom in.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.camera.zoom_out",
            "Zoom out.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.camera.fit",
            "Fit content to the viewport.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.workspace.get",
            "Get the current viewer workspace, layout, links, and viewport snapshots.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.workspace.layout.set",
            "Set the current single or two-viewport layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.workspace.layout.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.workspace.layout.get",
            "Get the current viewport workspace layout and ordered viewport IDs.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.workspace.swap",
            "Swap the two viewport positions in the current layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.workspace.layout.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewports.list",
            "List native viewports and their navigation and presentation snapshots.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewports.get",
            "Get one viewport by stable ID.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.create",
            "Clone a viewport into a horizontal or vertical comparison layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.created"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.clone",
            "Clone an explicit viewport into a horizontal or vertical comparison layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.created"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.rename",
            "Rename a viewport.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.remove",
            "Remove a viewport while preserving the final remaining view.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.removed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.set_active",
            "Set the active viewport used by native panels and legacy viewer methods.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.active_changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.set",
            "Configure camera, plane, and shared-selection links between viewports.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.get",
            "Get camera, plane, and shared-selection links for the workspace.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewport_links.list",
            "List the workspace's fixed comparison link group.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewport_links.create",
            "Configure the fixed comparison link group for the two workspace viewports.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.update",
            "Update fields in the fixed comparison link group.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.remove",
            "Disable optional navigation links while retaining document-shared selection.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.camera.get",
            "Get camera state for an explicit viewport.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.camera.set",
            "Set camera state for an explicit viewport and propagate configured links.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.navigation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.camera.fit",
            "Fit content in an explicit viewport and propagate configured camera links.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.navigation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.planes.get",
            "Get plane state for an explicit viewport.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.planes.set",
            "Set plane state for an explicit viewport and propagate configured links.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.navigation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.get",
            "Get channel presentation for an explicit viewport.",
            "viewer.channels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_visible",
            "Set visible channels for an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set",
            "Set the visible channel collection for an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_active",
            "Set the active channel in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_color",
            "Set channel color in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_contrast",
            "Set channel contrast in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_order",
            "Set channel order in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.list_groups",
            "List channel-group presentation for an explicit viewport.",
            "viewer.channels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_group",
            "Set channel-group membership and color presentation in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.style.get",
            "Get object presentation for an explicit viewport.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.rendering.get",
            "Get sampling, scale-bar, HUD, and tile-debug preferences for an explicit viewport.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.rendering.set",
            "Set sampling, scale-bar, HUD, and tile-debug preferences for an explicit viewport.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.style.set",
            "Set independent object presentation for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.legend.set",
            "Set independent object-property palette entries for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.filter.get",
            "Get the independent segmentation-object filter for an explicit viewport.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.filter.set",
            "Set an independent segmentation-object filter for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.filter.clear",
            "Clear the segmentation-object filter for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.list",
            "List channels and overlays with presentation state for an explicit viewport.",
            "viewer.layers.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.get",
            "Get one native layer and its complete presentation for an explicit viewport.",
            "viewer.layers.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set",
            "Set one native layer's independent presentation in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set_visibility",
            "Set native-layer visibility in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set_order",
            "Set native-layer order in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set_active",
            "Set the active native layer in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.state.replace",
            "Atomically replace actor-owned native-layer presentation for one viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.ui.set_right_tab",
            "Set the single-view right tab.",
            "viewer.write",
            true,
            false,
            Some("viewer.ui.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "mosaic.ui.set_right_tab",
            "Set the mosaic right tab.",
            "viewer.write",
            true,
            false,
            Some("mosaic.ui.changed"),
            MOSAIC_MODE,
            Object
        ),
        method!(
            "mosaic.layout.configure",
            "Configure mosaic layout.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.layout.changed"),
            MOSAIC_MODE,
            MosaicLayout
        ),
        method!(
            "mosaic.get_state",
            "Get complete mosaic layout, ROI, and focus state.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.items.list",
            "List positioned mosaic items with stable ordering and pagination.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            MosaicItems
        ),
        method!(
            "mosaic.selection.get",
            "Get selected mosaic ROIs.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.selection.set",
            "Select mosaic ROIs using stable IDs and replace, add, remove, toggle, all, or range semantics.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.selection.changed"),
            MOSAIC_MODE,
            MosaicSelect
        ),
        method!(
            "mosaic.selection.clear",
            "Clear the mosaic ROI selection.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.selection.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.focus.get",
            "Get the focused mosaic ROI.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.focus.set",
            "Focus a mosaic ROI by stable ROI ID or index.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            MosaicFocus
        ),
        method!(
            "mosaic.focus.next",
            "Focus the next mosaic ROI.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            StepPlane
        ),
        method!(
            "mosaic.focus.previous",
            "Focus the previous mosaic ROI.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            StepPlane
        ),
        method!(
            "mosaic.focus.fit",
            "Fit the focused mosaic ROI to the viewport.",
            "mosaic.write",
            true,
            false,
            Some("viewer.camera.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.focus.clear",
            "Clear focused mosaic ROI without changing selection.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.fit_all",
            "Fit all mosaic items to the viewport.",
            "mosaic.write",
            true,
            false,
            Some("viewer.camera.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.objects.get_state",
            "Get per-ROI mosaic object-source, loading, and allocation state.",
            "viewer.objects.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.objects.load_selected",
            "Load object segmentations for the selected mosaic ROIs and settle when all requested reads finish.",
            "viewer.objects.write",
            true,
            true,
            Some("mosaic.objects.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.objects.cancel_load",
            "Cancel remaining scheduled object loads while allowing an in-flight disk read to finish.",
            "viewer.objects.write",
            true,
            false,
            Some("mosaic.objects.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "app.navigation.show_project",
            "Show the project page.",
            "application.write",
            true,
            false,
            Some("application.mode.changed"),
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.screenshot.capture",
            "Capture the viewer canvas.",
            "viewer.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            VIEWER_MODES,
            CaptureScreenshot
        ),
        method!(
            "viewer.workspace.screenshot.capture",
            "Capture the composed multi-viewport canvas workspace.",
            "viewer.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            SINGLE_MODE,
            CaptureScreenshot
        ),
        method!(
            "app.screenshot.capture",
            "Capture the Odon window.",
            "application.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            READY_MODES,
            Object
        ),
        method!(
            "project.screenshot.capture",
            "Capture the project page.",
            "application.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            READY_MODES,
            Object
        ),
    ]
});

pub static PROTOCOL_METHODS: &[(&str, &str, &str, bool, bool)] = &[
    (
        "system.hello",
        "Authenticate and negotiate a control protocol version.",
        "system.connect",
        false,
        false,
    ),
    (
        "system.get_capabilities",
        "List capabilities granted by this Odon build.",
        "system.introspect",
        false,
        false,
    ),
    (
        "system.list_methods",
        "List control methods and request schemas.",
        "system.introspect",
        false,
        false,
    ),
    (
        "system.describe_methods",
        "List control methods and request schemas.",
        "system.introspect",
        false,
        false,
    ),
    (
        "system.describe_events",
        "Describe initial event families and envelope fields.",
        "system.introspect",
        false,
        false,
    ),
    (
        "system.get_application_surface",
        "Get the feature-to-control parity manifest.",
        "system.introspect",
        false,
        false,
    ),
    (
        "ui.describe_schema",
        "Describe declarative UI schema v1 limits and vocabulary.",
        "ui.read",
        false,
        false,
    ),
    (
        "system.batch",
        "Execute application commands in order.",
        "system.batch",
        true,
        false,
    ),
    (
        "system.get_diagnostics",
        "Inspect bounded control-server state.",
        "system.introspect",
        false,
        false,
    ),
    (
        "events.subscribe",
        "Subscribe to server-pushed event patterns.",
        "events.read",
        false,
        false,
    ),
    (
        "events.unsubscribe",
        "Remove event subscriptions.",
        "events.read",
        false,
        false,
    ),
    (
        "events.get_status",
        "Inspect event queue diagnostics.",
        "events.read",
        false,
        false,
    ),
    (
        "tasks.start",
        "Run an application command as a retained task.",
        "tasks.write",
        true,
        true,
    ),
    (
        "tasks.get",
        "Get a task snapshot.",
        "tasks.read",
        false,
        false,
    ),
    (
        "tasks.list",
        "List retained tasks.",
        "tasks.read",
        false,
        false,
    ),
    (
        "tasks.cancel",
        "Cancel a queued task.",
        "tasks.write",
        true,
        false,
    ),
    (
        "tasks.forget",
        "Forget a terminal task.",
        "tasks.write",
        true,
        false,
    ),
    (
        "data.resources.register",
        "Register referenced external data.",
        "data.write",
        true,
        false,
    ),
    (
        "data.resources.list",
        "List registered external data.",
        "data.read",
        false,
        false,
    ),
    (
        "data.resources.get",
        "Get an external data descriptor.",
        "data.read",
        false,
        false,
    ),
    (
        "data.resources.remove",
        "Remove an unreferenced data descriptor.",
        "data.write",
        true,
        false,
    ),
    (
        "viewer.layers.add",
        "Add a referenced viewer layer.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "viewer.layers.list",
        "List referenced viewer layers.",
        "viewer.layers.read",
        false,
        false,
    ),
    (
        "viewer.layers.get",
        "Get a referenced viewer layer.",
        "viewer.layers.read",
        false,
        false,
    ),
    (
        "viewer.layers.update",
        "Update layer state or style.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "viewer.layers.remove",
        "Remove a referenced layer.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "viewer.layers.reorder",
        "Set referenced layer order.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "ui.extensions.register",
        "Register a declarative UI extension.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.extensions.list",
        "List UI extensions.",
        "ui.read",
        false,
        false,
    ),
    (
        "ui.extensions.remove",
        "Remove an owned UI extension.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.contributions.register",
        "Register a validated component tree.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.contributions.list",
        "List component trees.",
        "ui.read",
        false,
        false,
    ),
    (
        "ui.contributions.patch_values",
        "Atomically patch retained UI values.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.contributions.remove",
        "Remove an owned component tree.",
        "ui.panels",
        true,
        false,
    ),
];

pub static METHOD_ALIASES: &[(&str, &str)] = &[
    ("get_current_view", "app.get_state"),
    ("viewer.get_state", "app.get_state"),
    ("get_loading_state", "app.get_loading_state"),
    ("show_project_page", "app.navigation.show_project"),
    ("app.show_project_page", "app.navigation.show_project"),
    ("list_project_rois", "project.rois.list"),
    ("project.get_state", "project.rois.list"),
    ("open_project", "project.open"),
    ("save_project", "project.save"),
    ("open_roi", "project.rois.open"),
    ("open_ome_zarr", "datasets.open_ome_zarr"),
    ("open_tiff", "datasets.open_tiff"),
    (
        "open_mosaic_samplesheet",
        "datasets.open_mosaic_samplesheet",
    ),
    ("list_channels", "viewer.channels.list"),
    ("list_visible_channels", "viewer.channels.list_visible"),
    ("get_active_channel", "viewer.channels.get_active"),
    ("set_active_channel", "viewer.channels.set_active"),
    ("set_visible_channels", "viewer.channels.set_visible"),
    ("get_channel_contrast", "viewer.channels.get_contrast"),
    ("set_channel_contrast", "viewer.channels.set_contrast"),
    ("set_channel_order", "viewer.channels.set_order"),
    ("list_channel_groups", "viewer.channels.list_groups"),
    ("set_channel_group", "viewer.channels.set_group"),
    ("get_side_panels", "viewer.panels.get"),
    ("set_side_panels", "viewer.panels.set"),
    ("get_smooth_pixels", "viewer.rendering.get_smooth_pixels"),
    ("set_smooth_pixels", "viewer.rendering.set_smooth_pixels"),
    ("get_camera", "viewer.camera.get"),
    ("set_camera", "viewer.camera.set"),
    ("fit_to_view", "viewer.camera.fit"),
    ("zoom_in", "viewer.camera.zoom_in"),
    ("zoom_out", "viewer.camera.zoom_out"),
    (
        "get_object_overlay_visibility",
        "viewer.objects.get_visibility",
    ),
    (
        "set_object_overlay_visibility",
        "viewer.objects.set_visibility",
    ),
    ("get_object_selection", "viewer.objects.get_selection"),
    ("query_object_ids_in_rect", "viewer.objects.query_rect"),
    ("query_object_ids_in_view", "viewer.objects.query_view"),
    ("select_object_ids_in_rect", "viewer.objects.select_rect"),
    ("clear_object_selection", "viewer.objects.clear_selection"),
    ("get_object_filter", "viewer.objects.get_filter"),
    ("set_object_filter_query", "viewer.objects.set_filter"),
    ("clear_object_filter", "viewer.objects.clear_filter"),
    (
        "get_channel_intensity_stats",
        "viewer.channels.intensity_stats",
    ),
    ("set_right_tab", "viewer.ui.set_right_tab"),
    ("set_mosaic_right_tab", "mosaic.ui.set_right_tab"),
    ("configure_mosaic_layout", "mosaic.layout.configure"),
    ("capture_screenshot", "viewer.screenshot.capture"),
    ("capture_window_screenshot", "app.screenshot.capture"),
    ("capture_project_screenshot", "project.screenshot.capture"),
];

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

fn request_schema(shape: RequestShape) -> Value {
    match shape {
        RequestShape::Empty => json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }),
        RequestShape::SetSidePanels => json!({
            "type": "object",
            "properties": {"left": {"type": "boolean"}, "right": {"type": "boolean"}},
            "additionalProperties": false,
        }),
        RequestShape::SetSmoothPixels => json!({
            "type": "object",
            "properties": {"smooth": {"type": "boolean"}},
            "required": ["smooth"],
            "additionalProperties": false,
        }),
        RequestShape::SetVisibleChannels => json!({
            "type": "object",
            "properties": {
                "channels": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]}},
                "mode": {"type": "string", "enum": ["only", "show", "hide"]}
            },
            "required": ["channels"],
            "additionalProperties": false,
        }),
        RequestShape::SetCamera => json!({
            "type": "object",
            "properties": {
                "center_world_lvl0": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "center_x": {"type": "number"}, "center_y": {"type": "number"},
                "zoom": {"type": "number", "exclusiveMinimum": 0},
                "zoom_screen_per_lvl0_px": {"type": "number", "exclusiveMinimum": 0}
            },
            "additionalProperties": false,
        }),
        RequestShape::CaptureScreenshot => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "overwrite": {"type": "boolean", "default": false}
            },
            "additionalProperties": false,
        }),
        RequestShape::AppSettings => json!({
            "type": "object",
            "properties": {
                "fast_object_rendering": {"type": "boolean"},
                "auto_contrast": {
                    "type": "object",
                    "properties": {
                        "enabled_on_open": {"type": "boolean"},
                        "method": {"type": "string", "enum": ["zero_to_p97", "p1_to_p99", "zero_to_max"]},
                        "lower_percentile": {"type": "integer", "minimum": 0, "maximum": 99},
                        "upper_percentile": {"type": "integer", "minimum": 1, "maximum": 100}
                    },
                    "additionalProperties": false
                }
            },
            "additionalProperties": false,
        }),
        RequestShape::LifecycleRequest => json!({
            "type": "object",
            "properties": {"save": {"type": "string", "enum": ["prompt", "save", "discard"], "default": "prompt"}},
            "additionalProperties": false,
        }),
        RequestShape::SetScaleBar => json!({
            "type": "object",
            "properties": {"visible": {"type": "boolean"}},
            "required": ["visible"],
            "additionalProperties": false,
        }),
        RequestShape::ScreenshotSettings => json!({
            "type": "object",
            "properties": {
                "output_dir": {"type": ["string", "null"]},
                "include_scale_bar": {"type": "boolean"},
                "include_legend": {"type": "boolean"},
                "scale_bar_scale": {"type": "number", "minimum": 0.5, "maximum": 3.0},
                "legend_scale": {"type": "number", "minimum": 0.5, "maximum": 3.0}
            },
            "additionalProperties": false,
        }),
        RequestShape::TileLoading => json!({
            "type": "object",
            "properties": {
                "workers": {"type": "integer", "minimum": 1, "maximum": 12},
                "prefetch_mode": {"type": "string", "enum": ["off", "target_halo", "target_and_finer_halo"]},
                "prefetch_aggressiveness": {"type": "string", "enum": ["conservative", "balanced", "aggressive"]},
                "prefer_pinned_finer_levels": {"type": "boolean"}
            },
            "additionalProperties": false,
        }),
        RequestShape::MemoryPin => json!({
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0},
                "channels": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]}, "uniqueItems": true},
                "scope": {"type": "string", "enum": ["focused", "item", "all"], "default": "focused"},
                "item": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "integer", "minimum": 0}]},
                "force": {"type": "boolean", "default": false}
            },
            "required": ["level"],
            "additionalProperties": false,
        }),
        RequestShape::MemoryUnpin => json!({
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0},
                "scope": {"type": "string", "enum": ["focused", "item", "all"], "default": "focused"},
                "item": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "integer", "minimum": 0}]}
            },
            "required": ["level"],
            "additionalProperties": false,
        }),
        RequestShape::LabelLoad => json!({
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1}},
            "additionalProperties": false,
        }),
        RequestShape::LabelVisibility => json!({
            "type": "object",
            "properties": {
                "visible": {"type": "boolean"},
                "name": {"type": "string", "minLength": 1}
            },
            "required": ["visible"],
            "additionalProperties": false,
        }),
        RequestShape::ChannelPresentation => json!({
            "type": "object",
            "properties": {
                "search": {"type": "string", "maxLength": 4096},
                "sort": {"type": "string", "enum": ["manual", "name_asc", "name_desc", "visible_first", "hidden_first"]}
            },
            "minProperties": 1,
            "additionalProperties": false,
        }),
        RequestShape::MethodAvailability => json!({
            "type": "object",
            "properties": {
                "methods": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 256,
                    "uniqueItems": true
                }
            },
            "additionalProperties": false,
        }),
        RequestShape::SetPlane => json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["xy", "xz", "yz"]},
                "slice": {"type": "integer", "minimum": 0}
            },
            "anyOf": [{"required": ["mode"]}, {"required": ["slice"]}],
            "additionalProperties": false,
        }),
        RequestShape::StepPlane => json!({
            "type": "object",
            "properties": {
                "step": {"type": "integer", "minimum": 1, "default": 1},
                "wrap": {"type": "boolean", "default": false}
            },
            "additionalProperties": false,
        }),
        RequestShape::SetChannelColor => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "channel_index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
                "marker": {"type": "string"},
                "color_rgb": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 255}, "minItems": 3, "maxItems": 3}
            },
            "required": ["color_rgb"],
            "additionalProperties": false,
        }),
        RequestShape::SetChannelNote => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "channel_index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
                "marker": {"type": "string"},
                "note": {"type": "string", "maxLength": 16384}
            },
            "required": ["note"],
            "additionalProperties": false,
        }),
        RequestShape::SetChannelTransform => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "channel_index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
                "marker": {"type": "string"},
                "offset_world": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "scale": {"type": "array", "items": {"type": "number", "minimum": 0.01, "maximum": 100}, "minItems": 2, "maxItems": 2},
                "rotation_rad": {"type": "number"}
            },
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerSelector => json!({
            "type": "object",
            "properties": {
                "layer_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1}
            },
            "oneOf": [{"required": ["layer_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerVisibility => json!({
            "type": "object",
            "properties": {
                "layer_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
                "visible": {"type": "boolean"}
            },
            "required": ["visible"],
            "oneOf": [{"required": ["layer_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerOrder => json!({
            "type": "object",
            "properties": {
                "stack": {"type": "string", "enum": ["channels", "overlays"]},
                "layers": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 4096, "uniqueItems": true}
            },
            "required": ["stack", "layers"],
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerOffset => json!({
            "type": "object",
            "properties": {
                "layer_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
                "offset_world": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
            },
            "required": ["offset_world"],
            "oneOf": [{"required": ["layer_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewSelector => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "name": {"type": "string", "minLength": 1}
            },
            "oneOf": [{"required": ["index"]}, {"required": ["name"]}],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewCreate => json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "spec": {"type": "object"}
            },
            "required": ["name"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewCapture => json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "viewport_id": {"type": "string", "minLength": 1, "maxLength": 128}
            },
            "required": ["name"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewRename => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "name": {"type": "string", "minLength": 1},
                "new_name": {"type": "string", "minLength": 1}
            },
            "required": ["new_name"],
            "oneOf": [{"required": ["index"]}, {"required": ["name"]}],
            "additionalProperties": false,
        }),
        RequestShape::ProjectCreate => json!({
            "type": "object",
            "properties": {"default_dataset": {"type": "string", "minLength": 1}},
            "additionalProperties": false,
        }),
        RequestShape::Path => json!({
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectMetadata => json!({
            "type": "object",
            "properties": {
                "default_dataset": {"type": ["string", "null"]},
                "secondary_dataset": {"type": ["string", "null"]},
                "default_threshold_marker": {"type": ["string", "null"]},
                "mosaic_segmentation_search_roots": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 4096}
            },
            "additionalProperties": false,
        }),
        RequestShape::SamplesheetInspect => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 200}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::SamplesheetExport => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "overwrite": {"type": "boolean", "default": false}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::SpatialDataOpen => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "image": {"type": "string", "minLength": 1},
                "extra_images": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "labels": {"type": ["string", "null"]},
                "shapes": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "points": {"type": ["string", "null"]},
                "points_max": {"type": "integer", "minimum": 0, "maximum": 200000000, "default": 200000}
            },
            "required": ["path", "image"],
            "additionalProperties": false,
        }),
        RequestShape::XeniumOpen => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "imagery": {"type": "string", "enum": ["auto", "ome_zarr", "tiff"], "default": "auto"},
                "load_cells": {"type": "boolean", "default": true},
                "load_transcripts": {"type": "boolean", "default": true}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::HttpOpen => json!({
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
            "additionalProperties": false,
        }),
        RequestShape::S3Session => json!({
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "minLength": 1},
                "region": {"type": "string", "minLength": 1, "default": "auto"},
                "bucket": {"type": "string", "minLength": 1},
                "access_key": {"type": "string", "minLength": 1},
                "secret_key": {"type": "string", "minLength": 1}
            },
            "required": ["endpoint", "bucket", "access_key", "secret_key"],
            "additionalProperties": false,
        }),
        RequestShape::S3Prefix => json!({
            "type": "object",
            "properties": {"prefix": {"type": "string", "default": ""}},
            "additionalProperties": false,
        }),
        RequestShape::TiffOpen => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "z": {"type": "integer", "minimum": 0, "default": 0},
                "t": {"type": "integer", "minimum": 0, "default": 0}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiId => json!({
            "type": "object",
            "properties": {"id": {"type": "string", "minLength": 1}},
            "required": ["id"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiAdd => json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "path": {"type": "string", "minLength": 1},
                "display_name": {"type": "string"},
                "dataset": {"type": "string"},
                "segmentation_path": {"type": "string"},
                "metadata": {"type": "object", "additionalProperties": {"type": "string"}},
                "replacement": {"type": "object", "description": "Complete project ROI used by the native command adapter"}
            },
            "oneOf": [
                {"required": ["id", "path"], "not": {"required": ["replacement"]}},
                {
                    "required": ["replacement"],
                    "not": {"anyOf": [
                        {"required": ["id"]},
                        {"required": ["path"]},
                        {"required": ["display_name"]},
                        {"required": ["dataset"]},
                        {"required": ["segmentation_path"]},
                        {"required": ["metadata"]}
                    ]}
                }
            ],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiUpdate => json!({
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "minLength": 1},
                "changes": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                        "display_name": {"type": ["string", "null"]},
                        "dataset": {"type": ["string", "null"]},
                        "segmentation_path": {"type": ["string", "null"]},
                        "metadata": {"type": "object", "additionalProperties": {"type": "string"}}
                    },
                    "minProperties": 1,
                    "additionalProperties": false
                },
                "replacement": {"type": "object", "description": "Complete project ROI used by the native command adapter"}
            },
            "required": ["target_id"],
            "oneOf": [
                {"required": ["changes"], "not": {"required": ["replacement"]}},
                {"required": ["replacement"], "not": {"required": ["changes"]}}
            ],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiOrder => json!({
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 100000, "uniqueItems": true}},
            "required": ["ids"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiSelect => json!({
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 100000, "uniqueItems": true},
                "mode": {"type": "string", "enum": ["replace", "add", "remove", "toggle"], "default": "replace"}
            },
            "required": ["ids"],
            "additionalProperties": false,
        }),
        RequestShape::MosaicFocus => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "roi_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
                "fit": {"type": "boolean", "default": true}
            },
            "oneOf": [{"required": ["index"]}, {"required": ["roi_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::MosaicItems => json!({
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 200}
            },
            "additionalProperties": false,
        }),
        RequestShape::MosaicSelect => json!({
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "mode": {"type": "string", "enum": ["replace", "add", "remove", "toggle", "all", "range"], "default": "replace"},
                "start": {"type": "string", "minLength": 1},
                "end": {"type": "string", "minLength": 1}
            },
            "additionalProperties": false,
        }),
        RequestShape::MosaicLayout => json!({
            "type": "object",
            "properties": {
                "group_by": {"type": "string"},
                "sort_by": {"type": "string", "minLength": 1},
                "sort_by_secondary": {"type": "string", "minLength": 1},
                "sort_secondary_enabled": {"type": "boolean"},
                "show_group_labels": {"type": "boolean"},
                "show_text_labels": {"type": "boolean"},
                "group_gap": {"type": "number", "minimum": 0},
                "columns": {"type": "integer", "minimum": 1},
                "layout": {"type": "string", "enum": ["fit_cells", "native_pixels"]},
                "layout_mode": {"type": "string", "enum": ["fit_cells", "native_pixels"]},
                "label_columns": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "fit": {"type": "boolean", "default": true}
            },
            "additionalProperties": false,
        }),
        RequestShape::ObjectPreloadStart => json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["full_geometry", "centroid_points"], "default": "full_geometry"},
                "lazy_properties": {"type": "boolean", "default": true}
            },
            "additionalProperties": false,
        }),
        RequestShape::DeepLinkUri => json!({
            "type": "object",
            "properties": {"url": {"type": "string", "pattern": "^[Oo][Dd][Oo][Nn]:(//)?"}},
            "required": ["url"],
            "additionalProperties": false,
        }),
        RequestShape::DeepLinkGenerate => json!({
            "type": "object",
            "properties": {
                "request": {"type": "object"},
                "include_project": {"type": "boolean", "default": true},
                "roi": {"type": ["string", "null"]}
            },
            "additionalProperties": false,
        }),
        RequestShape::DeepLinkApply => json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "pattern": "^[Oo][Dd][Oo][Nn]:(//)?"},
                "request": {"type": "object"}
            },
            "oneOf": [{"required": ["url"]}, {"required": ["request"]}],
            "additionalProperties": false,
        }),
        RequestShape::Object => json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true,
        }),
    }
}

fn request_schema_for(descriptor: &MethodDescriptor) -> Value {
    let mut schema = request_schema(descriptor.request_shape);
    if descriptor.mutates
        && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "if_revision".to_string(),
            json!({"type": "integer", "minimum": 0}),
        );
    }
    let explicit_viewport_method = descriptor.name.starts_with("viewer.viewports.")
        && !matches!(
            descriptor.name,
            "viewer.viewports.list" | "viewer.viewports.create"
        );
    if explicit_viewport_method {
        if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
            properties.insert(
                "viewport_id".to_string(),
                json!({"type": "string", "minLength": 1, "maxLength": 128}),
            );
        }
        schema["required"] = json!(["viewport_id"]);
    }
    if matches!(
        descriptor.name,
        "viewer.viewports.camera.set"
            | "viewer.viewports.camera.fit"
            | "viewer.viewports.planes.set"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "if_navigation_revision".to_string(),
            json!({"type": "integer", "minimum": 1}),
        );
    }
    if matches!(
        descriptor.name,
        "viewer.workspace.layout.set" | "viewer.viewports.create" | "viewer.viewports.clone"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "ratio".to_string(),
            json!({"type": "number", "minimum": 0.1, "maximum": 0.9}),
        );
        if descriptor.name == "viewer.workspace.layout.set" {
            properties.insert(
                "viewports".to_string(),
                json!({
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "minItems": 1,
                    "maxItems": 2,
                    "uniqueItems": true
                }),
            );
        }
    }
    if matches!(
        descriptor.name,
        "viewer.viewport_links.create"
            | "viewer.viewport_links.update"
            | "viewer.viewport_links.remove"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "link_group_id".to_string(),
            json!({"type": "string", "const": "comparison-navigation"}),
        );
        properties.insert(
            "viewports".to_string(),
            json!({
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "minItems": 2,
                "maxItems": 2,
                "uniqueItems": true
            }),
        );
        properties.insert(
            "fields".to_string(),
            json!({
                "type": "array",
                "items": {"type": "string", "enum": ["camera", "plane", "selection"]},
                "uniqueItems": true
            }),
        );
        if descriptor.name == "viewer.viewport_links.create" {
            schema["required"] = json!(["viewports", "fields"]);
        } else if descriptor.name == "viewer.viewport_links.update" {
            schema["required"] = json!(["fields"]);
        }
    }
    if matches!(
        descriptor.name,
        "viewer.objects.selection.select_filtered"
            | "viewer.analysis.histogram"
            | "viewer.analysis.suggest_thresholds"
            | "viewer.measurements.start"
            | "exports.objects.start"
            | "exports.objects.export_csv"
            | "exports.objects.export_geoparquet"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "viewport_id".to_string(),
            json!({"type": "string", "minLength": 1, "maxLength": 128}),
        );
        properties.insert("filter_query".to_string(), json!({"type": "string"}));
        properties.insert("use_all_objects".to_string(), json!({"type": "boolean"}));
        properties.insert(
            "use_active_viewport_filter".to_string(),
            json!({"type": "boolean"}),
        );
    }
    if matches!(
        descriptor.name,
        "viewer.viewports.rename"
            | "viewer.viewports.channels.set_visible"
            | "viewer.viewports.channels.set"
            | "viewer.viewports.channels.set_active"
            | "viewer.viewports.channels.set_color"
            | "viewer.viewports.channels.set_contrast"
            | "viewer.viewports.channels.set_order"
            | "viewer.viewports.channels.set_group"
            | "viewer.viewports.rendering.set"
            | "viewer.viewports.objects.style.set"
            | "viewer.viewports.objects.legend.set"
            | "viewer.viewports.objects.filter.set"
            | "viewer.viewports.objects.filter.clear"
            | "viewer.viewports.layers.set_visibility"
            | "viewer.viewports.layers.set"
            | "viewer.viewports.layers.set_order"
            | "viewer.viewports.layers.set_active"
            | "viewer.viewports.layers.state.replace"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "if_presentation_revision".to_string(),
            json!({"type": "integer", "minimum": 1}),
        );
    }
    schema
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_are_unique_and_capabilities_are_sorted() {
        let names = METHODS
            .iter()
            .map(|descriptor| descriptor.name)
            .chain(PROTOCOL_METHODS.iter().map(|method| method.0))
            .chain(METHOD_ALIASES.iter().map(|method| method.0))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names.len(),
            METHODS.len() + PROTOCOL_METHODS.len() + METHOD_ALIASES.len()
        );

        let capabilities = capabilities();
        assert!(capabilities.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(capabilities.contains(&"system.introspect".to_string()));
    }

    #[test]
    fn every_method_has_introspection_metadata() {
        for descriptor in METHODS.iter() {
            assert!(!descriptor.name.is_empty());
            assert!(
                descriptor.name.contains('.'),
                "canonical method is not hierarchical: {}",
                descriptor.name
            );
            assert!(!descriptor.summary.is_empty(), "{}", descriptor.name);
            assert!(!descriptor.capability.is_empty(), "{}", descriptor.name);
            assert!(!descriptor.available_in.is_empty(), "{}", descriptor.name);
            assert!(
                !descriptor
                    .execution_class
                    .readiness_requirements()
                    .is_empty(),
                "{}",
                descriptor.name
            );
            assert!(request_schema(descriptor.request_shape).is_object());
        }
        assert_eq!(
            method("viewer.viewports.camera.fit")
                .unwrap()
                .execution_class,
            ExecutionClass::Geometry
        );
        assert_eq!(
            method("datasets.open_ome_zarr").unwrap().execution_class,
            ExecutionClass::Resource
        );
        assert_eq!(
            method("viewer.screenshot.capture").unwrap().execution_class,
            ExecutionClass::Presentation
        );
        let catalog = catalog_json();
        assert!(catalog.as_array().unwrap().iter().all(|entry| {
            entry.get("execution_route").is_some()
                && entry["execution_route"]["by_mode"].is_object()
        }));
    }

    #[test]
    fn execution_routes_are_mode_and_parameter_aware() {
        let camera = method("viewer.camera.set").unwrap();
        assert_eq!(
            execution_owner(camera, "single", &json!({}), false),
            ExecutionOwner::Actor
        );
        assert_eq!(
            execution_owner(camera, "mosaic", &json!({}), false),
            ExecutionOwner::Actor
        );
        let selection = method("viewer.objects.get_selection").unwrap();
        assert_eq!(
            execution_owner(
                selection,
                "single",
                &json!({"target":"segmentation_objects"}),
                false,
            ),
            ExecutionOwner::Actor
        );
        assert_eq!(
            execution_owner(
                selection,
                "single",
                &json!({"target":"spatial_shape"}),
                false,
            ),
            ExecutionOwner::LegacyUi
        );
        let route = execution_route_json(camera);
        assert_eq!(route["by_mode"]["single"]["default_owner"], "actor");
        assert_eq!(route["by_mode"]["mosaic"]["default_owner"], "actor");
        assert_eq!(execution_route_summary(camera), "actor");

        let memory = method("memory.pin").unwrap();
        assert_eq!(
            execution_owner(memory, "single", &json!({}), false),
            ExecutionOwner::Actor
        );
        assert_eq!(
            execution_owner(memory, "mosaic", &json!({}), false),
            ExecutionOwner::Actor
        );
        let route = execution_route_json(memory);
        assert_eq!(route["by_mode"]["single"]["default_owner"], "actor");
        assert_eq!(route["by_mode"]["mosaic"]["default_owner"], "actor");
    }

    #[test]
    fn report_remaining_mosaic_legacy_routes() {
        let legacy = METHODS
            .iter()
            .filter(|descriptor| descriptor.available_in.contains(&"mosaic"))
            .filter(|descriptor| {
                execution_owner(descriptor, "mosaic", &json!({}), false) == ExecutionOwner::LegacyUi
            })
            .map(|descriptor| descriptor.name)
            .collect::<Vec<_>>();
        assert_eq!(
            legacy,
            vec![
                "viewer.screenshot.capture",
                "app.screenshot.capture",
                "project.screenshot.capture",
            ]
        );
    }

    #[test]
    fn flat_names_are_deprecated_aliases_for_hierarchical_methods() {
        assert_eq!(canonical_method("set_camera"), "viewer.camera.set");
        assert_eq!(canonical_method("viewer.camera.set"), "viewer.camera.set");
        assert!(aliases_for("viewer.camera.set").contains(&"set_camera"));
        assert_eq!(method("set_camera").unwrap().name, "viewer.camera.set");
    }

    #[test]
    fn availability_reports_mode_and_accepts_alias_filters() {
        let requested = vec!["get_camera".to_string(), "project.save".to_string()];
        let single = availability_catalog("single", Some(&requested));
        let methods = single["methods"].as_array().unwrap();
        assert_eq!(methods.len(), 2);
        assert!(methods.iter().all(|method| method["available"] == true));

        let project = availability_catalog("project", Some(&requested));
        let camera = project["methods"]
            .as_array()
            .unwrap()
            .iter()
            .find(|method| method["method"] == "viewer.camera.get")
            .unwrap();
        assert_eq!(camera["available"], false);
        assert_eq!(camera["reason"], "wrong_mode");

        let transition = availability_catalog("transition", Some(&requested));
        let camera = transition["methods"]
            .as_array()
            .unwrap()
            .iter()
            .find(|method| method["method"] == "viewer.camera.get")
            .unwrap();
        assert_eq!(camera["reason"], "not_ready");
    }

    #[test]
    fn multi_viewport_registry_contracts_expose_ids_revisions_events_and_modes() {
        let camera = method("viewer.viewports.camera.set").unwrap();
        assert!(camera.mutates);
        assert_eq!(camera.event, Some("viewer.viewports.navigation.changed"));
        assert_eq!(camera.available_in, SINGLE_MODE);
        let camera_schema = request_schema_for(camera);
        assert_eq!(camera_schema["required"], json!(["viewport_id"]));
        assert_eq!(
            camera_schema["properties"]["if_navigation_revision"]["minimum"],
            1
        );

        let style = method("viewer.viewports.objects.style.set").unwrap();
        assert_eq!(style.event, Some("viewer.viewports.presentation.changed"));
        let style_schema = request_schema_for(style);
        assert_eq!(
            style_schema["properties"]["if_presentation_revision"]["minimum"],
            1
        );

        let links = request_schema_for(method("viewer.viewport_links.create").unwrap());
        assert_eq!(links["required"], json!(["viewports", "fields"]));
        assert_eq!(links["properties"]["viewports"]["minItems"], 2);

        for name in [
            "viewer.workspace.get",
            "viewer.workspace.layout.get",
            "viewer.workspace.layout.set",
            "viewer.workspace.swap",
            "viewer.viewports.list",
            "viewer.viewports.get",
            "viewer.viewports.create",
            "viewer.viewports.clone",
            "viewer.viewports.rename",
            "viewer.viewports.remove",
            "viewer.viewports.set_active",
            "viewer.viewport_links.set",
            "viewer.viewport_links.get",
            "viewer.viewport_links.list",
            "viewer.viewport_links.create",
            "viewer.viewport_links.update",
            "viewer.viewport_links.remove",
            "viewer.viewports.camera.get",
            "viewer.viewports.camera.set",
            "viewer.viewports.camera.fit",
            "viewer.viewports.planes.get",
            "viewer.viewports.planes.set",
            "viewer.viewports.channels.get",
            "viewer.viewports.channels.set_visible",
            "viewer.viewports.channels.set",
            "viewer.viewports.channels.set_active",
            "viewer.viewports.channels.set_color",
            "viewer.viewports.channels.set_contrast",
            "viewer.viewports.channels.set_order",
            "viewer.viewports.channels.list_groups",
            "viewer.viewports.channels.set_group",
            "viewer.viewports.rendering.get",
            "viewer.viewports.rendering.set",
            "viewer.viewports.objects.style.get",
            "viewer.viewports.objects.style.set",
            "viewer.viewports.objects.legend.set",
            "viewer.viewports.objects.filter.get",
            "viewer.viewports.objects.filter.set",
            "viewer.viewports.objects.filter.clear",
            "viewer.viewports.layers.list",
            "viewer.viewports.layers.get",
            "viewer.viewports.layers.set",
            "viewer.viewports.layers.set_visibility",
            "viewer.viewports.layers.set_order",
            "viewer.viewports.layers.set_active",
            "viewer.workspace.screenshot.capture",
        ] {
            assert!(method(name).is_some(), "missing registry method {name}");
        }
    }

    #[test]
    fn actor_capability_registry_has_unique_known_methods() {
        let application_methods = METHODS
            .iter()
            .map(|descriptor| descriptor.name)
            .collect::<BTreeSet<_>>();
        let protocol_methods = PROTOCOL_METHODS
            .iter()
            .map(|descriptor| descriptor.0)
            .collect::<BTreeSet<_>>();
        let actor_only_methods = BTreeSet::from(["app.get_method_availability"]);
        let mut seen = BTreeSet::new();

        for name in ACTOR_CAPABLE_METHODS {
            assert!(seen.insert(*name), "duplicate actor-capable method {name}");
            assert!(
                application_methods.contains(name)
                    || protocol_methods.contains(name)
                    || actor_only_methods.contains(name),
                "actor-capable method is absent from every registry: {name}"
            );
        }
    }
}
