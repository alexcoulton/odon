use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::document::{DocumentDescriptor, DocumentObjectLayerResource};
use crate::data::ome::{ChannelInfo, DatasetRenderKind, OmeZarrDataset};
use crate::data::project_config::{
    ProjectChannelGroup, ProjectChannelGroupMember, ProjectLayerGroups, ProjectMaskLayer,
    ProjectRoi,
};
use crate::deep_link::{
    DeepLinkChannelColor, DeepLinkChannelContrast, DeepLinkChannelOrder,
    DeepLinkObjectFilterClause, DeepLinkObjectFilterLogic, DeepLinkRequest, object_filter_model,
    object_segmentation_requested, requested_bundled_label,
};
use crate::settings::{AppSettings, AutoContrastMethod, AutoContrastSettings};
use crate::viewports::{ViewportId, ViewportLayout, ViewportLinks, ViewportWorkspace};

use super::layers::NativeLayersModel;
use super::project::{ProjectModel, ProjectModelSnapshot};
use super::{
    AnalysisModel, AnnotationModel, ControlAnnotationLayerProjection, ControlLabelResource,
    ControlMosaicResource, ControlObjectFilterResult, ControlObjectResource,
    ControlPinnedLevelResource, ControlSegmentationGeoJsonResource,
    ControlThresholdPreviewResource, LabelZarrDataset, MaskModel, MeasurementMetric,
    MeasurementModel, MosaicModel, MosaicObjectLoadResult, MosaicObjectLoadSpec,
    ObjectExportFormat, ObjectExportModel, ObjectExportResult, ObjectExportSpec,
    ObjectSelectionModel, OperationKind, PinnedMemoryModel, ProjectObjectPreloadCatalog,
    ProjectObjectPreloadProjection, ProjectObjectPreloadScope, ProjectObjectPreloadSettings,
    ProjectObjectPreloadSource, ReadinessModel, ScreenshotPreferences, SegmentationGeoJsonLoadSpec,
    SegmentationGeoJsonModel, SystemMemorySnapshot, ThresholdPreviewModel, ThresholdScope,
    TileLoadingModel, TileLoadingPolicy, default_screenshot_filename, object_export_columns,
    parse_world_points, parse_world_rect, project_object_preload_candidates,
    project_roi_segmentation_path,
};

mod analysis;
mod annotations;
mod channel_compute;
mod construction;
mod dispatch;
mod masks;
mod measurements;
mod native_layers;
mod object_exports;
mod object_resources;
mod objects;
mod preferences_memory;
mod project_views;
mod projects;
mod runtime;
mod screenshots;
mod segmentation_geojson;
mod settings_deep_links;
mod thresholds;
mod viewport_commands;

use channel_compute::ChannelComputeModel;

const DEFAULT_LOGICAL_CANVAS: [f32; 2] = [960.0, 720.0];
const DEFAULT_LEFT_PANEL_WIDTH: f32 = 360.0;
const DEFAULT_RIGHT_PANEL_WIDTH: f32 = 380.0;
const THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS: u64 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelMode {
    Project,
    Single,
    Mosaic,
    Transition,
}

impl ModelMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Single => "single",
            Self::Mosaic => "mosaic",
            Self::Transition => "transition",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkRequestKind {
    Direct,
    Create,
    Update,
}

#[derive(Debug, Clone, PartialEq)]
struct ModelChannel {
    index: usize,
    name: String,
    visible: bool,
    color_rgb: [u8; 3],
    window: Option<(f32, f32)>,
    contrast_manual: bool,
    note: String,
    offset_world: [f32; 2],
    scale: [f32; 2],
    rotation_rad: f32,
}

impl From<&ChannelInfo> for ModelChannel {
    fn from(channel: &ChannelInfo) -> Self {
        Self {
            index: channel.index,
            name: channel.name.clone(),
            visible: channel.visible,
            color_rgb: channel.color_rgb,
            window: channel.window,
            contrast_manual: false,
            note: channel.note.clone(),
            offset_world: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation_rad: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ViewportModel {
    center: [f32; 2],
    zoom: f32,
    logical_size: [f32; 2],
    screen_origin: [f32; 2],
    plane_mode: String,
    plane_slices: [u64; 3],
    channels: Vec<ModelChannel>,
    active_channel: usize,
    channel_order: Vec<usize>,
    channel_sort: String,
    channel_search: String,
    channel_groups: ProjectLayerGroups,
    smooth_pixels: bool,
    show_scale_bar: bool,
    show_hud: bool,
    show_tile_debug: bool,
    objects: Value,
    segmentation_labels_visible: bool,
    segmentation_geojson_visible: bool,
    object_filter_indices: Arc<Vec<usize>>,
    object_filter_active: bool,
    object_filter_revision: u64,
    secondary_objects: HashMap<u64, SecondaryObjectViewportModel>,
    native_layers: NativeLayersModel,
}

#[derive(Debug, Clone, PartialEq)]
struct SecondaryObjectViewportModel {
    objects: Value,
    filter_indices: Arc<Vec<usize>>,
    filter_active: bool,
    filter_revision: u64,
}

impl SecondaryObjectViewportModel {
    fn new() -> Self {
        let mut objects = default_object_snapshot();
        let object = objects
            .as_object_mut()
            .expect("object presentation defaults are an object");
        object.insert("visible".to_string(), Value::Bool(true));
        object.insert("width_screen_px".to_string(), json!(1.0_f32));
        object.insert("color_rgb".to_string(), json!([0, 255, 120]));
        Self {
            objects,
            filter_indices: Arc::new(Vec::new()),
            filter_active: false,
            filter_revision: 1,
        }
    }
}

#[derive(Debug, Clone)]
struct SecondaryObjectLayerModel {
    layer_id: u64,
    name: String,
    generation: u64,
    resource: Arc<ControlObjectResource>,
    selection: ObjectSelectionModel,
}

impl ViewportModel {
    fn new(channels: &[ChannelInfo]) -> Self {
        let channel_order = channels.iter().map(|channel| channel.index).collect();
        Self {
            center: [0.0, 0.0],
            zoom: 1.0,
            logical_size: DEFAULT_LOGICAL_CANVAS,
            screen_origin: [0.0, 0.0],
            plane_mode: "xy".to_string(),
            plane_slices: [0, 0, 0],
            channels: channels.iter().map(ModelChannel::from).collect(),
            active_channel: 0,
            channel_order,
            channel_sort: "manual".to_string(),
            channel_search: String::new(),
            channel_groups: ProjectLayerGroups::default(),
            smooth_pixels: true,
            show_scale_bar: true,
            show_hud: true,
            show_tile_debug: false,
            objects: default_object_snapshot(),
            segmentation_labels_visible: true,
            segmentation_geojson_visible: false,
            object_filter_indices: Arc::new(Vec::new()),
            object_filter_active: false,
            object_filter_revision: 1,
            secondary_objects: HashMap::new(),
            native_layers: NativeLayersModel::channels(channels),
        }
    }

    fn object_presentation(&self, target: ObjectTarget) -> Option<&Value> {
        match target {
            ObjectTarget::Primary => Some(&self.objects),
            ObjectTarget::SpatialShape(id) => {
                self.secondary_objects.get(&id).map(|state| &state.objects)
            }
        }
    }

    fn object_presentation_mut(&mut self, target: ObjectTarget) -> Option<&mut Value> {
        match target {
            ObjectTarget::Primary => Some(&mut self.objects),
            ObjectTarget::SpatialShape(id) => self
                .secondary_objects
                .get_mut(&id)
                .map(|state| &mut state.objects),
        }
    }

    fn object_filter_state(&self, target: ObjectTarget) -> Option<(&Arc<Vec<usize>>, bool, u64)> {
        match target {
            ObjectTarget::Primary => Some((
                &self.object_filter_indices,
                self.object_filter_active,
                self.object_filter_revision,
            )),
            ObjectTarget::SpatialShape(id) => self.secondary_objects.get(&id).map(|state| {
                (
                    &state.filter_indices,
                    state.filter_active,
                    state.filter_revision,
                )
            }),
        }
    }

    fn replace_object_filter_state(
        &mut self,
        target: ObjectTarget,
        model: Value,
        indices: Arc<Vec<usize>>,
        active: bool,
    ) -> Option<u64> {
        let objects = self.object_presentation_mut(target)?;
        set_object_filter_model(objects, model);
        match target {
            ObjectTarget::Primary => {
                self.object_filter_indices = indices;
                self.object_filter_active = active;
                self.object_filter_revision = self.object_filter_revision.wrapping_add(1).max(1);
                Some(self.object_filter_revision)
            }
            ObjectTarget::SpatialShape(id) => {
                let state = self.secondary_objects.get_mut(&id)?;
                state.filter_indices = indices;
                state.filter_active = active;
                state.filter_revision = state.filter_revision.wrapping_add(1).max(1);
                Some(state.filter_revision)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DatasetModel {
    descriptor: DocumentDescriptor,
    world_size: [f32; 2],
    plane_extents: [u64; 3],
    orthogonal_planes: bool,
    logical_workspace_size: [f32; 2],
    geometry_source: GeometrySource,
    show_left_panel: bool,
    show_right_panel: bool,
    left_tab: String,
    right_tab: String,
    shared_resources: Value,
    performance: Value,
    object_resource: Option<std::sync::Arc<ControlObjectResource>>,
    segmentation_geojson: SegmentationGeoJsonModel,
    label_available: Vec<String>,
    label_selected: String,
    label_loaded: Option<String>,
    label_resource: Option<Arc<ControlLabelResource>>,
    label_status: String,
    label_generation: u64,
    label_pending: bool,
    label_actor_owned: bool,
    object_selection: ObjectSelectionModel,
    secondary_object_layers: HashMap<u64, SecondaryObjectLayerModel>,
    masks: MaskModel,
    workspace: ViewportWorkspace<ViewportModel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeometrySource {
    Bootstrap,
    Derived,
    Observed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ObjectTarget {
    Primary,
    SpatialShape(u64),
}

#[derive(Clone)]
pub struct ControlSecondaryObjectProjection {
    pub layer_id: u64,
    pub name: String,
    pub generation: u64,
    pub resource: Arc<ControlObjectResource>,
    pub selection: Value,
    pub analysis_generation: u64,
    pub analysis_state: Value,
}

impl ObjectTarget {
    fn response_name(self) -> &'static str {
        match self {
            Self::Primary => "segmentation_objects",
            Self::SpatialShape(_) => "spatial_shape",
        }
    }

    fn layer_id(self) -> String {
        match self {
            Self::Primary => "segmentation_objects".to_string(),
            Self::SpatialShape(id) => format!("spatial_shape:{id}"),
        }
    }
}

fn object_target_params(target: ObjectTarget) -> Value {
    match target {
        ObjectTarget::Primary => json!({"target":"segmentation_objects"}),
        ObjectTarget::SpatialShape(id) => json!({"target":"spatial_shape","layer_id":id}),
    }
}

fn object_target_not_found(target: ObjectTarget) -> ControlError {
    ControlError::new(
        ControlErrorKind::ResourceNotFound,
        format!("object target '{}' is not loaded", target.layer_id()),
    )
}

impl GeometrySource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Derived => "derived",
            Self::Observed => "observed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AppModel {
    mode: ModelMode,
    dataset: Option<DatasetModel>,
    project: ProjectModel,
    project_initialized: bool,
    readiness: ReadinessModel,
    projection_revision: u64,
    presented_projection_revision: u64,
    document_generation: u64,
    dataset_inspection_operation_generation: u64,
    remote_listing_operation_generation: u64,
    deep_link_resolve_operation_generation: u64,
    deep_link_apply_operation_generation: u64,
    deep_link_apply_pending: bool,
    project_view_apply_generation: u64,
    project_view_apply_pending: bool,
    project_operation_generation: u64,
    project_operation_pending: bool,
    project_object_preload: ProjectObjectPreloadCatalog,
    project_roi_open_generation: u64,
    project_roi_open_pending: bool,
    object_resource_generation: u64,
    installed_object_resource_generation: u64,
    object_resource_pending: bool,
    object_filter_operation_generation: u64,
    pending_object_filters: HashMap<(ViewportId, ObjectTarget), u64>,
    object_selection_filter_operation_generation: u64,
    pending_object_selection_filters: HashMap<ObjectTarget, u64>,
    mask_io_operation_generation: u64,
    channel_compute: ChannelComputeModel,
    settings: AppSettings,
    recent_project_exists: HashMap<PathBuf, bool>,
    settings_path: Option<PathBuf>,
    settings_status: String,
    settings_operation_generation: u64,
    settings_operation_pending: bool,
    screenshot_preferences: ScreenshotPreferences,
    screenshot_settings_generation: u64,
    screenshot_settings_pending: bool,
    tile_loading: TileLoadingModel,
    pinned_memory: PinnedMemoryModel,
    threshold_preview: ThresholdPreviewModel,
    annotations: AnnotationModel,
    analyses: HashMap<ObjectTarget, AnalysisModel>,
    measurement: MeasurementModel,
    object_export: ObjectExportModel,
    mosaic: MosaicModel,
    mosaic_operation_generation: u64,
    mosaic_operation_pending: bool,
    measured_viewports: HashSet<ViewportId>,
    renderer_gpu_available: bool,
}

#[derive(Debug)]
pub struct ModelDispatch {
    pub response: Value,
    pub present: bool,
}

#[derive(Debug, Clone)]
pub struct SettingsSaveOperation {
    pub generation: u64,
    pub path: PathBuf,
    pub settings: AppSettings,
    pub response: Value,
}

#[derive(Debug, Clone)]
pub enum SettingsMutationOutcome {
    Immediate(Value),
    Persist(SettingsSaveOperation),
}

#[derive(Debug, Clone)]
pub struct ChannelIntensitySpec {
    pub channel_index: usize,
    pub channel_name: String,
    pub level_number: usize,
    pub downsample: f32,
    pub zarr_path: String,
    pub dtype: String,
    pub ranges: Vec<Range<u64>>,
    pub bins: Option<usize>,
    pub abs_max: f32,
    pub client_request_id: Option<u64>,
    pub source_channel: usize,
    pub region: [u64; 4],
}

#[derive(Debug, Clone)]
pub(crate) struct AutoContrastSpec {
    pub(crate) document_generation: u64,
    pub(crate) operation_generation: u64,
    pub(crate) viewport_id: ViewportId,
    pub(crate) settings: AutoContrastSettings,
    pub(crate) channels: Vec<AutoContrastChannelSpec>,
}

#[derive(Debug, Clone)]
pub(crate) struct AutoContrastChannelSpec {
    pub(crate) intensity: ChannelIntensitySpec,
    pub(crate) baseline_window: Option<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub(crate) struct AutoContrastChannelResult {
    pub(crate) channel_index: usize,
    pub(crate) channel_name: String,
    pub(crate) min: u16,
    pub(crate) max: u16,
    pub(crate) sample_count: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryPinSpec {
    pub document_generation: u64,
    pub operation_generation: u64,
    pub level: usize,
    pub channel_ids: Vec<u64>,
    pub estimated_bytes: u64,
    pub pinned_bytes: u64,
    pub force: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ThresholdPreviewLoadSpec {
    pub(crate) document_generation: u64,
    pub(crate) operation_generation: u64,
    pub(crate) channel_index: usize,
    pub(crate) channel_name: String,
    pub(crate) scope: ThresholdScope,
    pub(crate) level: usize,
    pub(crate) downsample: f32,
    pub(crate) x0: u64,
    pub(crate) y0: u64,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) zarr_path: String,
    pub(crate) dtype: String,
    pub(crate) ranges: Vec<Range<u64>>,
    pub(crate) threshold: u16,
    pub(crate) min_component_pixels: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ThresholdPreviewRecomputeSpec {
    pub(crate) document_generation: u64,
    pub(crate) operation_generation: u64,
    pub(crate) preview: Arc<ControlThresholdPreviewResource>,
}

#[derive(Debug, Clone)]
pub(crate) struct ThresholdPreviewApplySpec {
    pub(crate) document_generation: u64,
    pub(crate) operation_generation: u64,
    pub(crate) preview: Arc<ControlThresholdPreviewResource>,
    pub(crate) pivot: [f32; 2],
    pub(crate) offset: [f32; 2],
    pub(crate) scale: [f32; 2],
    pub(crate) rotation_rad: f32,
}

#[derive(Clone)]
pub(crate) struct AnalysisResourceSpec {
    pub(crate) document_generation: u64,
    pub(crate) resource_generation: u64,
    pub(crate) operation_generation: u64,
    pub(crate) operation_scope: String,
    pub(crate) target: ObjectTarget,
    pub(crate) resource: Arc<ControlObjectResource>,
    pub(crate) indices: Option<Arc<Vec<usize>>>,
    pub(crate) filtered: bool,
}

#[derive(Clone)]
pub(crate) struct MeasurementSpec {
    pub(crate) document_generation: u64,
    pub(crate) resource_generation: u64,
    pub(crate) operation_generation: u64,
    pub(crate) target: ObjectTarget,
    pub(crate) level: usize,
    pub(crate) metric: MeasurementMetric,
    pub(crate) prefix: String,
    pub(crate) resource: Arc<ControlObjectResource>,
    pub(crate) target_indices: Arc<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeepLinkApplyGuard {
    pub(crate) project_load_generation: u64,
    pub(crate) project_config_generation: u64,
    pub(crate) document_generation: u64,
}

#[derive(Clone)]
pub(crate) struct DeepLinkCurrentResources {
    pub(crate) source_key: String,
    pub(crate) object: Option<Arc<ControlObjectResource>>,
    pub(crate) label_available: Vec<String>,
    pub(crate) label_loaded: Option<String>,
    pub(crate) label: Option<Arc<ControlLabelResource>>,
}

#[derive(Clone)]
pub(crate) struct ProjectViewApplySpec {
    pub(crate) operation_generation: u64,
    pub(crate) document_generation: u64,
    pub(crate) project_config_generation: u64,
    pub(crate) params: Value,
    pub(crate) object_path: Option<PathBuf>,
    pub(crate) label_name: Option<String>,
}

fn update_logical_geometry(dataset: &mut DatasetModel) {
    let layout = dataset.workspace.layout();
    let ratio = dataset.workspace.split_ratio();
    let whole = dataset.logical_workspace_size;
    let ids = dataset
        .workspace
        .viewports()
        .iter()
        .map(|slot| slot.id.clone())
        .collect::<Vec<_>>();
    for (index, id) in ids.into_iter().enumerate() {
        let size = match layout {
            ViewportLayout::Single => whole,
            ViewportLayout::Horizontal => [
                whole[0] * if index == 0 { ratio } else { 1.0 - ratio },
                whole[1],
            ],
            ViewportLayout::Vertical => [
                whole[0],
                whole[1] * if index == 0 { ratio } else { 1.0 - ratio },
            ],
        };
        if let Some(slot) = dataset.workspace.get_mut(&id) {
            slot.state.logical_size = size;
        }
    }
    if dataset.geometry_source == GeometrySource::Observed {
        dataset.geometry_source = GeometrySource::Derived;
    }
}

fn renderer_viewport_size(value: &Value) -> Option<[f32; 2]> {
    let viewport = value.get("camera")?.get("viewport")?;
    if let Some(rect) = viewport.as_array().filter(|rect| rect.len() == 4) {
        let width = (rect[2].as_f64()? - rect[0].as_f64()?) as f32;
        let height = (rect[3].as_f64()? - rect[1].as_f64()?) as f32;
        return (width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0)
            .then_some([width, height]);
    }
    let rect = viewport.get("screen_rect")?.as_array()?;
    if rect.len() != 4 {
        return None;
    }
    let width = (rect[2].as_f64()? - rect[0].as_f64()?) as f32;
    let height = (rect[3].as_f64()? - rect[1].as_f64()?) as f32;
    (width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0)
        .then_some([width, height])
}

fn project_roi_view_workspace_snapshot(
    view: &Value,
    channel_count: usize,
    base: &Value,
) -> Result<Value, ControlError> {
    let mut snapshot = base.clone();
    if let Some(workspace) = view.get("workspace").filter(|value| value.is_object()) {
        let saved_viewports = workspace
            .get("viewports")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("saved project workspace has no viewport array"))?;
        let base_viewport = base
            .get("viewports")
            .and_then(Value::as_array)
            .and_then(|viewports| viewports.first())
            .ok_or_else(|| invalid("installed document has no base viewport"))?;
        let projected = saved_viewports
            .iter()
            .map(|saved| overlay_project_viewport(base_viewport, saved, channel_count))
            .collect::<Result<Vec<_>, _>>()?;
        snapshot["viewports"] = Value::Array(projected);
        for (saved_name, projected_name) in [
            ("layout", "layout"),
            ("split_ratio", "ratio"),
            ("active_viewport_id", "active_viewport_id"),
        ] {
            if let Some(value) = workspace.get(saved_name) {
                snapshot[projected_name] = value.clone();
            }
        }
        snapshot["links"] = json!({
            "camera": workspace.get("link_camera").and_then(Value::as_bool).unwrap_or(true),
            "plane": workspace.get("link_plane").and_then(Value::as_bool).unwrap_or(true),
            "selection": workspace.get("link_selection").and_then(Value::as_bool).unwrap_or(true),
        });
    } else {
        let base_viewport = base
            .get("viewports")
            .and_then(Value::as_array)
            .and_then(|viewports| viewports.first())
            .ok_or_else(|| invalid("installed document has no base viewport"))?;
        snapshot["viewports"] = Value::Array(vec![overlay_project_viewport(
            base_viewport,
            view,
            channel_count,
        )?]);
        snapshot["layout"] = Value::String("single".to_string());
        snapshot["active_viewport_id"] = Value::String("viewport-1".to_string());
    }

    if let Some(ui) = view.get("ui") {
        if let Some(left) = ui.get("show_left_panel") {
            snapshot["panels"]["left"] = left.clone();
        }
        if let Some(right) = ui.get("show_right_panel") {
            snapshot["panels"]["right"] = right.clone();
        }
        if let Some(right_tab) = ui.get("right_tab") {
            snapshot["ui"]["right_tab"] = right_tab.clone();
        }
        if let Some(left_tab) = ui.get("left_tab") {
            snapshot["ui"]["left_tab"] = left_tab.clone();
        }
        if let Some(active) = snapshot
            .get_mut("viewports")
            .and_then(Value::as_array_mut)
            .and_then(|viewports| viewports.first_mut())
        {
            for (saved_name, projected_name) in [
                ("smooth_pixels", "smooth_pixels"),
                ("show_scale_bar", "show_scale_bar"),
                ("show_hud", "show_hud"),
                ("show_tile_debug", "show_tile_debug"),
            ] {
                if let Some(value) = ui.get(saved_name) {
                    active["rendering"][projected_name] = value.clone();
                }
            }
            if let Some(sort) = ui.get("channel_sort") {
                active["channel_sort"] = sort.clone();
            }
        }
    }
    Ok(snapshot)
}

fn overlay_project_viewport(
    base: &Value,
    saved: &Value,
    channel_count: usize,
) -> Result<Value, ControlError> {
    let mut projected = base.clone();
    if let Some(presentation) = saved.get("presentation") {
        for name in [
            "channel_groups",
            "channel_sort",
            "objects",
            "object_overlay_visibility",
            "native_layers",
            "rendering",
        ] {
            if let Some(value) = presentation.get(name) {
                projected[name] = value.clone();
            }
        }
    }
    for (saved_name, projected_name) in [
        ("id", "viewport_id"),
        ("title", "title"),
        ("navigation_revision", "navigation_revision"),
        ("presentation_revision", "presentation_revision"),
    ] {
        if let Some(value) = saved.get(saved_name) {
            projected[projected_name] = value.clone();
        }
    }
    if let Some(camera) = saved.get("camera") {
        projected["camera"] = camera.clone();
    }
    if let Some(mode) = saved.get("plane_mode").and_then(Value::as_str) {
        let slice_name = match mode.to_ascii_lowercase().as_str() {
            "yz" => "x_level0",
            "xz" => "y_level0",
            _ => "z_level0",
        };
        projected["plane"] = json!({
            "mode": mode,
            "slice": saved.get(slice_name).and_then(Value::as_u64).unwrap_or(0),
        });
    }
    if let Some(order) = saved.get("channel_order").and_then(Value::as_array) {
        let indices = order.iter().filter_map(Value::as_u64).collect::<Vec<_>>();
        let unique = indices.iter().copied().collect::<HashSet<_>>();
        if indices.len() == channel_count && unique.len() == channel_count {
            projected["channel_order"] = Value::Array(order.clone());
        }
    }
    if let Some(channels) = saved.get("channels").and_then(Value::as_array)
        && let Some(projected_channels) =
            projected.get_mut("channels").and_then(Value::as_array_mut)
    {
        for (index, channel) in channels.iter().take(channel_count).enumerate() {
            let Some(target) = projected_channels.get_mut(index) else {
                continue;
            };
            for name in [
                "visible",
                "color_rgb",
                "window",
                "offset_world",
                "scale",
                "rotation_rad",
                "note",
            ] {
                if let Some(value) = channel.get(name) {
                    target[name] = value.clone();
                }
            }
        }
    }
    if let Some(active) = saved.get("active_channel").and_then(Value::as_u64)
        && let Some(projected_channels) =
            projected.get_mut("channels").and_then(Value::as_array_mut)
    {
        for (index, channel) in projected_channels.iter_mut().enumerate() {
            channel["selected"] = Value::Bool(index as u64 == active);
        }
    }

    let mut objects = projected
        .get("objects")
        .cloned()
        .unwrap_or_else(default_object_snapshot);
    if let Some(display) = saved
        .get("segmentation")
        .and_then(|segmentation| segmentation.get("object_display"))
    {
        for (saved_name, projected_name) in [
            ("color_property_key", "color_property"),
            ("color_level_overrides", "color_level_overrides"),
            ("fill_cells", "fill_cells"),
            ("fill_opacity", "fill_opacity"),
            ("selected_fill_opacity", "selected_fill_opacity"),
            ("fast_rendering", "fast_rendering"),
        ] {
            if let Some(value) = display.get(saved_name) {
                objects[projected_name] = value.clone();
            }
        }
    }
    if let Some(analysis) = saved.get("analysis")
        && let Some(value) = analysis.get("show_selection_overlay")
    {
        objects["show_selection_overlay"] = value.clone();
    }
    for (saved_name, projected_name) in [
        ("object_visible", "visible"),
        ("object_opacity", "opacity"),
        ("object_width_screen_px", "width_screen_px"),
        ("object_color_rgb", "color_rgb"),
        ("object_show_selection_overlay", "show_selection_overlay"),
    ] {
        if let Some(value) = saved.get(saved_name) {
            objects[projected_name] = value.clone();
        }
    }
    if let Some(filter) = saved.get("object_filter") {
        objects["filter"] = filter.clone();
    }
    projected["objects"] = objects;
    Ok(projected)
}

fn apply_workspace_viewport(state: &mut ViewportModel, value: &Value) -> Result<(), ControlError> {
    if let Some(camera) = value.get("camera") {
        if let Some(center) = camera
            .get("center_world_lvl0")
            .and_then(Value::as_array)
            .filter(|center| center.len() == 2)
        {
            let center = [
                center[0]
                    .as_f64()
                    .ok_or_else(|| invalid("renderer camera center x is invalid"))?
                    as f32,
                center[1]
                    .as_f64()
                    .ok_or_else(|| invalid("renderer camera center y is invalid"))?
                    as f32,
            ];
            if !center.iter().all(|value| value.is_finite()) {
                return Err(invalid("renderer camera center is not finite"));
            }
            state.center = center;
        }
        if let Some(zoom) = camera
            .get("zoom_screen_per_lvl0_px")
            .and_then(Value::as_f64)
        {
            let zoom = zoom as f32;
            if !zoom.is_finite() || zoom <= 0.0 {
                return Err(invalid("renderer camera zoom is invalid"));
            }
            state.zoom = zoom;
        }
        if let Some(size) = renderer_viewport_size(value) {
            state.logical_size = size;
        }
    }
    if let Some(plane) = value.get("plane") {
        if let Some(mode) = plane.get("mode").and_then(Value::as_str) {
            state.plane_mode = normalize_plane_mode(mode)?.to_string();
        }
        if let Some(slice) = plane.get("slice").and_then(Value::as_u64) {
            let index = plane_mode_index(&state.plane_mode);
            state.plane_slices[index] = slice;
        }
    }
    if let Some(channels) = value.get("channels").and_then(Value::as_array) {
        for projected in channels {
            let Some(index) = projected
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
            else {
                continue;
            };
            let Some(channel) = state
                .channels
                .iter_mut()
                .find(|channel| channel.index == index)
            else {
                continue;
            };
            if let Some(visible) = projected.get("visible").and_then(Value::as_bool) {
                channel.visible = visible;
            }
            if let Some(color) = projected
                .get("color_rgb")
                .and_then(Value::as_array)
                .filter(|color| color.len() >= 3)
            {
                channel.color_rgb = [to_u8(&color[0])?, to_u8(&color[1])?, to_u8(&color[2])?];
            }
            channel.window = projected.get("window").and_then(|window| {
                if window.is_null() {
                    return None;
                }
                if let Some(values) = window.as_array().filter(|values| values.len() == 2) {
                    return Some((values[0].as_f64()? as f32, values[1].as_f64()? as f32));
                }
                Some((
                    window.get("min")?.as_f64()? as f32,
                    window.get("max")?.as_f64()? as f32,
                ))
            });
            channel.contrast_manual = channel.window.is_some();
            if let Some(offset) = projected
                .get("offset_world")
                .and_then(Value::as_array)
                .filter(|values| values.len() == 2)
            {
                channel.offset_world = [
                    offset[0]
                        .as_f64()
                        .ok_or_else(|| invalid("renderer channel offset x is invalid"))?
                        as f32,
                    offset[1]
                        .as_f64()
                        .ok_or_else(|| invalid("renderer channel offset y is invalid"))?
                        as f32,
                ];
            }
            if let Some(scale) = projected
                .get("scale")
                .and_then(Value::as_array)
                .filter(|values| values.len() == 2)
            {
                channel.scale = [
                    scale[0]
                        .as_f64()
                        .ok_or_else(|| invalid("renderer channel scale x is invalid"))?
                        as f32,
                    scale[1]
                        .as_f64()
                        .ok_or_else(|| invalid("renderer channel scale y is invalid"))?
                        as f32,
                ];
            }
            if let Some(rotation) = projected.get("rotation_rad").and_then(Value::as_f64) {
                channel.rotation_rad = rotation as f32;
            }
            if let Some(note) = projected.get("note").and_then(Value::as_str) {
                channel.note = note.to_string();
            }
            if projected
                .get("selected")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                state.active_channel = index;
            }
        }
    }
    apply_renderer_channel_presentation(state, value)?;
    if let Some(rendering) = value.get("rendering") {
        if let Some(value) = rendering.get("smooth_pixels").and_then(Value::as_bool) {
            state.smooth_pixels = value;
        }
        if let Some(value) = rendering.get("show_scale_bar").and_then(Value::as_bool) {
            state.show_scale_bar = value;
        }
        if let Some(value) = rendering.get("show_hud").and_then(Value::as_bool) {
            state.show_hud = value;
        }
        if let Some(value) = rendering.get("show_tile_debug").and_then(Value::as_bool) {
            state.show_tile_debug = value;
        }
    }
    if let Some(objects) = value.get("objects") {
        state.objects = objects.clone();
    }
    if let Some(overlays) = value.get("object_overlay_visibility") {
        if let Some(visible) = overlays.get("segmentation_labels").and_then(Value::as_bool) {
            state.segmentation_labels_visible = visible;
        }
        if let Some(visible) = overlays
            .get("segmentation_geojson")
            .and_then(Value::as_bool)
        {
            state.segmentation_geojson_visible = visible;
        }
    }
    if let Some(native_layers) = value.get("native_layers") {
        state.native_layers = NativeLayersModel::restore(native_layers)?;
    }
    Ok(())
}

fn apply_renderer_channel_presentation(
    state: &mut ViewportModel,
    value: &Value,
) -> Result<(), ControlError> {
    if let Some(order) = value.get("channel_order").and_then(Value::as_array) {
        let order = order
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|index| usize::try_from(index).ok())
                    .filter(|index| *index < state.channels.len())
                    .ok_or_else(|| invalid("renderer channel order contains an invalid index"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unique = order.iter().copied().collect::<HashSet<_>>();
        if order.len() != state.channels.len() || unique.len() != order.len() {
            return Err(invalid(
                "renderer channel order must contain every channel exactly once",
            ));
        }
        state.channel_order = order;
    }
    if let Some(sort) = value.get("channel_sort").and_then(Value::as_str) {
        state.channel_sort = canonical_channel_sort(sort)
            .ok_or_else(|| invalid(format!("renderer channel sort mode '{sort}' is invalid")))?
            .to_string();
    }
    if let Some(groups) = value.get("channel_groups").and_then(Value::as_array) {
        state.channel_groups = parse_channel_groups_snapshot(groups, &state.channels)?;
    }
    Ok(())
}

fn parse_channel_groups_snapshot(
    groups: &[Value],
    channels: &[ModelChannel],
) -> Result<ProjectLayerGroups, ControlError> {
    let channel_names = channels
        .iter()
        .map(|channel| channel.name.as_str())
        .collect::<HashSet<_>>();
    let mut parsed = ProjectLayerGroups::default();
    let mut ids = HashSet::new();
    for group in groups {
        let id = group
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid("channel group has no valid id"))?;
        if !ids.insert(id) {
            return Err(invalid(format!("channel group id {id} is duplicated")));
        }
        let name = group
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| invalid("channel group has no valid name"))?
            .to_string();
        let color = group
            .get("color_rgb")
            .and_then(Value::as_array)
            .filter(|values| values.len() == 3)
            .ok_or_else(|| invalid("channel group has an invalid color"))?;
        parsed.channel_groups.push(ProjectChannelGroup {
            id,
            name,
            expanded: group
                .get("expanded")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            color_rgb: [to_u8(&color[0])?, to_u8(&color[1])?, to_u8(&color[2])?],
        });
        if let Some(members) = group.get("members").and_then(Value::as_array) {
            for member in members {
                let name = member
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("channel group member has no name"))?;
                if !channel_names.contains(name) {
                    return Err(invalid(format!(
                        "channel group member '{name}' was not found"
                    )));
                }
                parsed.channel_members.insert(
                    name.to_string(),
                    ProjectChannelGroupMember {
                        group_id: id,
                        inherit_color: member
                            .get("inherit_color")
                            .and_then(Value::as_bool)
                            .unwrap_or(true),
                    },
                );
            }
        }
    }
    Ok(parsed)
}

fn observed_workspace_size(workspace: &ViewportWorkspace<ViewportModel>) -> [f32; 2] {
    match workspace.layout() {
        ViewportLayout::Single => workspace.active().state.logical_size,
        ViewportLayout::Horizontal => [
            workspace
                .viewports()
                .iter()
                .map(|viewport| viewport.state.logical_size[0])
                .sum(),
            workspace
                .viewports()
                .iter()
                .map(|viewport| viewport.state.logical_size[1])
                .fold(0.0_f32, f32::max),
        ],
        ViewportLayout::Vertical => [
            workspace
                .viewports()
                .iter()
                .map(|viewport| viewport.state.logical_size[0])
                .fold(0.0_f32, f32::max),
            workspace
                .viewports()
                .iter()
                .map(|viewport| viewport.state.logical_size[1])
                .sum(),
        ],
    }
}

fn fit_camera(viewport: &mut ViewportModel, world: [f32; 2]) {
    viewport.center = [world[0] * 0.5, world[1] * 0.5];
    viewport.zoom = ((viewport.logical_size[0] / world[0].max(1.0))
        .min(viewport.logical_size[1] / world[1].max(1.0))
        * 0.95)
        .clamp(0.000_01, 5000.0);
}

fn propagate_camera(
    workspace: &mut ViewportWorkspace<ViewportModel>,
    source: &ViewportId,
    state: &ViewportModel,
) {
    let ids = workspace
        .viewports()
        .iter()
        .filter(|slot| slot.id != *source)
        .map(|slot| slot.id.clone())
        .collect::<Vec<_>>();
    for id in ids {
        if let Some(slot) = workspace.get_mut(&id) {
            slot.state.center = state.center;
            slot.state.zoom = state.zoom;
        }
        let _ = workspace.bump_navigation_revision(&id);
    }
}

fn validate_viewport_set(
    workspace: &ViewportWorkspace<ViewportModel>,
    requested: &Value,
) -> Result<(), ControlError> {
    let requested = requested
        .as_array()
        .ok_or_else(|| invalid("viewports must be an array of viewport IDs"))?;
    let requested = requested
        .iter()
        .map(Value::as_str)
        .collect::<Option<HashSet<_>>>()
        .ok_or_else(|| invalid("viewports must contain only viewport ID strings"))?;
    let current = workspace
        .viewports()
        .iter()
        .map(|slot| slot.id.as_str())
        .collect::<HashSet<_>>();
    if current.len() != 2 || requested.len() != 2 || requested != current {
        return Err(invalid(
            "viewports must identify exactly the two current workspace viewports",
        ));
    }
    Ok(())
}

fn validate_viewport_order(
    workspace: &ViewportWorkspace<ViewportModel>,
    requested: &Value,
) -> Result<(), ControlError> {
    let requested = requested
        .as_array()
        .ok_or_else(|| invalid("viewports must be an array of viewport IDs"))?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("viewports must contain only viewport ID strings"))?;
    let current = workspace
        .viewports()
        .iter()
        .map(|viewport| viewport.id.as_str())
        .collect::<Vec<_>>();
    if requested != current {
        return Err(invalid(
            "viewports must match the current workspace order; use viewer.workspace.swap to reorder",
        ));
    }
    Ok(())
}

fn resolve_channels(
    channels: &[ModelChannel],
    selectors: &[Value],
) -> Result<HashSet<usize>, ControlError> {
    selectors
        .iter()
        .map(|selector| resolve_channel(channels, selector))
        .collect()
}

fn resolve_channel_list_ordered(
    channels: &[ModelChannel],
    selectors: &[Value],
) -> Result<Vec<usize>, ControlError> {
    let mut indices = Vec::new();
    let mut unresolved = Vec::new();
    for selector in selectors {
        match resolve_channel(channels, selector) {
            Ok(index) if !indices.contains(&index) => indices.push(index),
            Ok(_) => {}
            Err(error) => unresolved.push(error.message),
        }
    }
    if unresolved.is_empty() {
        Ok(indices)
    } else {
        Err(invalid(format!(
            "unresolved channel(s): {}",
            unresolved.join("; ")
        )))
    }
}

fn channel_selector_from_params(params: &Value) -> Result<&Value, ControlError> {
    params
        .get("index")
        .or_else(|| params.get("channel_index"))
        .or_else(|| params.get("name"))
        .or_else(|| params.get("channel"))
        .or_else(|| params.get("marker"))
        .ok_or_else(|| invalid("provide index, name, channel, or marker"))
}

fn resolve_channel(channels: &[ModelChannel], selector: &Value) -> Result<usize, ControlError> {
    if let Some(index) = selector.as_u64().and_then(|v| usize::try_from(v).ok()) {
        return channels
            .get(index)
            .map(|_| index)
            .ok_or_else(|| invalid(format!("channel index {index} is out of range")));
    }
    if let Some(name) = selector.as_str() {
        return channels
            .iter()
            .position(|channel| channel.name == name)
            .ok_or_else(|| invalid(format!("channel '{name}' was not found")));
    }
    Err(invalid("channel must be a name or index"))
}

fn normalize_plane_mode(mode: &str) -> Result<&'static str, ControlError> {
    match mode.to_ascii_lowercase().as_str() {
        "xy" => Ok("xy"),
        "xz" => Ok("xz"),
        "yz" => Ok("yz"),
        _ => Err(invalid("mode must be 'xy', 'xz', or 'yz'")),
    }
}

fn plane_mode_index(mode: &str) -> usize {
    match mode {
        "xz" => 1,
        "yz" => 2,
        _ => 0,
    }
}

fn has_channel_selector(params: &Value) -> bool {
    ["index", "channel_index", "name", "channel", "marker"]
        .iter()
        .any(|key| params.get(key).is_some())
}

fn optional_threshold_scope(params: &Value) -> Result<Option<ThresholdScope>, ControlError> {
    params
        .get("scope")
        .map(|value| match value.as_str() {
            Some("visible" | "visible_region") => Ok(ThresholdScope::Visible),
            Some("entire_image" | "full" | "full_image") => Ok(ThresholdScope::EntireImage),
            _ => Err(invalid("scope must be 'visible' or 'entire_image'")),
        })
        .transpose()
}

fn optional_threshold_level(
    params: &Value,
    level_count: usize,
) -> Result<Option<usize>, ControlError> {
    params
        .get("level")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|level| *level < level_count)
                .ok_or_else(|| invalid("level must be a valid non-negative pyramid index"))
        })
        .transpose()
}

fn optional_threshold_min_pixels(params: &Value) -> Result<Option<usize>, ControlError> {
    params
        .get("min_component_pixels")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| (1..=1_000_000).contains(value))
                .ok_or_else(|| invalid("min_component_pixels must be an integer from 1 to 1000000"))
        })
        .transpose()
}

fn optional_threshold_value(params: &Value) -> Result<Option<u16>, ControlError> {
    params
        .get("threshold")
        .map(|value| {
            value
                .as_u64()
                .filter(|value| *value <= u16::MAX as u64)
                .map(|value| value as u16)
                .ok_or_else(|| invalid("threshold must be an integer from 0 to 65535"))
        })
        .transpose()
}

fn threshold_extent(
    dataset: &DatasetModel,
    viewport: &ViewportModel,
    channel_index: usize,
    scope: ThresholdScope,
    level: &crate::data::ome::LevelInfo,
) -> Result<(u64, u64, u64, u64), ControlError> {
    let width = level
        .shape
        .get(dataset.descriptor.dims.x)
        .copied()
        .ok_or_else(|| invalid("threshold level has no x dimension"))?;
    let height = level
        .shape
        .get(dataset.descriptor.dims.y)
        .copied()
        .ok_or_else(|| invalid("threshold level has no y dimension"))?;
    if scope == ThresholdScope::EntireImage {
        return Ok((0, 0, width, height));
    }
    let channel = viewport
        .channels
        .get(channel_index)
        .ok_or_else(|| invalid("threshold channel is out of range"))?;
    let zoom = viewport.zoom.max(1e-6);
    let half_width = viewport.logical_size[0].max(1.0) * 0.5 / zoom;
    let half_height = viewport.logical_size[1].max(1.0) * 0.5 / zoom;
    let corners = [
        [
            viewport.center[0] - half_width,
            viewport.center[1] - half_height,
        ],
        [
            viewport.center[0] + half_width,
            viewport.center[1] - half_height,
        ],
        [
            viewport.center[0] + half_width,
            viewport.center[1] + half_height,
        ],
        [
            viewport.center[0] - half_width,
            viewport.center[1] + half_height,
        ],
    ];
    let pivot = [dataset.world_size[0] * 0.5, dataset.world_size[1] * 0.5];
    let mut minimum = [f32::INFINITY; 2];
    let mut maximum = [f32::NEG_INFINITY; 2];
    for point in corners {
        let local = inverse_channel_point(
            point,
            pivot,
            channel.offset_world,
            channel.scale,
            channel.rotation_rad,
        );
        minimum[0] = minimum[0].min(local[0]);
        minimum[1] = minimum[1].min(local[1]);
        maximum[0] = maximum[0].max(local[0]);
        maximum[1] = maximum[1].max(local[1]);
    }
    let downsample = level.downsample.max(1e-6);
    let x0 = (minimum[0].max(0.0) / downsample).floor() as u64;
    let y0 = (minimum[1].max(0.0) / downsample).floor() as u64;
    let x1 = (maximum[0].min(dataset.world_size[0]).max(0.0) / downsample).ceil() as u64;
    let y1 = (maximum[1].min(dataset.world_size[1]).max(0.0) / downsample).ceil() as u64;
    Ok((x0.min(width), y0.min(height), x1.min(width), y1.min(height)))
}

fn inverse_channel_point(
    point: [f32; 2],
    pivot: [f32; 2],
    offset: [f32; 2],
    scale: [f32; 2],
    rotation_rad: f32,
) -> [f32; 2] {
    let x = point[0] - pivot[0] - offset[0];
    let y = point[1] - pivot[1] - offset[1];
    let (sin, cos) = (-rotation_rad).sin_cos();
    let rotated_x = x * cos - y * sin;
    let rotated_y = x * sin + y * cos;
    [
        pivot[0] + rotated_x / scale[0].abs().max(1e-6),
        pivot[1] + rotated_y / scale[1].abs().max(1e-6),
    ]
}

fn estimate_pinned_level_bytes(
    descriptor: &DocumentDescriptor,
    level_index: usize,
    selected_channel_count: usize,
) -> u64 {
    let Some(level) = descriptor.levels.get(level_index) else {
        return 0;
    };
    if selected_channel_count == 0 {
        return 0;
    }
    let Some(&height) = level.shape.get(descriptor.dims.y) else {
        return 0;
    };
    let Some(&width) = level.shape.get(descriptor.dims.x) else {
        return 0;
    };
    let channel_count = if descriptor.dims.c.is_some() {
        selected_channel_count as u64
    } else {
        1
    };
    let bytes_per_sample = match level.dtype.as_str() {
        "|u1" | "|i1" => 1,
        "<u2" | ">u2" | "<i2" | ">i2" => 2,
        "<f4" | ">f4" | "<u4" | ">u4" | "<i4" | ">i4" => 4,
        _ => 2,
    };
    channel_count
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(bytes_per_sample))
        .unwrap_or(0)
}

fn mosaic_memory_scope(spec: &super::MosaicMemoryPinSpec) -> String {
    let mut item_ids = spec
        .items
        .iter()
        .map(|item| item.item_id)
        .collect::<Vec<_>>();
    item_ids.sort_unstable();
    format!(
        "mosaic:{}:level:{}",
        item_ids
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
        spec.level
    )
}

fn numeric_object_properties(resource: &ControlObjectResource) -> Vec<String> {
    let mut properties = resource
        .property_names
        .iter()
        .filter(|property| {
            property.as_str() != "id"
                && resource.features.iter().any(|feature| {
                    feature
                        .properties
                        .get(property.as_str())
                        .and_then(Value::as_f64)
                        .is_some_and(f64::is_finite)
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    properties.sort();
    properties.dedup();
    properties
}

fn map_level0_axis_index(
    level0: &crate::data::ome::LevelInfo,
    level: &crate::data::ome::LevelInfo,
    dimension: usize,
    index_level0: u64,
) -> Option<u64> {
    let level0_len = *level0.shape.get(dimension)?;
    let level_len = *level.shape.get(dimension)?;
    if level0_len == 0 || level_len == 0 {
        return None;
    }
    let level0_scale = level0
        .scale
        .get(dimension)
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let level_scale = level
        .scale
        .get(dimension)
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(level0_scale.max(1.0));
    let level0_translation = level0
        .translation
        .get(dimension)
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    let level_translation = level
        .translation
        .get(dimension)
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    let index_level0 = index_level0.min(level0_len.saturating_sub(1));
    let center_world = level0_translation + (index_level0 as f32 + 0.5) * level0_scale;
    let mapped = ((center_world - level_translation) / level_scale).floor();
    let mapped = if mapped.is_finite() { mapped as i64 } else { 0 };
    Some(mapped.clamp(0, level_len.saturating_sub(1) as i64) as u64)
}

fn current_plane_slice(viewport: &ViewportModel) -> u64 {
    viewport.plane_slices[plane_mode_index(&viewport.plane_mode)]
}

fn camera_changed(before: &ViewportModel, after: &ViewportModel) -> bool {
    before.center != after.center || before.zoom != after.zoom
}

fn plane_changed(before: &ViewportModel, after: &ViewportModel) -> bool {
    before.plane_mode != after.plane_mode || before.plane_slices != after.plane_slices
}

fn presentation_changed(before: &ViewportModel, after: &ViewportModel) -> bool {
    before.channels != after.channels
        || before.active_channel != after.active_channel
        || before.channel_order != after.channel_order
        || before.channel_sort != after.channel_sort
        || before.channel_search != after.channel_search
        || before.channel_groups != after.channel_groups
        || before.objects != after.objects
        || before.secondary_objects != after.secondary_objects
        || before.segmentation_labels_visible != after.segmentation_labels_visible
        || before.segmentation_geojson_visible != after.segmentation_geojson_visible
        || before.native_layers != after.native_layers
        || before.smooth_pixels != after.smooth_pixels
        || before.show_scale_bar != after.show_scale_bar
        || before.show_hud != after.show_hud
        || before.show_tile_debug != after.show_tile_debug
}

fn clamp_current_plane_slice(viewport: &mut ViewportModel, extents: [u64; 3]) {
    let index = plane_mode_index(&viewport.plane_mode);
    viewport.plane_slices[index] =
        viewport.plane_slices[index].min(extents[index].max(1).saturating_sub(1));
}

fn set_current_plane_slice(viewport: &mut ViewportModel, slice: u64, extents: [u64; 3]) {
    let index = plane_mode_index(&viewport.plane_mode);
    viewport.plane_slices[index] = slice.min(extents[index].max(1).saturating_sub(1));
}

fn to_u8(value: &Value) -> Result<u8, ControlError> {
    value
        .as_u64()
        .and_then(|v| u8::try_from(v).ok())
        .ok_or_else(|| invalid("color values must be integers from 0 to 255"))
}

fn optional_rgb(params: &Value, name: &str) -> Result<Option<[u8; 3]>, ControlError> {
    let Some(value) = params.get(name) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| invalid(format!("{name} must contain three integers")))?;
    Ok(Some([
        to_u8(&values[0])?,
        to_u8(&values[1])?,
        to_u8(&values[2])?,
    ]))
}

fn optional_finite_pair(params: &Value, name: &str) -> Result<Option<[f32; 2]>, ControlError> {
    let Some(value) = params.get(name) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or_else(|| invalid(format!("{name} must contain exactly two numbers")))?;
    let pair = [
        values[0]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid(format!("{name} values must be finite numbers")))?
            as f32,
        values[1]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid(format!("{name} values must be finite numbers")))?
            as f32,
    ];
    Ok(Some(pair))
}

fn canonical_channel_sort(value: &str) -> Option<&'static str> {
    match value {
        "manual" | "project_order" => Some("manual"),
        "name_asc" | "alphabetical_asc" => Some("name_asc"),
        "name_desc" | "alphabetical_desc" => Some("name_desc"),
        "visible_first" | "enabled_desc" => Some("visible_first"),
        "hidden_first" | "enabled_asc" => Some("hidden_first"),
        _ => None,
    }
}

fn channel_transform_json(channel: &ModelChannel, index: usize) -> Value {
    json!({
        "index": index,
        "name": channel.name,
        "offset_world": channel.offset_world,
        "scale": channel.scale,
        "rotation_rad": channel.rotation_rad,
        "rotation_degrees": channel.rotation_rad.to_degrees(),
    })
}

fn channel_order_json(viewport: &ViewportModel) -> Value {
    Value::Array(
        viewport
            .channel_order
            .iter()
            .filter_map(|index| {
                viewport.channels.get(*index).map(|channel| {
                    json!({
                        "index": index,
                        "name": channel.name,
                        "visible": channel.visible,
                    })
                })
            })
            .collect(),
    )
}

fn channel_presentation_json(viewport: &ViewportModel) -> Value {
    json!({
        "search": viewport.channel_search,
        "sort": viewport.channel_sort,
        "order": channel_order_json(viewport),
    })
}

fn channel_groups_json(viewport: &ViewportModel) -> Value {
    Value::Array(
        viewport
            .channel_groups
            .channel_groups
            .iter()
            .map(|group| {
                let members = viewport
                    .channels
                    .iter()
                    .enumerate()
                    .filter_map(|(index, channel)| {
                        let member = viewport
                            .channel_groups
                            .channel_members
                            .get(channel.name.as_str())?;
                        (member.group_id == group.id).then(|| {
                            json!({
                                "index": index,
                                "name": channel.name,
                                "inherit_color": member.inherit_color,
                            })
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "id": group.id,
                    "name": group.name,
                    "expanded": group.expanded,
                    "color_rgb": group.color_rgb,
                    "members": members,
                })
            })
            .collect(),
    )
}

fn ensure_model_channel_group(
    groups: &mut ProjectLayerGroups,
    requested_group_id: Option<u64>,
    requested_name: Option<&str>,
    color_rgb: Option<[u8; 3]>,
) -> u64 {
    if let Some(group_id) = requested_group_id
        && let Some(group) = groups
            .channel_groups
            .iter_mut()
            .find(|group| group.id == group_id)
    {
        if let Some(name) = requested_name {
            group.name = name.to_string();
        }
        if let Some(color) = color_rgb {
            group.color_rgb = color;
        }
        return group_id;
    }
    if let Some(name) = requested_name
        && let Some(group) = groups
            .channel_groups
            .iter_mut()
            .find(|group| group.name == name)
    {
        if let Some(color) = color_rgb {
            group.color_rgb = color;
        }
        return group.id;
    }
    let group_id = requested_group_id
        .filter(|id| !groups.channel_groups.iter().any(|group| group.id == *id))
        .unwrap_or_else(|| {
            groups
                .channel_groups
                .iter()
                .map(|group| group.id)
                .max()
                .unwrap_or(0)
                .wrapping_add(1)
                .max(1)
        });
    groups.channel_groups.push(ProjectChannelGroup {
        id: group_id,
        name: requested_name
            .map(str::to_string)
            .unwrap_or_else(|| format!("Group {group_id}")),
        expanded: true,
        color_rgb: color_rgb.unwrap_or([255, 255, 255]),
    });
    group_id
}

fn apply_workspace_channel_transforms(
    workspace: &mut ViewportWorkspace<ViewportModel>,
    snapshot: &Value,
) {
    let Some(transforms) = snapshot.get("channel_transforms").and_then(Value::as_array) else {
        return;
    };
    for transform in transforms {
        let Some(index) = transform
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
        else {
            continue;
        };
        let offset = transform
            .get("offset_world")
            .and_then(Value::as_array)
            .filter(|values| values.len() == 2)
            .and_then(|values| Some([values[0].as_f64()? as f32, values[1].as_f64()? as f32]));
        let scale = transform
            .get("scale")
            .and_then(Value::as_array)
            .filter(|values| values.len() == 2)
            .and_then(|values| Some([values[0].as_f64()? as f32, values[1].as_f64()? as f32]));
        let rotation = transform
            .get("rotation_rad")
            .and_then(Value::as_f64)
            .map(|value| value as f32);
        for slot in workspace.viewports_mut() {
            let Some(channel) = slot.state.channels.get_mut(index) else {
                continue;
            };
            if let Some(offset) = offset.filter(|pair| pair.iter().all(|value| value.is_finite())) {
                channel.offset_world = offset;
            }
            if let Some(scale) = scale.filter(|pair| pair.iter().all(|value| value.is_finite())) {
                channel.scale = scale;
            }
            if let Some(rotation) = rotation.filter(|value| value.is_finite()) {
                channel.rotation_rad = rotation;
            }
        }
    }
}

fn apply_workspace_channel_metadata(
    workspace: &mut ViewportWorkspace<ViewportModel>,
    snapshot: &Value,
) {
    let Some(metadata) = snapshot.get("channel_metadata").and_then(Value::as_array) else {
        return;
    };
    for item in metadata {
        let Some(index) = item
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
        else {
            continue;
        };
        let Some(note) = item.get("note").and_then(Value::as_str) else {
            continue;
        };
        for slot in workspace.viewports_mut() {
            if let Some(channel) = slot.state.channels.get_mut(index) {
                channel.note = note.to_string();
            }
        }
    }
}

fn channel_json(channel: &ModelChannel, selected: bool) -> Value {
    json!({
        "index": channel.index,
        "name": channel.name,
        "visible": channel.visible,
        "selected": selected,
        "color_rgb": channel.color_rgb,
        "window": channel.window.map(|(min,max)| json!({"min":min,"max":max})),
    })
}

fn full_channel_json(channel: &ModelChannel, selected: bool) -> Value {
    let mut value = channel_json(channel, selected);
    value
        .as_object_mut()
        .expect("channel snapshot is an object")
        .insert("note".to_string(), Value::String(channel.note.clone()));
    value
}

fn channels_json(viewport: &ViewportModel) -> Vec<Value> {
    viewport
        .channels
        .iter()
        .enumerate()
        .map(|(index, channel)| channel_json(channel, index == viewport.active_channel))
        .collect()
}

fn full_channels_json(viewport: &ViewportModel) -> Vec<Value> {
    viewport
        .channels
        .iter()
        .enumerate()
        .map(|(index, channel)| full_channel_json(channel, index == viewport.active_channel))
        .collect()
}

fn visible_channels_json(viewport: &ViewportModel) -> Vec<Value> {
    viewport
        .channels
        .iter()
        .enumerate()
        .filter(|(_, channel)| channel.visible)
        .map(|(index, channel)| {
            json!({
                "index": channel.index,
                "name": channel.name,
                "selected": index == viewport.active_channel,
            })
        })
        .collect()
}

fn active_channel_json(channel: &ModelChannel) -> Value {
    json!({
        "index": channel.index,
        "name": channel.name,
        "visible": channel.visible,
        "note": channel.note,
    })
}

fn contrast_json(channel: &ModelChannel, index: usize, abs_max: f32) -> Value {
    let (min, max) = channel.window.unwrap_or((0.0, abs_max));
    json!({
        "index": index,
        "name": channel.name,
        "min": min,
        "max": max,
        "abs_max": abs_max,
    })
}

fn canonical_visibility_mode(mode: &str) -> &str {
    match mode {
        "add" => "show",
        "remove" => "hide",
        other => other,
    }
}

fn workspace_camera_json(viewport: &ViewportModel) -> Value {
    json!({"center_world_lvl0": viewport.center, "zoom_screen_per_lvl0_px": viewport.zoom, "viewport": [0.0, 0.0, viewport.logical_size[0], viewport.logical_size[1]]})
}

fn control_camera_json(viewport: &ViewportModel) -> Value {
    let half_world = [
        viewport.logical_size[0] / viewport.zoom.max(1.0e-6) * 0.5,
        viewport.logical_size[1] / viewport.zoom.max(1.0e-6) * 0.5,
    ];
    json!({
        "center_world_lvl0": viewport.center,
        "zoom_screen_per_lvl0_px": viewport.zoom,
        "viewport": {
            "screen_rect": [
                viewport.screen_origin[0],
                viewport.screen_origin[1],
                viewport.screen_origin[0] + viewport.logical_size[0],
                viewport.screen_origin[1] + viewport.logical_size[1]
            ],
            "visible_world_lvl0": [
                viewport.center[0] - half_world[0],
                viewport.center[1] - half_world[1],
                viewport.center[0] + half_world[0],
                viewport.center[1] + half_world[1],
            ],
            "geometry_source": "logical",
        },
    })
}

fn rendering_json(viewport: &ViewportModel) -> Value {
    json!({"smooth_pixels": viewport.smooth_pixels, "show_scale_bar": viewport.show_scale_bar, "show_hud": viewport.show_hud, "show_tile_debug": viewport.show_tile_debug})
}

fn set_rendering_bool(
    params: &Value,
    aliases: &[&str],
    label: &str,
    target: &mut bool,
    saw_field: &mut bool,
) -> Result<(), ControlError> {
    let Some(value) = aliases.iter().find_map(|name| params.get(*name)) else {
        return Ok(());
    };
    *saw_field = true;
    let value = value
        .as_bool()
        .ok_or_else(|| invalid(format!("{label} must be a boolean")))?;
    *target = value;
    Ok(())
}

fn control_plane_json(
    viewport: &ViewportModel,
    extents: [u64; 3],
    orthogonal_planes: bool,
) -> Value {
    let mode_index = plane_mode_index(&viewport.plane_mode);
    let supported_modes = if orthogonal_planes {
        vec!["xy", "xz", "yz"]
    } else {
        vec!["xy"]
    };
    let slice_axis = match viewport.plane_mode.as_str() {
        "xz" => "y",
        "yz" => "x",
        _ => "z",
    };
    json!({
        "mode": viewport.plane_mode,
        "slice": viewport.plane_slices[mode_index],
        "slice_axis": slice_axis,
        "extent": extents[mode_index].max(1),
        "supported_modes": supported_modes,
        "xy_only_operations_available": viewport.plane_mode == "xy",
    })
}

fn object_style_json(objects: &Value) -> Value {
    let defaults = default_object_snapshot();
    let value = |name: &str| {
        objects
            .get(name)
            .cloned()
            .or_else(|| defaults.get(name).cloned())
            .unwrap_or(Value::Null)
    };
    let color_property = value("color_property");
    let legend = objects
        .get("color_level_overrides")
        .and_then(Value::as_object)
        .map(|overrides| {
            overrides
                .iter()
                .map(|(label, style)| {
                    json!({
                        "value": label,
                        "count": 0,
                        "color_rgb": style.get("color_rgb").cloned().unwrap_or(Value::Null),
                        "visible": style.get("visible").and_then(Value::as_bool).unwrap_or(true),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "visible": value("visible"),
        "opacity": value("opacity"),
        "width_screen_px": value("width_screen_px"),
        "color_rgb": value("color_rgb"),
        "fill_cells": value("fill_cells"),
        "fill_opacity": value("fill_opacity"),
        "selected_fill_opacity": value("selected_fill_opacity"),
        "show_selection_overlay": value("show_selection_overlay"),
        "fast_rendering": value("fast_rendering"),
        "color_mode": if color_property.as_str().is_some_and(|value| !value.is_empty()) {
            "property"
        } else {
            "single"
        },
        "color_property": color_property,
        "legend": legend,
    })
}

fn apply_object_style_patch(objects: &mut Value, params: &Value) -> Result<bool, ControlError> {
    let mut next = objects.clone();
    if !next.is_object() {
        next = default_object_snapshot();
    }
    let object = next
        .as_object_mut()
        .expect("normalized object presentation is an object");
    for name in [
        "visible",
        "fill_cells",
        "show_selection_overlay",
        "fast_rendering",
    ] {
        if let Some(value) = params.get(name) {
            let value = value
                .as_bool()
                .ok_or_else(|| invalid(format!("{name} must be a boolean")))?;
            object.insert(name.to_string(), Value::Bool(value));
        }
    }
    for name in ["opacity", "fill_opacity", "selected_fill_opacity"] {
        if let Some(value) = params.get(name) {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .ok_or_else(|| invalid(format!("{name} must be between 0 and 1")))?;
            object.insert(name.to_string(), json!(value as f32));
        }
    }
    if let Some(value) = params.get("width_screen_px") {
        let value = value
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0 && *value <= 100.0)
            .ok_or_else(|| invalid("width_screen_px must be greater than 0 and at most 100"))?;
        object.insert("width_screen_px".to_string(), json!(value as f32));
    }
    if let Some(value) = params.get("color_rgb") {
        let color = value
            .as_array()
            .filter(|values| values.len() == 3)
            .ok_or_else(|| invalid("color_rgb must contain three integers from 0 to 255"))?;
        let color = [to_u8(&color[0])?, to_u8(&color[1])?, to_u8(&color[2])?];
        object.insert("color_rgb".to_string(), json!(color));
    }
    if let Some(value) = params.get("color_property") {
        let property = match value {
            Value::Null => Value::Null,
            Value::String(value) if value.trim().is_empty() => Value::Null,
            Value::String(value) => Value::String(value.trim().to_string()),
            _ => return Err(invalid("color_property must be a string or null")),
        };
        object.insert("color_property".to_string(), property);
    }
    let changed = &next != objects;
    *objects = next;
    Ok(changed)
}

fn apply_native_object_layer_presentation(
    objects: &mut Value,
    presentation: &Value,
) -> Result<bool, ControlError> {
    let mut changed = apply_object_style_patch(objects, presentation)?;
    let Some(display) = presentation.get("display") else {
        return Ok(changed);
    };
    let display = display
        .as_object()
        .ok_or_else(|| invalid("native object display must be an object"))?;
    let mut style = serde_json::Map::new();
    style.insert(
        "color_property".to_string(),
        display
            .get("color_property_key")
            .cloned()
            .unwrap_or(Value::Null),
    );
    for name in [
        "fill_cells",
        "fill_opacity",
        "selected_fill_opacity",
        "fast_rendering",
    ] {
        if let Some(value) = display.get(name) {
            style.insert(name.to_string(), value.clone());
        }
    }
    changed |= apply_object_style_patch(objects, &Value::Object(style))?;

    let overrides = display
        .get("color_level_overrides")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut normalized = serde_json::Map::new();
    for (label, value) in overrides {
        if label.trim().is_empty() {
            return Err(invalid("object legend labels must not be empty"));
        }
        let value = value
            .as_object()
            .ok_or_else(|| invalid("object legend entries must be objects"))?;
        let visible = value
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("object legend visible must be a boolean"))?;
        let color = match value.get("color_rgb") {
            None | Some(Value::Null) => Value::Null,
            Some(Value::Array(values)) if values.len() == 3 => {
                json!([to_u8(&values[0])?, to_u8(&values[1])?, to_u8(&values[2])?])
            }
            Some(_) => {
                return Err(invalid(
                    "object legend color_rgb must be null or three integers from 0 to 255",
                ));
            }
        };
        normalized.insert(label, json!({"visible":visible,"color_rgb":color}));
    }
    let object = objects
        .as_object_mut()
        .expect("object style patch normalizes the object presentation");
    let next = Value::Object(normalized);
    changed |= object.get("color_level_overrides") != Some(&next);
    object.insert("color_level_overrides".to_string(), next);
    Ok(changed)
}

fn apply_object_legend_patch(objects: &mut Value, params: &Value) -> Result<(), ControlError> {
    let property = objects
        .get("color_property")
        .and_then(Value::as_str)
        .filter(|property| !property.is_empty())
        .ok_or_else(|| invalid("Select a color_property before editing its legend."))?
        .to_string();
    let entries = params
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("entries is required"))?;
    let mut overrides = objects
        .get("color_level_overrides")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for entry in entries {
        let label = entry
            .get("value")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("each legend entry requires a non-empty value"))?;
        let mut style = overrides
            .get(label)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(|| serde_json::Map::from_iter([("visible".to_string(), json!(true))]));
        if let Some(value) = entry.get("visible") {
            style.insert(
                "visible".to_string(),
                Value::Bool(
                    value
                        .as_bool()
                        .ok_or_else(|| invalid("legend visible must be a boolean"))?,
                ),
            );
        }
        if let Some(value) = entry.get("color_rgb") {
            let color = match value {
                Value::Null => Value::Null,
                Value::Array(values) if values.len() == 3 => {
                    json!([to_u8(&values[0])?, to_u8(&values[1])?, to_u8(&values[2])?])
                }
                _ => {
                    return Err(invalid(
                        "legend color_rgb must be null or three integers from 0 to 255",
                    ));
                }
            };
            style.insert("color_rgb".to_string(), color);
        }
        overrides.insert(label.to_string(), Value::Object(style));
    }
    let object = objects
        .as_object_mut()
        .ok_or_else(|| invalid("object presentation must be an object"))?;
    object.insert(
        "color_level_overrides".to_string(),
        Value::Object(overrides),
    );
    object.insert("color_property".to_string(), Value::String(property));
    Ok(())
}

fn apply_deep_link_viewport(
    viewport: &mut ViewportModel,
    request: &DeepLinkRequest,
    object_resource: Option<&ControlObjectResource>,
    object_filter: Option<ControlObjectFilterResult>,
    abs_max: f32,
) -> Result<Vec<String>, ControlError> {
    let mut notes = Vec::new();
    let active_terms = if request.channel_alternatives.is_empty() {
        request.channel.iter().cloned().collect::<Vec<_>>()
    } else {
        request.channel_alternatives.clone()
    };
    if !active_terms.is_empty() {
        if let Some(index) = find_deep_link_channel(&viewport.channels, &active_terms) {
            viewport.active_channel = index;
            viewport.channels[index].visible = true;
        } else {
            notes.push(format!(
                "channel '{}' was not found",
                active_terms.join("' or '")
            ));
        }
    }

    let visible_groups = deep_link_channel_term_groups(
        &request.visible_channels,
        &request.visible_channel_alternatives,
    );
    let mut visible_indices = Vec::new();
    if !visible_groups.is_empty() {
        for channel in &mut viewport.channels {
            channel.visible = false;
        }
        for terms in &visible_groups {
            if let Some(index) = find_deep_link_channel(&viewport.channels, terms) {
                if !visible_indices.contains(&index) {
                    visible_indices.push(index);
                }
                viewport.channels[index].visible = true;
            } else {
                notes.push(format!(
                    "visible channel '{}' was not found",
                    terms.join("' or '")
                ));
            }
        }
        if request.group_visible_channels || request.visible_channel_group.is_some() {
            if visible_indices.is_empty() {
                notes.push("no visible channels were available to group".to_string());
            } else {
                let name = request
                    .visible_channel_group
                    .as_deref()
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .unwrap_or("Deep link channels");
                let group_id = ensure_model_channel_group(
                    &mut viewport.channel_groups,
                    None,
                    Some(name),
                    request.visible_channel_group_color,
                );
                viewport
                    .channel_groups
                    .channel_members
                    .retain(|_, member| member.group_id != group_id);
                for index in &visible_indices {
                    viewport.channel_groups.channel_members.insert(
                        viewport.channels[*index].name.clone(),
                        ProjectChannelGroupMember {
                            group_id,
                            inherit_color: true,
                        },
                    );
                }
            }
        }
        if request.channel_order == Some(DeepLinkChannelOrder::Listed)
            && !visible_indices.is_empty()
        {
            viewport.channel_order = visible_indices
                .iter()
                .copied()
                .chain(
                    viewport
                        .channel_order
                        .iter()
                        .copied()
                        .filter(|index| !visible_indices.contains(index)),
                )
                .collect();
        }
    }

    for terms in deep_link_channel_term_groups(
        &request.hidden_channels,
        &request.hidden_channel_alternatives,
    ) {
        if let Some(index) = find_deep_link_channel(&viewport.channels, &terms) {
            viewport.channels[index].visible = false;
        } else {
            notes.push(format!(
                "hidden channel '{}' was not found",
                terms.join("' or '")
            ));
        }
    }

    for requested in &request.channel_colors {
        if let Some(index) =
            find_deep_link_channel(&viewport.channels, std::slice::from_ref(&requested.channel))
        {
            viewport.channels[index].color_rgb = requested.color_rgb;
            if let Some(member) = viewport
                .channel_groups
                .channel_members
                .get_mut(&viewport.channels[index].name)
            {
                member.inherit_color = false;
            }
        } else {
            notes.push(format!(
                "channel colour target '{}' was not found",
                requested.channel
            ));
        }
    }

    if request.contrast_min.is_some() || request.contrast_max.is_some() {
        let index = viewport
            .active_channel
            .min(viewport.channels.len().saturating_sub(1));
        if let Some(channel) = viewport.channels.get_mut(index) {
            let (old_min, old_max) = channel.window.unwrap_or((0.0, abs_max));
            let min = request.contrast_min.unwrap_or(old_min).clamp(0.0, abs_max);
            let max = request.contrast_max.unwrap_or(old_max).clamp(0.0, abs_max);
            if min.is_finite() && max.is_finite() && max > min {
                channel.window = Some((min, max));
                channel.contrast_manual = true;
            } else {
                notes.push(format!(
                    "contrast limits for channel '{}' were invalid",
                    channel.name
                ));
            }
        }
    }
    for contrast in &request.channel_contrasts {
        if let Some(index) =
            find_deep_link_channel(&viewport.channels, std::slice::from_ref(&contrast.channel))
        {
            let min = contrast.min.clamp(0.0, abs_max);
            let max = contrast.max.clamp(0.0, abs_max);
            if min.is_finite() && max.is_finite() && max > min {
                viewport.channels[index].window = Some((min, max));
                viewport.channels[index].contrast_manual = true;
            } else {
                notes.push(format!(
                    "contrast limits for channel '{}' were invalid",
                    contrast.channel
                ));
            }
        } else {
            notes.push(format!(
                "contrast channel '{}' was not found",
                contrast.channel
            ));
        }
    }

    if !viewport.objects.is_object() {
        viewport.objects = default_object_snapshot();
    }
    if let Some(property) = request.cell_color_by.as_deref() {
        viewport.objects["color_property"] = Value::String(property.to_string());
        viewport.objects["fill_cells"] = Value::Bool(request.fill_cells.unwrap_or(true));
    } else if let Some(fill) = request.fill_cells {
        viewport.objects["fill_cells"] = Value::Bool(fill);
    }
    if let Some(show) = request.show_selection_overlay {
        viewport.objects["show_selection_overlay"] = Value::Bool(show);
    }
    if let Some(fast) = request.fast_object_rendering {
        viewport.objects["fast_rendering"] = Value::Bool(fast);
    }
    apply_deep_link_object_legend(viewport, request, object_resource);

    if let Some(result) = object_filter {
        viewport.objects["filter"] = result.model;
        viewport.object_filter_indices = result.matching_indices;
        viewport.object_filter_active = result.active;
        viewport.object_filter_revision = viewport.object_filter_revision.wrapping_add(1).max(1);
    } else if let Some(model) = object_filter_model(request) {
        viewport.objects["filter"] = model;
        viewport.object_filter_indices = Arc::new(Vec::new());
        viewport.object_filter_active = false;
        viewport.object_filter_revision = viewport.object_filter_revision.wrapping_add(1).max(1);
        if object_resource.is_none() {
            notes.push("object filter was retained but object data is unavailable".to_string());
        }
    }

    if let Some(center) = request.center_world {
        if !center.iter().all(|value| value.is_finite()) {
            return Err(invalid("deep-link camera center must be finite"));
        }
        viewport.center = center;
    }
    if let Some(zoom) = request.zoom {
        if !zoom.is_finite() || zoom <= 0.0 {
            return Err(invalid("deep-link camera zoom must be positive and finite"));
        }
        viewport.zoom = zoom;
    }
    Ok(notes)
}

fn find_deep_link_channel(channels: &[ModelChannel], terms: &[String]) -> Option<usize> {
    for term in terms {
        let needle = normalize_deep_link_term(term);
        if needle.is_empty() {
            continue;
        }
        if let Some(index) = channels
            .iter()
            .position(|channel| normalize_deep_link_term(&channel.name) == needle)
        {
            return Some(index);
        }
        if let Some(index) = channels.iter().position(|channel| {
            normalize_deep_link_term(marker_from_channel_label(&channel.name)) == needle
        }) {
            return Some(index);
        }
        let marker_matches = channels
            .iter()
            .enumerate()
            .filter_map(|(index, channel)| {
                deep_link_marker_alias_matches(term, marker_from_channel_label(&channel.name))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if marker_matches.len() == 1 {
            return marker_matches.first().copied();
        }
        let contains = channels
            .iter()
            .enumerate()
            .filter_map(|(index, channel)| {
                normalize_deep_link_term(&channel.name)
                    .contains(&needle)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if contains.len() == 1 {
            return contains.first().copied();
        }
    }
    None
}

fn normalize_deep_link_term(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect()
}

fn marker_from_channel_label(value: &str) -> &str {
    let marker = value
        .split_once(" - ")
        .map(|(_, marker)| marker)
        .unwrap_or(value)
        .trim();
    marker
        .split_once(" (")
        .map(|(marker, _)| marker)
        .or_else(|| marker.split_once(" [").map(|(marker, _)| marker))
        .unwrap_or(marker)
        .trim()
}

fn deep_link_marker_alias_matches(requested: &str, candidate: &str) -> bool {
    let requested = normalize_deep_link_term(requested);
    let candidate = normalize_deep_link_term(candidate);
    if requested.is_empty() || candidate.is_empty() {
        return false;
    }
    if requested == candidate {
        return true;
    }
    let Some((requested_digits, requested_suffix)) = deep_link_cd_marker_suffix(&requested) else {
        return false;
    };
    let Some((candidate_digits, candidate_suffix)) = deep_link_cd_marker_suffix(&candidate) else {
        return false;
    };
    requested_digits == candidate_digits
        && if requested_suffix.is_empty() {
            candidate_suffix
                .chars()
                .next()
                .is_none_or(|ch| ch.is_ascii_alphabetic())
        } else {
            requested_suffix == candidate_suffix
        }
}

fn deep_link_cd_marker_suffix(value: &str) -> Option<(&str, &str)> {
    let rest = value.strip_prefix("cd")?;
    let digits = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
    (digits > 0).then(|| rest.split_at(digits))
}

fn deep_link_channel_term_groups(raw: &[String], alternatives: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut groups = alternatives
        .iter()
        .filter_map(|terms| {
            let terms = terms
                .iter()
                .map(|term| term.trim())
                .filter(|term| !term.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            (!terms.is_empty()).then_some(terms)
        })
        .collect::<Vec<_>>();
    groups.extend(
        raw.iter()
            .map(|term| term.trim())
            .filter(|term| !term.is_empty())
            .map(|term| vec![term.to_string()]),
    );
    groups
}

fn apply_deep_link_object_legend(
    viewport: &mut ViewportModel,
    request: &DeepLinkRequest,
    resource: Option<&ControlObjectResource>,
) {
    let property = request
        .cell_color_by
        .as_deref()
        .or_else(|| {
            viewport
                .objects
                .get("color_property")
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|property| !property.is_empty());
    let Some(property) = property else {
        return;
    };
    let mut values = resource
        .into_iter()
        .flat_map(|resource| resource.features.iter())
        .filter_map(|feature| feature.properties.get(property))
        .filter_map(object_value_label)
        .collect::<Vec<_>>();
    values.extend(request.visible_cell_types.iter().cloned());
    values.extend(request.hidden_cell_types.iter().cloned());
    values.extend(
        request
            .object_level_colors
            .iter()
            .map(|entry| entry.value.clone()),
    );
    values.sort();
    values.dedup();
    let visible = request
        .visible_cell_types
        .iter()
        .map(|value| normalize_deep_link_term(value))
        .collect::<HashSet<_>>();
    let hidden = request
        .hidden_cell_types
        .iter()
        .map(|value| normalize_deep_link_term(value))
        .collect::<HashSet<_>>();
    let colors = request
        .object_level_colors
        .iter()
        .map(|entry| (normalize_deep_link_term(&entry.value), entry.color_rgb))
        .collect::<HashMap<_, _>>();
    let overrides = viewport
        .objects
        .as_object_mut()
        .expect("deep-link object presentation is normalized")
        .entry("color_level_overrides")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("object legend overrides are an object");
    for value in values {
        let normalized = normalize_deep_link_term(&value);
        let style = overrides
            .entry(value)
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("object legend style is an object");
        if !visible.is_empty() || !hidden.is_empty() {
            style.insert(
                "visible".to_string(),
                Value::Bool(
                    (visible.is_empty() || visible.contains(&normalized))
                        && !hidden.contains(&normalized),
                ),
            );
        }
        if let Some(color) = colors.get(&normalized) {
            style.insert("color_rgb".to_string(), json!(color));
        }
    }
}

fn object_value_label(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn project_channel_view_json(channel: &ModelChannel) -> Value {
    json!({
        "name":channel.name,
        "visible":channel.visible,
        "color_rgb":channel.color_rgb,
        "window":channel.window.map(|(min,max)| [min,max]),
        "offset_world":channel.offset_world,
        "scale":channel.scale,
        "rotation_rad":channel.rotation_rad,
        "note":channel.note,
    })
}

fn project_segmentation_view_json(dataset: &DatasetModel, viewport: &ViewportModel) -> Value {
    json!({
        "label_name":dataset.label_loaded.as_ref().unwrap_or(&dataset.label_selected),
        "object_display":{
            "color_property_key":viewport.objects.get("color_property").cloned().unwrap_or(Value::Null),
            "color_level_overrides":viewport.objects.get("color_level_overrides").cloned().unwrap_or_else(|| json!({})),
            "fill_cells":viewport.objects.get("fill_cells").cloned().unwrap_or(Value::Bool(false)),
            "fill_opacity":viewport.objects.get("fill_opacity").cloned().unwrap_or(json!(0.30_f32)),
            "selected_fill_opacity":viewport.objects.get("selected_fill_opacity").cloned().unwrap_or(json!(0.70_f32)),
            "fast_rendering":viewport.objects.get("fast_rendering").cloned().unwrap_or(Value::Bool(true)),
        },
    })
}

fn project_workspace_view_json(workspace: &ViewportWorkspace<ViewportModel>) -> Value {
    let active = workspace.active_id();
    json!({
        "version":1,
        "layout":workspace.layout().as_str(),
        "split_ratio":workspace.split_ratio(),
        "active_viewport_id":active.as_str(),
        "link_camera":workspace.links().camera,
        "link_plane":workspace.links().plane,
        "link_selection":workspace.links().selection,
        "viewports":workspace.viewports().iter().map(|slot| {
            let viewport = &slot.state;
            let plane_slice = current_plane_slice(viewport);
            let (x,y,z) = match viewport.plane_mode.as_str() {
                "yz" => (Some(plane_slice), None, None),
                "xz" => (None, Some(plane_slice), None),
                _ => (None, None, Some(plane_slice)),
            };
            json!({
                "id":slot.id.as_str(),
                "title":slot.title,
                "navigation_revision":slot.navigation_revision,
                "presentation_revision":slot.presentation_revision,
                "camera":{"center_world_lvl0":viewport.center,"zoom_screen_per_lvl0_px":viewport.zoom},
                "plane_mode":viewport.plane_mode,
                "x_level0":x,
                "y_level0":y,
                "z_level0":z,
                "channel_order":viewport.channel_order,
                "channels":viewport.channels.iter().map(project_channel_view_json).collect::<Vec<_>>(),
                "active_channel":viewport.active_channel,
                "object_filter":viewport.objects.get("filter").cloned().unwrap_or_else(default_object_filter_model),
                "object_visible":viewport.objects.get("visible").cloned().unwrap_or(Value::Bool(false)),
                "object_opacity":viewport.objects.get("opacity").cloned().unwrap_or(json!(0.75_f32)),
                "object_width_screen_px":viewport.objects.get("width_screen_px").cloned().unwrap_or(json!(1.25_f32)),
                "object_color_rgb":viewport.objects.get("color_rgb").cloned().unwrap_or(json!([255,255,255])),
                "object_show_selection_overlay":viewport.objects.get("show_selection_overlay").cloned().unwrap_or(Value::Bool(true)),
                "presentation":viewport_json(slot, slot.id == *active),
            })
        }).collect::<Vec<_>>(),
    })
}

fn default_object_snapshot() -> Value {
    json!({
        "visible": false,
        "opacity": 0.75_f32,
        "width_screen_px": 1.25_f32,
        "color_rgb": [255, 255, 255],
        "fill_cells": false,
        "fill_opacity": 0.30_f32,
        "selected_fill_opacity": 0.70_f32,
        "show_selection_overlay": true,
        "fast_rendering": true,
        "color_property": "",
        "color_level_overrides": {},
        "filter": default_object_filter_model(),
    })
}

fn spatial_object_native_presentation(objects: &Value) -> Value {
    let defaults = default_object_snapshot();
    let value = |name: &str| {
        objects
            .get(name)
            .cloned()
            .or_else(|| defaults.get(name).cloned())
            .unwrap_or(Value::Null)
    };
    json!({
        "visible": value("visible"),
        "opacity": value("opacity"),
        "width_screen_px": value("width_screen_px"),
        "color_rgb": value("color_rgb"),
        "objects": objects,
    })
}

fn default_object_filter_model() -> Value {
    json!({
        "mode": "simple",
        "logic": "all",
        "clauses": [{"enabled": true, "property": "id", "query": ""}],
    })
}

fn set_object_filter_model(objects: &mut Value, model: Value) {
    if !objects.is_object() {
        *objects = default_object_snapshot();
    }
    objects
        .as_object_mut()
        .expect("default object snapshot is an object")
        .insert("filter".to_string(), model);
}

fn object_filter_snapshot_for_target(
    state: &ViewportModel,
    target: ObjectTarget,
    total_count: usize,
) -> Result<Value, ControlError> {
    let objects = state
        .object_presentation(target)
        .ok_or_else(|| object_target_not_found(target))?;
    let (indices, filter_active, filter_revision) = state
        .object_filter_state(target)
        .ok_or_else(|| object_target_not_found(target))?;
    let model = objects
        .get("filter")
        .cloned()
        .unwrap_or_else(default_object_filter_model);
    let mode = model
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("simple");
    let logic = model.get("logic").and_then(Value::as_str).unwrap_or("all");
    let clauses = model
        .get("clauses")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| {
            default_object_filter_model()["clauses"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        });
    let query = model
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let visible_count = if filter_active {
        indices.len()
    } else {
        total_count
    };
    Ok(json!({
        "revision": filter_revision,
        "active": filter_active,
        "mode": mode,
        "logic": logic,
        "total_count": total_count,
        "visible_count": visible_count,
        "hidden_count": total_count.saturating_sub(visible_count),
        "simple": {"logic": logic, "clauses": clauses},
        "query": {
            "text": query,
            "applied": mode == "query" && filter_active,
            "error": Value::Null,
        },
    }))
}

fn default_shared_resources(dataset_source: String) -> Value {
    json!({
        "document_instances": 1,
        "dataset_source": dataset_source,
        "dataset_instances": 1,
        "cpu_tile_cache_instances": 1,
        "cpu_tile_cache_entries": 0,
        "cpu_decoded_tile_cache_instances": 1,
        "cpu_decoded_tile_cache_entries": 0,
        "cpu_decoded_tile_cache_bytes": 0,
        "cpu_decode_requests": 0,
        "cpu_source_reads": 0,
        "cpu_decoded_cache_hits": 0,
        "gpu_raw_tile_cache_instances": 0,
        "gpu_raw_tile_cache_entries": 0,
        "primary_object_geometry_instances": 1,
        "primary_object_count": 0,
    })
}

fn default_performance_snapshot() -> Value {
    json!({
        "frame_plan_last_ms": null,
        "frame_plan_ema_ms": null,
        "frame_plan_samples": 0,
    })
}

fn is_project_model_method(method: &str) -> bool {
    matches!(
        method,
        "project.rois.list"
            | "project.get"
            | "project.create"
            | "project.update_metadata"
            | "project.rois.get"
            | "project.rois.add"
            | "project.rois.update"
            | "project.rois.remove"
            | "project.rois.reorder"
            | "project.rois.get_selection"
            | "project.rois.select"
            | "project.rois.focus"
            | "project.rois.next"
            | "project.rois.previous"
            | "project.views.list"
            | "project.views.get"
            | "project.views.create"
            | "project.views.rename"
            | "project.views.delete"
    )
}

fn viewport_response(
    workspace: &ViewportWorkspace<ViewportModel>,
    viewport_id: &ViewportId,
    result: Value,
    affected: Vec<ViewportId>,
    active_viewport_changed: bool,
) -> Value {
    let viewport = workspace
        .get(viewport_id)
        .expect("viewport response target remains in the workspace");
    let affected_viewport_ids = affected.iter().map(ViewportId::as_str).collect::<Vec<_>>();
    let link_transaction_id = (affected_viewport_ids.len() > 1)
        .then(|| format!("{}-{}", viewport_id.as_str(), viewport.navigation_revision));
    json!({
        "viewport_id": viewport_id.as_str(),
        "navigation_revision": viewport.navigation_revision,
        "presentation_revision": viewport.presentation_revision,
        "affected_viewport_ids": affected_viewport_ids,
        "link_transaction_id": link_transaction_id,
        "active_viewport_id": workspace.active_id().as_str(),
        "active_viewport_changed": active_viewport_changed,
        "result": result,
    })
}

fn viewport_json(slot: &crate::viewports::ViewportSlot<ViewportModel>, active: bool) -> Value {
    json!({
        "viewport_id": slot.id.as_str(), "title": slot.title, "active": active,
        "navigation_revision": slot.navigation_revision, "presentation_revision": slot.presentation_revision,
        "camera": workspace_camera_json(&slot.state),
        "plane": {"mode": slot.state.plane_mode, "slice": current_plane_slice(&slot.state)},
        "channels": channels_json(&slot.state),
        "channel_order": slot.state.channel_order,
        "channel_sort": slot.state.channel_sort,
        "channel_groups": channel_groups_json(&slot.state),
        "objects": slot.state.objects,
        "object_overlay_visibility": {
            "segmentation_labels":slot.state.segmentation_labels_visible,
            "segmentation_geojson":slot.state.segmentation_geojson_visible,
        },
        "native_layers": AppModel::effective_native_layers(&slot.state),
        "rendering": rendering_json(&slot.state),
    })
}

fn links_json(links: ViewportLinks) -> Value {
    json!({"camera": links.camera, "plane": links.plane, "selection": links.selection})
}

fn bounded_offset(params: &Value, name: &str) -> Result<usize, ControlError> {
    params
        .get(name)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid(format!("{name} must be a non-negative integer")))
        })
        .transpose()
        .map(|value| value.unwrap_or(0))
}

fn bounded_limit(params: &Value, default: usize) -> Result<usize, ControlError> {
    let limit = bounded_offset(params, "limit")?.max(1);
    Ok(if params.get("limit").is_some() {
        limit.min(10_000)
    } else {
        default
    })
}

fn required_nonempty_string<'a>(
    params: &'a Value,
    names: &[&str],
    label: &str,
) -> Result<&'a str, ControlError> {
    names
        .iter()
        .find_map(|name| params.get(*name).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid(format!("{label} is required")))
}

fn object_property_type(values: &[Value]) -> &'static str {
    if values.is_empty() {
        "unknown"
    } else if values.iter().all(Value::is_boolean) {
        "boolean"
    } else if values.iter().all(|value| value.as_i64().is_some()) {
        "integer"
    } else if values.iter().all(Value::is_number) {
        "number"
    } else if values.iter().all(Value::is_string) {
        "string"
    } else {
        "json"
    }
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

fn settings_snapshot_for(
    settings: &AppSettings,
    path: Option<&std::path::Path>,
    status: impl Into<String>,
    generation: u64,
    persisting: bool,
) -> Value {
    json!({
        "auto_contrast":settings.auto_contrast,
        "fast_object_rendering":settings.fast_object_rendering,
        "settings_path":path.map(|path| path.to_string_lossy().into_owned()),
        "status":status.into(),
        "generation":generation,
        "persisting":persisting,
    })
}
fn wrong_mode(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::WrongMode, message)
}
fn not_found(id: &ViewportId) -> ControlError {
    ControlError::new(
        ControlErrorKind::ResourceNotFound,
        format!("viewport '{id}' was not found"),
    )
}

#[cfg(test)]
#[path = "app/tests.rs"]
mod tests;
