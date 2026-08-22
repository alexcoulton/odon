use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::document::DocumentDescriptor;
use crate::data::ome::{ChannelInfo, DatasetRenderKind, OmeZarrDataset};
use crate::data::project_config::{
    ProjectChannelGroup, ProjectChannelGroupMember, ProjectLayerGroups,
};
use crate::deep_link::{
    DeepLinkChannelColor, DeepLinkChannelContrast, DeepLinkChannelOrder,
    DeepLinkObjectFilterClause, DeepLinkObjectFilterLogic, DeepLinkRequest,
};
use crate::settings::AppSettings;
use crate::viewports::{ViewportId, ViewportLayout, ViewportLinks, ViewportWorkspace};

use super::layers::NativeLayersModel;
use super::project::{ProjectModel, ProjectModelSnapshot};
use super::{
    ControlLabelResource, ControlObjectFilterResult, ControlObjectResource, LabelZarrDataset,
    MaskModel, ObjectSelectionModel, OperationKind, ReadinessModel, parse_world_points,
    parse_world_rect,
};

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
    native_layers: NativeLayersModel,
}

impl ViewportModel {
    fn new(channels: &[ChannelInfo]) -> Self {
        let channel_order = channels.iter().map(|channel| channel.index).collect();
        Self {
            center: [0.0, 0.0],
            zoom: 1.0,
            logical_size: DEFAULT_LOGICAL_CANVAS,
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
            native_layers: NativeLayersModel::channels(channels),
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
    right_tab: String,
    shared_resources: Value,
    performance: Value,
    object_resource: Option<std::sync::Arc<ControlObjectResource>>,
    label_available: Vec<String>,
    label_selected: String,
    label_loaded: Option<String>,
    label_resource: Option<Arc<ControlLabelResource>>,
    label_status: String,
    label_generation: u64,
    label_pending: bool,
    label_actor_owned: bool,
    object_selection: ObjectSelectionModel,
    masks: MaskModel,
    workspace: ViewportWorkspace<ViewportModel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeometrySource {
    Bootstrap,
    Derived,
    Observed,
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
    deep_link_resolve_operation_generation: u64,
    project_operation_generation: u64,
    project_operation_pending: bool,
    object_resource_generation: u64,
    installed_object_resource_generation: u64,
    object_resource_pending: bool,
    object_filter_operation_generation: u64,
    pending_object_filters: HashMap<ViewportId, u64>,
    object_selection_filter_operation_generation: u64,
    pending_object_selection_filter: Option<u64>,
    mask_io_operation_generation: u64,
    settings: AppSettings,
    recent_project_exists: HashMap<PathBuf, bool>,
    settings_path: Option<PathBuf>,
    settings_status: String,
    settings_operation_generation: u64,
    settings_operation_pending: bool,
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
}

impl AppModel {
    pub fn project() -> Self {
        Self {
            mode: ModelMode::Project,
            dataset: None,
            project: ProjectModel::default(),
            project_initialized: false,
            readiness: ReadinessModel::default(),
            projection_revision: 0,
            presented_projection_revision: 0,
            document_generation: 0,
            dataset_inspection_operation_generation: 0,
            deep_link_resolve_operation_generation: 0,
            project_operation_generation: 0,
            project_operation_pending: false,
            object_resource_generation: 0,
            installed_object_resource_generation: 0,
            object_resource_pending: false,
            object_filter_operation_generation: 0,
            pending_object_filters: HashMap::new(),
            object_selection_filter_operation_generation: 0,
            pending_object_selection_filter: None,
            mask_io_operation_generation: 0,
            settings: AppSettings::default(),
            recent_project_exists: HashMap::new(),
            settings_path: None,
            settings_status: String::new(),
            settings_operation_generation: 0,
            settings_operation_pending: false,
            measured_viewports: HashSet::new(),
            renderer_gpu_available: false,
        }
    }

    pub fn mode(&self) -> ModelMode {
        self.mode
    }

    pub fn project_snapshot(&self) -> ProjectModelSnapshot {
        self.project.snapshot()
    }

    pub fn bootstrap_settings(
        &mut self,
        settings: AppSettings,
        path: Option<PathBuf>,
        recent_project_exists: Vec<(PathBuf, bool)>,
    ) {
        if self.settings_operation_pending {
            return;
        }
        self.settings = settings.normalized();
        self.recent_project_exists = recent_project_exists.into_iter().collect();
        self.settings_path = path;
        self.settings_status.clear();
    }

    pub fn settings(&self) -> &AppSettings {
        &self.settings
    }

    pub fn settings_snapshot(&self) -> Value {
        json!({
            "auto_contrast":self.settings.auto_contrast,
            "fast_object_rendering":self.settings.fast_object_rendering,
            "settings_path":self.settings_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "status":self.settings_status,
            "generation":self.settings_operation_generation,
            "persisting":self.settings_operation_pending,
        })
    }

    pub fn recent_projects_snapshot(&self) -> Value {
        json!({
            "projects":self.settings.recent_projects.iter().map(|project| json!({
                "path":project.path.to_string_lossy(),
                "display_name":project.display_name(),
                "last_opened_unix_ms":project.last_opened_unix_ms,
                "exists":self.recent_project_exists.get(&project.path).copied().unwrap_or(false),
            })).collect::<Vec<_>>(),
        })
    }

    pub fn lifecycle_state(&self) -> Value {
        let project = self.project_snapshot();
        let mask_dirty = self
            .dataset
            .as_ref()
            .is_some_and(|dataset| dataset.masks.dirty());
        json!({
            "dirty":project.dirty || mask_dirty,
            "project_path":project.saved_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "can_save":project.saved_path.is_some(),
            "mode":self.mode.as_str(),
        })
    }

    pub fn deep_link_request_from_params(params: &Value) -> Result<DeepLinkRequest, ControlError> {
        if let Some(url) = params.get("url").and_then(Value::as_str) {
            return match DeepLinkRequest::parse_arg(url) {
                Ok(Some(request)) => Ok(request),
                Ok(None) => Err(invalid("url must use the odon: scheme")),
                Err(error) => Err(invalid(format!("invalid deep link: {error}"))),
            };
        }
        if let Some(value) = params.get("request") {
            return serde_json::from_value::<DeepLinkRequest>(value.clone())
                .map_err(|error| invalid(format!("invalid deep-link request: {error}")));
        }
        Err(invalid("url or request is required"))
    }

    fn parse_deep_link(params: &Value) -> Result<Value, ControlError> {
        let url = params
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("url is required"))?;
        let request = match DeepLinkRequest::parse_arg(url) {
            Ok(Some(request)) => request,
            Ok(None) => return Err(invalid("url must use the odon: scheme")),
            Err(error) => return Err(invalid(format!("invalid deep link: {error}"))),
        };
        Ok(json!({
            "valid":true,
            "url":request.to_url(),
            "request":request,
        }))
    }

    fn deep_link_filters(params: &Value) -> Result<Value, ControlError> {
        let request = Self::deep_link_request_from_params(params)?;
        Ok(json!({
            "object_filters":request.object_filters,
            "object_filter_logic":request.object_filter_logic,
            "object_query":request.object_query,
            "visible_cell_types":request.visible_cell_types,
            "hidden_cell_types":request.hidden_cell_types,
        }))
    }

    fn generate_deep_link(&self, params: &Value) -> Result<Value, ControlError> {
        let explicit = params.get("request").is_some();
        let mut request = if let Some(value) = params.get("request") {
            serde_json::from_value::<DeepLinkRequest>(value.clone())
                .map_err(|error| invalid(format!("invalid deep-link request: {error}")))?
        } else {
            self.current_deep_link_request()?
        };
        if !explicit
            && params
                .get("include_project")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        {
            request.project_path = self.project_snapshot().saved_path;
        }
        if params.get("roi").is_some() {
            request.roi = params
                .get("roi")
                .and_then(Value::as_str)
                .map(str::to_string);
        } else if !explicit {
            let project = self.project_snapshot();
            request.roi = project.focused_source_key.as_deref().and_then(|focused| {
                project
                    .rois
                    .iter()
                    .find(|roi| roi.source_key().as_deref() == Some(focused))
                    .map(|roi| roi.id.clone())
            });
        }
        Ok(json!({
            "url":request.to_url(),
            "request":request,
            "source":if explicit { "request" } else { "current_state" },
        }))
    }

    fn current_deep_link_request(&self) -> Result<DeepLinkRequest, ControlError> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Ok(DeepLinkRequest::default());
        };
        let viewport = &dataset.workspace.active().state;
        let active_channel = viewport
            .channels
            .get(viewport.active_channel)
            .map(|channel| channel.name.clone());
        let visible_channels = viewport
            .channel_order
            .iter()
            .filter_map(|index| viewport.channels.get(*index))
            .filter(|channel| channel.visible)
            .map(|channel| channel.name.clone())
            .collect::<Vec<_>>();
        let channel_contrasts = viewport
            .channels
            .iter()
            .filter_map(|channel| {
                channel.window.map(|(min, max)| DeepLinkChannelContrast {
                    channel: channel.name.clone(),
                    min,
                    max,
                })
            })
            .collect();
        let channel_colors = viewport
            .channels
            .iter()
            .map(|channel| DeepLinkChannelColor {
                channel: channel.name.clone(),
                color_rgb: channel.color_rgb,
            })
            .collect();
        let filter = viewport
            .objects
            .get("filter")
            .cloned()
            .unwrap_or_else(default_object_filter_model);
        let object_query = filter
            .get("query")
            .and_then(Value::as_str)
            .map(str::to_string);
        let object_filters = filter
            .get("clauses")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|clause| clause.get("enabled").and_then(Value::as_bool) != Some(false))
            .filter_map(|clause| {
                Some(DeepLinkObjectFilterClause {
                    property_key: clause.get("property")?.as_str()?.to_string(),
                    query: clause.get("query")?.as_str()?.to_string(),
                })
            })
            .filter(|clause| !clause.property_key.is_empty() && !clause.query.is_empty())
            .collect();
        Ok(DeepLinkRequest {
            channel: active_channel,
            visible_channels,
            channel_order: Some(DeepLinkChannelOrder::Listed),
            channel_contrasts,
            channel_colors,
            cell_color_by: viewport
                .objects
                .get("color_property")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            fill_cells: viewport.objects.get("fill_cells").and_then(Value::as_bool),
            show_selection_overlay: viewport
                .objects
                .get("show_selection_overlay")
                .and_then(Value::as_bool),
            fast_object_rendering: viewport
                .objects
                .get("fast_rendering")
                .and_then(Value::as_bool),
            object_filters,
            object_filter_logic: match filter.get("logic").and_then(Value::as_str) {
                Some("any") => Some(DeepLinkObjectFilterLogic::Any),
                Some("all") => Some(DeepLinkObjectFilterLogic::All),
                _ => None,
            },
            object_query,
            center_world: Some(viewport.center),
            zoom: Some(viewport.zoom),
            ..DeepLinkRequest::default()
        })
    }

    pub fn prepare_lifecycle_project_save(&mut self) -> Result<(Value, u64), ControlError> {
        if self
            .dataset
            .as_ref()
            .is_some_and(|dataset| dataset.masks.dirty())
        {
            self.sync_masks_to_project()?;
        }
        self.project_persistence_payload()
    }

    pub fn prepare_settings_set(
        &mut self,
        params: &Value,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let candidate = self.settings.patched(params).map_err(invalid)?;
        if candidate == self.settings {
            return Ok(SettingsMutationOutcome::Immediate(self.settings_snapshot()));
        }
        let path = self.settings_save_path()?;
        let response = settings_snapshot_for(
            &candidate,
            Some(&path),
            format!("Saved settings to {}.", path.display()),
            self.settings_operation_generation.wrapping_add(1).max(1),
            false,
        );
        Ok(SettingsMutationOutcome::Persist(
            self.begin_settings_save(candidate, path, response)?,
        ))
    }

    pub fn prepare_recent_project_forget(
        &mut self,
        path: PathBuf,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let mut candidate = self.settings.clone();
        let forgotten = candidate.forget_recent_project(&path);
        let response = json!({
            "forgotten":forgotten,
            "path":path.to_string_lossy(),
            "remaining":candidate.recent_projects.len(),
        });
        if !forgotten {
            return Ok(SettingsMutationOutcome::Immediate(response));
        }
        let save_path = self.settings_save_path()?;
        let operation = self.begin_settings_save(candidate, save_path, response)?;
        self.recent_project_exists
            .retain(|candidate, _| candidate != &path);
        Ok(SettingsMutationOutcome::Persist(operation))
    }

    pub fn prepare_recent_project_record(
        &mut self,
        path: PathBuf,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let mut candidate = self.settings.clone();
        if !candidate.record_recent_project(&path) {
            return Ok(SettingsMutationOutcome::Immediate(json!({
                "recorded":false,
                "path":path.to_string_lossy(),
            })));
        }
        let recorded_path = candidate
            .recent_projects
            .first()
            .map(|item| item.path.clone());
        let Some(save_path) = self.settings_path.clone() else {
            if let Some(recorded_path) = recorded_path {
                self.recent_project_exists.insert(recorded_path, true);
            }
            self.settings = candidate;
            return Ok(SettingsMutationOutcome::Immediate(json!({
                "recorded":true,
                "path":path.to_string_lossy(),
                "persisted":false,
            })));
        };
        let operation = self.begin_settings_save(
            candidate.clone(),
            save_path,
            json!({"recorded":true,"path":path.to_string_lossy(),"persisted":true}),
        )?;
        // The recent-project entry belongs to the successful project transaction. Persisting it
        // remains asynchronous, but actor queries immediately observe the canonical new list.
        self.settings = candidate;
        if let Some(recorded_path) = recorded_path {
            self.recent_project_exists.insert(recorded_path, true);
        }
        Ok(SettingsMutationOutcome::Persist(operation))
    }

    pub fn prepare_recent_projects_clear(
        &mut self,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let mut candidate = self.settings.clone();
        let cleared = candidate.recent_projects.len();
        if !candidate.clear_recent_projects() {
            return Ok(SettingsMutationOutcome::Immediate(json!({"cleared":0})));
        }
        let path = self.settings_save_path()?;
        let operation = self.begin_settings_save(candidate, path, json!({"cleared":cleared}))?;
        self.recent_project_exists.clear();
        Ok(SettingsMutationOutcome::Persist(operation))
    }

    pub fn install_settings_for_generation(
        &mut self,
        generation: u64,
        settings: AppSettings,
        response: Value,
    ) -> Option<Value> {
        if !self.settings_operation_pending || generation != self.settings_operation_generation {
            return None;
        }
        self.settings = settings;
        self.recent_project_exists.retain(|path, _| {
            self.settings
                .recent_projects
                .iter()
                .any(|project| &project.path == path)
        });
        self.settings_operation_pending = false;
        self.settings_status = self
            .settings_path
            .as_ref()
            .map(|path| format!("Saved settings to {}.", path.display()))
            .unwrap_or_else(|| "Saved settings.".to_string());
        self.readiness
            .finish(OperationKind::SettingsIo, generation, "Settings saved");
        Some(response)
    }

    pub fn fail_settings_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if !self.settings_operation_pending || generation != self.settings_operation_generation {
            return false;
        }
        self.settings_operation_pending = false;
        self.settings_status = message.into();
        self.readiness.fail(
            OperationKind::SettingsIo,
            generation,
            self.settings_status.clone(),
        );
        true
    }

    fn settings_save_path(&self) -> Result<PathBuf, ControlError> {
        self.settings_path.clone().ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "application settings path has not been bootstrapped",
            )
        })
    }

    fn begin_settings_save(
        &mut self,
        settings: AppSettings,
        path: PathBuf,
        response: Value,
    ) -> Result<SettingsSaveOperation, ControlError> {
        if self.settings_operation_pending {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "another settings persistence operation is already active",
            )
            .with_data(json!({"loading":self.loading_state()["loading"]})));
        }
        self.settings_operation_generation =
            self.settings_operation_generation.wrapping_add(1).max(1);
        self.settings_operation_pending = true;
        self.settings_status = format!("Saving settings to {}...", path.display());
        self.readiness.begin(
            OperationKind::SettingsIo,
            self.settings_operation_generation,
            self.settings_status.clone(),
        );
        Ok(SettingsSaveOperation {
            generation: self.settings_operation_generation,
            path,
            settings,
            response,
        })
    }

    pub fn bootstrap_project_from_renderer(&mut self, snapshot: ProjectModelSnapshot) -> bool {
        if self.project_initialized {
            return false;
        }
        self.project.replace(snapshot);
        self.project_initialized = true;
        true
    }

    pub fn begin_project_operation(&mut self, description: impl Into<String>) -> u64 {
        self.project_operation_generation =
            self.project_operation_generation.wrapping_add(1).max(1);
        self.project_operation_pending = true;
        self.readiness.begin(
            OperationKind::ProjectIo,
            self.project_operation_generation,
            description,
        );
        self.project_operation_generation
    }

    pub fn project_operation_is_current(&self, generation: u64) -> bool {
        self.project_operation_pending
            && generation == self.project_operation_generation
            && self
                .readiness
                .is_pending(OperationKind::ProjectIo, generation)
    }

    pub fn finish_project_operation_for_generation(&mut self, generation: u64) -> bool {
        if !self.project_operation_is_current(generation) {
            return false;
        }
        self.project_operation_pending = false;
        self.readiness
            .finish(OperationKind::ProjectIo, generation, "Ready");
        true
    }

    pub fn replace_project_rois_from_samplesheet_for_generation(
        &mut self,
        generation: u64,
        rois: Vec<crate::data::project_config::ProjectRoi>,
    ) -> Result<Option<Value>, ControlError> {
        if !self.project_operation_is_current(generation) {
            return Ok(None);
        }
        let response = self.project.replace_rois_from_samplesheet(rois)?;
        self.project_initialized = true;
        self.finish_project_operation_for_generation(generation);
        Ok(Some(response))
    }

    pub fn add_discovered_project_roots_for_generation(
        &mut self,
        generation: u64,
        roots: Vec<PathBuf>,
    ) -> Result<Option<(usize, Value)>, ControlError> {
        if !self.project_operation_is_current(generation) {
            return Ok(None);
        }
        let response = self.project.add_discovered_roots(roots)?;
        self.project_initialized = true;
        self.finish_project_operation_for_generation(generation);
        Ok(Some(response))
    }

    pub fn install_project_for_generation(
        &mut self,
        generation: u64,
        path: PathBuf,
        config: crate::data::project_config::ProjectConfig,
        state: Value,
    ) -> Result<bool, ControlError> {
        if !self.project_operation_is_current(generation) {
            return Ok(false);
        }
        self.project.install_loaded(path, config, state)?;
        self.project_initialized = true;
        self.project_operation_pending = false;
        self.readiness
            .finish(OperationKind::ProjectIo, generation, "Ready");
        Ok(true)
    }

    pub fn finish_project_save_for_generation(
        &mut self,
        generation: u64,
        path: PathBuf,
        saved_config_generation: u64,
    ) -> bool {
        if !self.project_operation_is_current(generation) {
            return false;
        }
        self.project.mark_saved(path, saved_config_generation);
        self.project_operation_pending = false;
        self.readiness
            .finish(OperationKind::ProjectIo, generation, "Ready");
        true
    }

    pub fn fail_project_operation(&mut self, generation: u64, message: impl Into<String>) -> bool {
        if !self.project_operation_is_current(generation) {
            return false;
        }
        self.project_operation_pending = false;
        self.readiness
            .fail(OperationKind::ProjectIo, generation, message);
        true
    }

    pub fn project_persistence_payload(&self) -> Result<(Value, u64), ControlError> {
        self.project.persistence_payload()
    }

    pub fn update_project_manifest(&mut self, resources: Vec<Value>, layers: Vec<Value>) -> bool {
        self.project.update_manifest(resources, layers)
    }

    pub fn channel_intensity_spec(
        &self,
        dataset: &OmeZarrDataset,
        params: &Value,
    ) -> Result<ChannelIntensitySpec, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        let channel_index = if params.as_object().is_some_and(|object| !object.is_empty())
            && channel_selector_from_params(params).is_ok()
        {
            resolve_channel(&viewport.channels, channel_selector_from_params(params)?)?
        } else {
            viewport.active_channel
        };
        let channel = viewport
            .channels
            .get(channel_index)
            .ok_or_else(|| invalid(format!("channel index {channel_index} is out of range")))?;
        let level0 = dataset
            .levels
            .first()
            .ok_or_else(|| invalid("dataset has no pyramid levels"))?;
        let requested_level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|level| usize::try_from(level).ok());
        let level_index = requested_level
            .unwrap_or_else(|| dataset.levels.len().saturating_sub(1))
            .min(dataset.levels.len().saturating_sub(1));
        let level = dataset
            .levels
            .get(level_index)
            .ok_or_else(|| invalid(format!("level {level_index} is out of range")))?;
        let (vertical, horizontal, slice_dimension) = match viewport.plane_mode.as_str() {
            "xy" => (dataset.dims.y, dataset.dims.x, dataset.dims.z),
            "xz" => (
                dataset
                    .dims
                    .z
                    .ok_or_else(|| invalid("current view plane has no display axes"))?,
                dataset.dims.x,
                Some(dataset.dims.y),
            ),
            "yz" => (
                dataset
                    .dims
                    .z
                    .ok_or_else(|| invalid("current view plane has no display axes"))?,
                dataset.dims.y,
                Some(dataset.dims.x),
            ),
            _ => return Err(invalid("current view plane has no display axes")),
        };
        if vertical >= level.shape.len() || horizontal >= level.shape.len() {
            return Err(invalid("display axes are outside image shape"));
        }
        let slice_level0 = current_plane_slice(viewport);
        let mapped_slice = slice_dimension
            .and_then(|dimension| map_level0_axis_index(level0, level, dimension, slice_level0));
        let mut ranges = Vec::with_capacity(level.shape.len());
        for dimension in 0..level.shape.len() {
            let length = level.shape[dimension];
            if Some(dimension) == dataset.dims.c {
                let selected = (channel.index as u64).min(length.saturating_sub(1));
                ranges.push(selected..selected.saturating_add(1));
            } else if Some(dimension) == slice_dimension {
                let selected = mapped_slice.unwrap_or(0).min(length.saturating_sub(1));
                ranges.push(selected..selected.saturating_add(1));
            } else if dimension == vertical || dimension == horizontal {
                ranges.push(0..length);
            } else {
                ranges.push(0..length.min(1));
            }
        }
        Ok(ChannelIntensitySpec {
            channel_index,
            channel_name: channel.name.clone(),
            level_number: level.index,
            downsample: level.downsample,
            zarr_path: format!("/{}", level.path.trim_start_matches('/')),
            dtype: level.dtype.clone(),
            ranges,
        })
    }

    pub fn document_generation(&self) -> u64 {
        self.document_generation
    }

    pub fn begin_object_resource_load(&mut self, source: impl Into<String>) -> (u64, u64) {
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        self.object_resource_pending = true;
        self.readiness.begin(
            OperationKind::Objects,
            self.object_resource_generation,
            format!("Loading objects: {}", source.into()),
        );
        (self.document_generation, self.object_resource_generation)
    }

    pub fn current_object_resource_request(&self) -> Option<(PathBuf, f32)> {
        let resource = self.dataset.as_ref()?.object_resource.as_ref()?;
        Some((resource.source.clone(), resource.downsample_factor))
    }

    pub fn install_object_resource_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        resource: std::sync::Arc<ControlObjectResource>,
    ) -> bool {
        if self.mode != ModelMode::Single
            || document_generation != self.document_generation
            || resource_generation != self.object_resource_generation
            || !self.object_resource_pending
        {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        dataset.object_resource = Some(resource);
        dataset.object_selection.reset();
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.native_layers.set_primary_objects(true);
            set_object_filter_model(&mut viewport.state.objects, default_object_filter_model());
            viewport.state.object_filter_indices = Arc::new(Vec::new());
            viewport.state.object_filter_active = false;
            viewport.state.object_filter_revision =
                viewport.state.object_filter_revision.wrapping_add(1).max(1);
        }
        self.pending_object_filters.clear();
        self.pending_object_selection_filter = None;
        self.readiness.cancel_kind_pending(
            OperationKind::ObjectFilter,
            "Object filters superseded by object resource replacement",
        );
        self.installed_object_resource_generation = resource_generation;
        self.object_resource_pending = false;
        self.readiness
            .finish(OperationKind::Objects, resource_generation, "Ready");
        true
    }

    pub fn fail_object_resource_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if document_generation != self.document_generation
            || resource_generation != self.object_resource_generation
            || !self.object_resource_pending
        {
            return false;
        }
        self.object_resource_pending = false;
        self.readiness
            .fail(OperationKind::Objects, resource_generation, message);
        true
    }

    pub fn clear_object_resource(&mut self) -> Result<Value, ControlError> {
        let dataset = self.dataset_mut()?;
        let previous = dataset.object_resource.take();
        dataset.object_selection.reset();
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.native_layers.set_primary_objects(false);
        }
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        self.installed_object_resource_generation = self.object_resource_generation;
        self.object_resource_pending = false;
        self.pending_object_filters.clear();
        self.pending_object_selection_filter = None;
        self.readiness.cancel_kind_pending(
            OperationKind::ObjectFilter,
            "Object filters superseded by clearing the object resource",
        );
        self.readiness.mark_ready(
            OperationKind::Objects,
            self.object_resource_generation,
            "Ready",
        );
        Ok(json!({
            "cleared": previous.is_some(),
            "previous_path": previous.as_ref().map(|resource| resource.source.to_string_lossy()),
            "previous_count": previous.as_ref().map_or(0, |resource| resource.features.len()),
        }))
    }

    pub fn cancel_object_resource_load(&mut self) -> Value {
        let cancelled = self.object_resource_pending;
        if cancelled {
            let cancelled_generation = self.object_resource_generation;
            self.readiness.cancel(
                OperationKind::Objects,
                cancelled_generation,
                "Object load cancelled.",
            );
            self.object_resource_generation =
                self.object_resource_generation.wrapping_add(1).max(1);
            self.object_resource_pending = false;
        }
        json!({"cancelled": cancelled, "state": self.object_resource_state()})
    }

    pub fn object_resource_state(&self) -> Value {
        let resource = self
            .dataset
            .as_ref()
            .and_then(|dataset| dataset.object_resource.as_ref());
        json!({
            "source": resource.map(|resource| resource.source.to_string_lossy()),
            "loading": self.object_resource_pending,
            "status": self.readiness.status_for(OperationKind::Objects).unwrap_or("Ready"),
            "object_count": resource.map_or(0, |resource| resource.features.len()),
            "generation": self.installed_object_resource_generation,
            "request_generation": self.object_resource_generation,
            "available_properties": resource
                .map(|resource| resource.property_names.as_ref().clone())
                .unwrap_or_default(),
        })
    }

    pub fn object_resource(&self) -> Option<std::sync::Arc<ControlObjectResource>> {
        self.dataset
            .as_ref()
            .and_then(|dataset| dataset.object_resource.clone())
    }

    pub fn label_resource(&self) -> Option<Arc<ControlLabelResource>> {
        self.dataset
            .as_ref()
            .and_then(|dataset| dataset.label_resource.clone())
    }

    pub fn labels_require_load(&self, params: &Value) -> bool {
        params.get("visible").and_then(Value::as_bool) == Some(true)
            && self
                .dataset
                .as_ref()
                .is_some_and(|dataset| dataset.label_loaded.is_none())
    }

    pub fn begin_label_load(&mut self, params: &Value) -> Result<(u64, u64, String), ControlError> {
        let document_generation = self.document_generation;
        let dataset = self.dataset_mut()?;
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(dataset.label_selected.as_str())
            .to_string();
        if name.is_empty() {
            return Err(invalid(
                "label name is required because this dataset has no default label group",
            ));
        }
        dataset.label_generation = dataset.label_generation.wrapping_add(1).max(1);
        dataset.label_pending = true;
        dataset.label_status = format!("Loading labels/{name}...");
        let status = dataset.label_status.clone();
        let label_generation = dataset.label_generation;
        self.readiness
            .begin(OperationKind::Labels, label_generation, status);
        Ok((document_generation, label_generation, name))
    }

    pub fn fail_label_load_for_generation(
        &mut self,
        document_generation: u64,
        label_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if document_generation != self.document_generation {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        if dataset.label_generation != label_generation {
            return false;
        }
        dataset.label_pending = false;
        dataset.label_status = message.into();
        self.readiness.fail(
            OperationKind::Labels,
            label_generation,
            dataset.label_status.clone(),
        );
        true
    }

    pub fn install_label_resource_for_generation(
        &mut self,
        document_generation: u64,
        label_generation: u64,
        resource: Arc<ControlLabelResource>,
    ) -> bool {
        if document_generation != self.document_generation {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        if dataset.label_generation != label_generation {
            return false;
        }
        let name = resource.dataset.label_name.clone();
        if !dataset.label_available.contains(&name) {
            dataset.label_available.push(name.clone());
            dataset.label_available.sort();
            dataset.label_available.dedup();
        }
        dataset.label_selected = name.clone();
        dataset.label_loaded = Some(name.clone());
        dataset.label_resource = Some(resource);
        dataset.label_pending = false;
        dataset.label_actor_owned = true;
        dataset.label_status = format!("Loaded labels/{name}.");
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.segmentation_labels_visible = true;
            viewport
                .state
                .native_layers
                .set_segmentation_labels(true, true);
        }
        self.readiness
            .finish(OperationKind::Labels, label_generation, "Ready");
        true
    }

    fn labels_snapshot(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        let offset = viewport
            .native_layers
            .get("segmentation_labels")
            .map(|layer| layer.offset_world)
            .unwrap_or([0.0, 0.0]);
        Ok(json!({
            "available":dataset.label_available,
            "selected":dataset.label_selected,
            "loaded":dataset.label_loaded,
            "visible":viewport.segmentation_labels_visible,
            "busy":dataset.label_pending,
            "gpu_available":self.renderer_gpu_available,
            "status":dataset.label_status,
            "offset_world":offset,
            "generation":dataset.label_generation,
            "actor_owned":dataset.label_actor_owned,
        }))
    }

    fn unload_labels(&mut self) -> Result<Value, ControlError> {
        let (unloaded, label_generation) = {
            let dataset = self.dataset_mut()?;
            let unloaded = dataset.label_loaded.take();
            dataset.label_resource = None;
            dataset.label_generation = dataset.label_generation.wrapping_add(1).max(1);
            dataset.label_pending = false;
            dataset.label_actor_owned = true;
            dataset.label_status = "Unloaded segmentation labels.".to_string();
            for viewport in dataset.workspace.viewports_mut() {
                viewport.state.segmentation_labels_visible = false;
                viewport
                    .state
                    .native_layers
                    .set_segmentation_labels(false, false);
            }
            (unloaded, dataset.label_generation)
        };
        self.readiness.mark_ready(
            OperationKind::Labels,
            label_generation,
            "Unloaded segmentation labels.",
        );
        Ok(json!({"unloaded":unloaded,"labels":self.labels_snapshot()?}))
    }

    fn set_labels_visibility(&mut self, params: &Value) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible must be a boolean"))?;
        if visible && self.dataset()?.label_loaded.is_none() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "the selected label resource must be loaded before it can be shown",
            ));
        }
        let dataset = self.dataset_mut()?;
        let viewport_id = dataset.workspace.active_id().clone();
        let viewport = &mut dataset.workspace.active_mut().state;
        let mut changed = viewport.segmentation_labels_visible != visible;
        viewport.segmentation_labels_visible = visible;
        if viewport.native_layers.get("segmentation_labels").is_some() {
            changed |= viewport
                .native_layers
                .set_visibility("segmentation_labels", visible)?;
        }
        if changed {
            let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        }
        self.labels_snapshot()
    }

    pub(crate) fn project_view_apply_requires_legacy(&mut self, params: &Value) -> bool {
        let Ok(view) = self.project.dispatch("project.views.get", params) else {
            return false;
        };
        let spec = view.get("spec").unwrap_or(&Value::Null);
        let Some(dataset) = self.dataset.as_ref() else {
            return false;
        };
        let needs_objects = spec
            .get("segmentation_source")
            .and_then(Value::as_str)
            .is_some_and(|source| !source.trim().is_empty())
            && spec.get("load_labels").and_then(Value::as_bool) != Some(true);
        let needs_labels = spec.get("load_labels").and_then(Value::as_bool) == Some(true);
        (needs_objects && dataset.object_resource.is_none())
            || (needs_labels
                && dataset
                    .workspace
                    .active()
                    .state
                    .native_layers
                    .get("segmentation_labels")
                    .is_none())
    }

    pub fn mask_generation(&self) -> Result<(u64, u64), ControlError> {
        Ok((self.document_generation, self.dataset()?.masks.generation()))
    }

    pub fn begin_mask_import_operation(&mut self) -> Result<(u64, u64, u64, String), ControlError> {
        let (document_generation, mask_generation) = self.mask_generation()?;
        let (operation_generation, scope) = self.begin_mask_io_operation("import");
        self.readiness.begin_scoped(
            OperationKind::MaskIo,
            &scope,
            operation_generation,
            "Importing mask GeoJSON",
        );
        Ok((
            document_generation,
            mask_generation,
            operation_generation,
            scope,
        ))
    }

    pub fn begin_mask_export_operation(&mut self) -> Result<(u64, String), ControlError> {
        self.dataset()?;
        let (operation_generation, scope) = self.begin_mask_io_operation("export");
        self.readiness.begin_scoped(
            OperationKind::MaskIo,
            &scope,
            operation_generation,
            "Exporting mask GeoJSON",
        );
        Ok((operation_generation, scope))
    }

    pub fn fail_mask_io_for_generation(
        &mut self,
        scope: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        self.readiness
            .fail_scoped(OperationKind::MaskIo, scope, operation_generation, message)
    }

    pub fn cancel_mask_io_for_generation(
        &mut self,
        scope: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        self.readiness
            .cancel_scoped(OperationKind::MaskIo, scope, operation_generation, message)
    }

    pub fn finish_mask_io_for_generation(
        &mut self,
        scope: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        self.readiness
            .finish_scoped(OperationKind::MaskIo, scope, operation_generation, message)
    }

    fn begin_mask_io_operation(&mut self, direction: &str) -> (u64, String) {
        self.mask_io_operation_generation =
            self.mask_io_operation_generation.wrapping_add(1).max(1);
        let generation = self.mask_io_operation_generation;
        (generation, format!("{direction}:{generation}"))
    }

    pub fn mask_export_layers(
        &self,
        layer_id: Option<u64>,
    ) -> Result<Vec<crate::data::project_config::ProjectMaskLayer>, ControlError> {
        self.dataset()?.masks.export_layers(layer_id)
    }

    pub fn install_imported_masks_for_generation(
        &mut self,
        document_generation: u64,
        mask_generation: u64,
        operation_generation: u64,
        operation_scope: &str,
        name: String,
        editable: bool,
        polygons_world: Vec<Vec<[f32; 2]>>,
        source_geojson: PathBuf,
    ) -> Option<Value> {
        if !self.readiness.is_pending_scoped(
            OperationKind::MaskIo,
            operation_scope,
            operation_generation,
        ) {
            return None;
        }
        if document_generation != self.document_generation
            || self
                .dataset
                .as_ref()
                .is_none_or(|dataset| dataset.masks.generation() != mask_generation)
        {
            self.readiness.cancel_scoped(
                OperationKind::MaskIo,
                operation_scope,
                operation_generation,
                "Mask import superseded by newer document or mask state",
            );
            return None;
        }
        let dataset = self.dataset.as_mut()?;
        let response =
            dataset
                .masks
                .install_imported_layer(name, editable, polygons_world, source_geojson);
        Self::sync_mask_native_layers(dataset);
        self.readiness.finish_scoped(
            OperationKind::MaskIo,
            operation_scope,
            operation_generation,
            "Mask import ready",
        );
        Some(response)
    }

    fn sync_mask_native_layers(dataset: &mut DatasetModel) {
        let projection = dataset.masks.projection_json();
        let active = projection.get("active_layer_id").and_then(Value::as_u64);
        let masks = dataset.masks.export_layers(None).unwrap_or_default();
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.native_layers.sync_masks(&masks, active);
        }
    }

    fn mask_persistence_state(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let local_path = dataset.descriptor.source.local_path();
        Ok(json!({
            "dirty": dataset.masks.dirty(),
            "dataset_local": local_path.is_some(),
            "dataset_path": local_path.map(|path| path.to_string_lossy().into_owned()),
            "project_path": self.project_snapshot().saved_path.map(|path| path.to_string_lossy().into_owned()),
            "live_layer_count": dataset.masks.export_layers(None)?.len(),
            "persisted_layer_count": self.project.mask_layer_count_for_source(&dataset.descriptor.source),
        }))
    }

    fn sync_masks_to_project(&mut self) -> Result<Value, ControlError> {
        let (source, layers) = {
            let dataset = self.dataset()?;
            if dataset.descriptor.source.local_path().is_none() {
                return Err(invalid("mask project persistence requires a local dataset"));
            }
            (
                dataset.descriptor.source.clone(),
                dataset.masks.export_layers(None)?,
            )
        };
        self.project.sync_mask_layers_for_source(source, layers)?;
        self.project_initialized = true;
        self.dataset_mut()?.masks.mark_persisted();
        Ok(json!({
            "synced": true,
            "persistence": self.mask_persistence_state()?,
        }))
    }

    fn required_object_resource(
        &self,
        method: &str,
    ) -> Result<&ControlObjectResource, ControlError> {
        self.dataset()?.object_resource.as_deref().ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::NotReady,
                format!("{method} requires object data to be loaded"),
            )
            .with_data(json!({
                "method": method,
                "required_readiness": ["object_resource"],
                "object_state": self.object_resource_state(),
            }))
        })
    }

    fn object_properties_list(&self, params: &Value) -> Result<Value, ControlError> {
        let method = "viewer.objects.properties.list";
        let resource = self.required_object_resource(method)?;
        let offset = bounded_offset(params, "offset")?;
        let limit = bounded_limit(params, 200)?;
        let total = resource.property_names.len();
        let columns = resource
            .property_names
            .iter()
            .skip(offset)
            .take(limit)
            .map(|name| {
                let values = resource
                    .features
                    .iter()
                    .filter_map(|feature| {
                        if name == "id" {
                            Some(Value::String(feature.id.clone()))
                        } else {
                            feature.properties.get(name).cloned()
                        }
                    })
                    .filter(|value| !value.is_null())
                    .collect::<Vec<_>>();
                let kind = object_property_type(&values);
                let categorical = values
                    .iter()
                    .map(Value::to_string)
                    .collect::<HashSet<_>>()
                    .len()
                    <= 256;
                json!({
                    "name": name,
                    "loaded": true,
                    "loading": false,
                    "type": kind,
                    "numeric": matches!(kind, "integer" | "number"),
                    "categorical": categorical,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(columns.len()) < total,
            "columns": columns,
        }))
    }

    fn object_property_load(&self, params: &Value) -> Result<Value, ControlError> {
        let method = "viewer.objects.properties.load";
        let resource = self.required_object_resource(method)?;
        let property = required_nonempty_string(params, &["property", "name"], "property")?;
        if !resource.property_names.iter().any(|name| name == property) {
            return Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("unknown object property '{property}'"),
            ));
        }
        Ok(json!({"property": property, "loaded": true, "loading": false}))
    }

    fn object_property_values(&self, params: &Value) -> Result<Value, ControlError> {
        let method = "viewer.objects.properties.values";
        let resource = self.required_object_resource(method)?;
        let property = required_nonempty_string(params, &["property", "name"], "property")?;
        if !resource.property_names.iter().any(|name| name == property) {
            return Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("unknown object property '{property}'"),
            ));
        }
        let offset = bounded_offset(params, "offset")?;
        let limit = bounded_limit(params, 200)?;
        let total = resource.features.len();
        let values = resource
            .features
            .iter()
            .enumerate()
            .skip(offset)
            .take(limit)
            .map(|(index, feature)| {
                json!({
                    "index": index,
                    "id": feature.id,
                    "value": resource.property_value(index, property).unwrap_or(Value::Null),
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "property": property,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(values.len()) < total,
            "values": values,
        }))
    }

    fn require_primary_object_target(params: &Value) -> Result<(), ControlError> {
        match params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("segmentation_objects")
        {
            "active" | "objects" | "segmentation_objects" => Ok(()),
            "spatial_shape" => Err(ControlError::new(
                ControlErrorKind::WrongMode,
                "spatial-shape object selection has not been installed in the actor resource registry",
            )),
            target => Err(invalid(format!(
                "unknown object selection target '{target}'"
            ))),
        }
    }

    fn object_selection_get(&self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let dataset = self.dataset()?;
        Ok(json!({
            "target":"segmentation_objects",
            "selection":dataset.object_selection.snapshot(dataset.object_resource.as_deref(), limit),
        }))
    }

    fn object_selection_clear(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let resource = self.object_resource();
        Ok(self
            .dataset_mut()?
            .object_selection
            .clear(resource.as_deref(), limit))
    }

    fn object_selection_select_ids(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let resource = self.required_object_resource("viewer.objects.selection.select_ids")?;
        let resource = Arc::new(resource.clone());
        self.dataset_mut()?
            .object_selection
            .select_ids(resource.as_ref(), params, limit)
    }

    fn object_selection_select_filtered(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        if params.get("filter_query").is_some() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "standalone object filter selection must be evaluated by a resource worker",
            ));
        }
        let limit = bounded_limit(params, 200)?;
        let resource = Arc::new(
            self.required_object_resource("viewer.objects.selection.select_filtered")?
                .clone(),
        );
        let dataset = self.dataset()?;
        let explicit_viewport = params.get("viewport_id").and_then(Value::as_str);
        let use_all = params
            .get("use_all_objects")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let use_active = params
            .get("use_active_viewport_filter")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let source_count = usize::from(explicit_viewport.is_some())
            + usize::from(use_all)
            + usize::from(use_active);
        if source_count > 1 {
            return Err(invalid("select_filtered accepts exactly one filter source"));
        }
        if source_count == 0 && dataset.workspace.len() > 1 {
            return Err(invalid(
                "multi-viewport filtered selection requires viewport_id, filter_query, use_all_objects=true, or use_active_viewport_filter=true",
            ));
        }
        let (indices, revision) = if use_all {
            (None, 0)
        } else {
            let slot = if let Some(id) = explicit_viewport {
                let id = ViewportId::new(id).map_err(|error| invalid(error.to_string()))?;
                dataset.workspace.get(&id).ok_or_else(|| not_found(&id))?
            } else {
                dataset.workspace.active()
            };
            (
                slot.state
                    .object_filter_active
                    .then(|| slot.state.object_filter_indices.as_ref().clone()),
                slot.state.object_filter_revision,
            )
        };
        self.dataset_mut()?.object_selection.select_filtered(
            resource.as_ref(),
            indices.as_deref(),
            revision,
            params,
            limit,
        )
    }

    pub fn begin_object_selection_filter_evaluation(
        &mut self,
        params: &Value,
    ) -> Result<
        (
            u64,
            u64,
            u64,
            u64,
            Arc<ControlObjectResource>,
            Value,
            String,
            usize,
        ),
        ControlError,
    > {
        Self::require_primary_object_target(params)?;
        let query = params
            .get("filter_query")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("filter_query must be a string"))?;
        let conflicting_source = params.get("viewport_id").is_some()
            || params
                .get("use_all_objects")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || params
                .get("use_active_viewport_filter")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        if conflicting_source {
            return Err(invalid("select_filtered accepts exactly one filter source"));
        }
        let mode = params
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("replace");
        if !matches!(mode, "replace" | "add" | "remove" | "toggle") {
            return Err(invalid(
                "selection mode must be replace, add, remove, or toggle",
            ));
        }
        let limit = bounded_limit(params, 200)?;
        let resource = self.object_resource().ok_or_else(|| {
            ControlError::new(ControlErrorKind::NotReady, "object data is not loaded")
        })?;
        self.object_selection_filter_operation_generation = self
            .object_selection_filter_operation_generation
            .wrapping_add(1)
            .max(1);
        let operation_generation = self.object_selection_filter_operation_generation;
        self.pending_object_selection_filter = Some(operation_generation);
        self.readiness.begin_scoped(
            OperationKind::ObjectFilter,
            "selection",
            operation_generation,
            "Evaluating object selection filter",
        );
        Ok((
            self.document_generation,
            self.installed_object_resource_generation,
            self.dataset()?.object_selection.generation(),
            operation_generation,
            resource,
            json!({"mode":"query","query":query}),
            mode.to_string(),
            limit,
        ))
    }

    pub fn fail_object_selection_filter_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if self.pending_object_selection_filter == Some(generation) {
            self.pending_object_selection_filter = None;
            self.readiness.fail_scoped(
                OperationKind::ObjectFilter,
                "selection",
                generation,
                message,
            );
            return true;
        }
        false
    }

    pub fn cancel_object_selection_filter_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if self.pending_object_selection_filter == Some(generation) {
            self.pending_object_selection_filter = None;
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                "selection",
                generation,
                message,
            );
            return true;
        }
        false
    }

    pub fn install_object_selection_filter_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        selection_generation: u64,
        operation_generation: u64,
        result: ControlObjectFilterResult,
        mode: &str,
        limit: usize,
    ) -> Option<Value> {
        if self.pending_object_selection_filter != Some(operation_generation) {
            return None;
        }
        if document_generation != self.document_generation
            || resource_generation != self.installed_object_resource_generation
            || self.dataset.as_ref()?.object_selection.generation() != selection_generation
        {
            self.pending_object_selection_filter = None;
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                "selection",
                operation_generation,
                "Object selection filter superseded by newer state",
            );
            return None;
        }
        self.pending_object_selection_filter = None;
        let resource = self.object_resource()?;
        let params = json!({"mode":mode});
        let response = self
            .dataset
            .as_mut()?
            .object_selection
            .select_filtered(
                resource.as_ref(),
                Some(result.matching_indices.as_ref()),
                operation_generation,
                &params,
                limit,
            )
            .ok()?;
        self.readiness.finish_scoped(
            OperationKind::ObjectFilter,
            "selection",
            operation_generation,
            "Object selection filter ready",
        );
        Some(response)
    }

    fn object_selection_query_rect(&self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let rect = parse_world_rect(params)?;
        let resource = self.required_object_resource("viewer.objects.query_rect")?;
        let visible = self.object_query_filter(params)?;
        Ok(json!({
            "target":"segmentation_objects",
            "query":self.dataset()?.object_selection.query_rect(resource, rect, visible.as_deref(), limit),
            "selection":self.dataset()?.object_selection.snapshot(Some(resource), limit),
        }))
    }

    fn object_selection_select_rect(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let rect = parse_world_rect(params)?;
        let resource = Arc::new(
            self.required_object_resource("viewer.objects.select_rect")?
                .clone(),
        );
        let visible = self.object_query_filter(params)?;
        self.dataset_mut()?.object_selection.select_rect(
            resource.as_ref(),
            rect,
            visible.as_deref(),
            params,
            limit,
        )
    }

    fn object_selection_query_lasso(&self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let points = parse_world_points(params)?;
        let resource = self.required_object_resource("viewer.objects.query_lasso")?;
        let visible = self.object_query_filter(params)?;
        Ok(self.dataset()?.object_selection.query_lasso(
            resource,
            &points,
            visible.as_deref(),
            limit,
        ))
    }

    fn object_selection_select_lasso(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let points = parse_world_points(params)?;
        let resource = Arc::new(
            self.required_object_resource("viewer.objects.select_lasso")?
                .clone(),
        );
        let visible = self.object_query_filter(params)?;
        self.dataset_mut()?.object_selection.select_lasso(
            resource.as_ref(),
            &points,
            visible.as_deref(),
            params,
            limit,
        )
    }

    fn object_selection_query_view(&self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let viewport = self.selection_viewport(params)?;
        let half_width = viewport.logical_size[0] / viewport.zoom.max(0.000_01) * 0.5;
        let half_height = viewport.logical_size[1] / viewport.zoom.max(0.000_01) * 0.5;
        let rect = [
            viewport.center[0] - half_width,
            viewport.center[1] - half_height,
            viewport.center[0] + half_width,
            viewport.center[1] + half_height,
        ];
        let mut scoped = params.as_object().cloned().unwrap_or_default();
        scoped.insert("world_rect".to_string(), json!(rect));
        self.object_selection_query_rect(&Value::Object(scoped))
    }

    fn object_selection_focus(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let resource = Arc::new(
            self.required_object_resource("viewer.objects.focus.set")?
                .clone(),
        );
        let (response, bounds) = self
            .dataset_mut()?
            .object_selection
            .focus(resource.as_ref(), params)?;
        if let Some(bounds) = bounds {
            self.fit_selection_bounds(params, bounds)?;
        }
        Ok(response)
    }

    fn object_selection_clear_focus(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        Ok(self.dataset_mut()?.object_selection.clear_focus())
    }

    fn object_selection_replace(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let resource = self.object_resource();
        self.dataset_mut()?
            .object_selection
            .replace_transaction(resource.as_deref(), params, limit)
    }

    fn selection_viewport(&self, params: &Value) -> Result<&ViewportModel, ControlError> {
        let workspace = &self.dataset()?.workspace;
        if let Some(id) = params.get("viewport_id").and_then(Value::as_str) {
            let id = ViewportId::new(id).map_err(|error| invalid(error.to_string()))?;
            return workspace
                .get(&id)
                .map(|slot| &slot.state)
                .ok_or_else(|| not_found(&id));
        }
        Ok(&workspace.active().state)
    }

    fn object_query_filter(&self, params: &Value) -> Result<Option<Vec<usize>>, ControlError> {
        let workspace = &self.dataset()?.workspace;
        let slot = if let Some(id) = params.get("viewport_id").and_then(Value::as_str) {
            let id = ViewportId::new(id).map_err(|error| invalid(error.to_string()))?;
            workspace.get(&id).ok_or_else(|| not_found(&id))?
        } else {
            workspace.active()
        };
        Ok(slot
            .state
            .object_filter_active
            .then(|| slot.state.object_filter_indices.as_ref().clone()))
    }

    fn fit_selection_bounds(
        &mut self,
        params: &Value,
        bounds: [f32; 4],
    ) -> Result<(), ControlError> {
        if !params.get("fit").and_then(Value::as_bool).unwrap_or(true) {
            return Ok(());
        }
        let id = if let Some(id) = params.get("viewport_id").and_then(Value::as_str) {
            ViewportId::new(id).map_err(|error| invalid(error.to_string()))?
        } else {
            self.dataset()?.workspace.active_id().clone()
        };
        let dataset = self.dataset_mut()?;
        let links = dataset.workspace.links();
        let target = dataset
            .workspace
            .get_mut(&id)
            .ok_or_else(|| not_found(&id))?;
        let [x0, y0, x1, y1] = bounds;
        target.state.center = [(x0 + x1) * 0.5, (y0 + y1) * 0.5];
        let width = (x1 - x0).abs().max(32.0);
        let height = (y1 - y0).abs().max(32.0);
        target.state.zoom = ((target.state.logical_size[0] / width)
            .min(target.state.logical_size[1] / height)
            * 0.84)
            .clamp(0.000_01, 5000.0);
        let state = target.state.clone();
        let _ = dataset.workspace.bump_navigation_revision(&id);
        if links.camera {
            propagate_camera(&mut dataset.workspace, &id, &state);
        }
        Ok(())
    }

    pub fn begin_object_filter_evaluation(
        &mut self,
        params: &Value,
    ) -> Result<
        (
            u64,
            u64,
            u64,
            String,
            u64,
            Arc<ControlObjectResource>,
            Value,
        ),
        ControlError,
    > {
        let scoped_params = if params.get("viewport_id").is_some() || params.get("id").is_some() {
            params.clone()
        } else {
            self.active_scoped_params(params)?
        };
        self.check_viewport_revision(&scoped_params)?;
        let viewport_id = Self::viewport_id(&scoped_params)?;
        let (presentation_revision, resource) = {
            let dataset = self.dataset()?;
            let viewport = dataset
                .workspace
                .get(&viewport_id)
                .ok_or_else(|| not_found(&viewport_id))?;
            let resource = dataset.object_resource.clone().ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::NotReady,
                    "object filtering requires object data to be loaded",
                )
                .with_data(json!({
                    "required_readiness": ["object_resource"],
                    "object_state": self.object_resource_state(),
                }))
            })?;
            (viewport.presentation_revision, resource)
        };
        self.object_filter_operation_generation = self
            .object_filter_operation_generation
            .wrapping_add(1)
            .max(1);
        let operation_generation = self.object_filter_operation_generation;
        self.pending_object_filters
            .insert(viewport_id.clone(), operation_generation);
        self.readiness.begin_scoped(
            OperationKind::ObjectFilter,
            viewport_id.as_str(),
            operation_generation,
            format!("Evaluating object filter for {}", viewport_id.as_str()),
        );
        Ok((
            self.document_generation,
            self.installed_object_resource_generation,
            operation_generation,
            viewport_id.as_str().to_string(),
            presentation_revision,
            resource,
            scoped_params,
        ))
    }

    pub fn fail_object_filter_for_generation(
        &mut self,
        viewport_id: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        let Ok(viewport_id) = ViewportId::new(viewport_id) else {
            return false;
        };
        if self.pending_object_filters.get(&viewport_id).copied() != Some(operation_generation) {
            return false;
        }
        self.pending_object_filters.remove(&viewport_id);
        self.readiness.fail_scoped(
            OperationKind::ObjectFilter,
            viewport_id.as_str(),
            operation_generation,
            message,
        );
        true
    }

    pub fn cancel_object_filter_for_generation(
        &mut self,
        viewport_id: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        let Ok(viewport_id) = ViewportId::new(viewport_id) else {
            return false;
        };
        if self.pending_object_filters.get(&viewport_id).copied() != Some(operation_generation) {
            return false;
        }
        self.pending_object_filters.remove(&viewport_id);
        self.readiness.cancel_scoped(
            OperationKind::ObjectFilter,
            viewport_id.as_str(),
            operation_generation,
            message,
        );
        true
    }

    pub fn install_object_filter_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        operation_generation: u64,
        viewport_id: &str,
        expected_presentation_revision: u64,
        result: ControlObjectFilterResult,
    ) -> Option<Value> {
        let viewport_id = ViewportId::new(viewport_id).ok()?;
        if self.pending_object_filters.get(&viewport_id).copied() != Some(operation_generation) {
            return None;
        }
        if document_generation != self.document_generation
            || resource_generation != self.installed_object_resource_generation
        {
            self.pending_object_filters.remove(&viewport_id);
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                viewport_id.as_str(),
                operation_generation,
                "Object filter superseded by newer resource state",
            );
            return None;
        }
        let (total_count, presentation_matches) = {
            let dataset = self.dataset.as_ref()?;
            let viewport = dataset.workspace.get(&viewport_id)?;
            (
                dataset
                    .object_resource
                    .as_ref()
                    .map_or(0, |resource| resource.features.len()),
                viewport.presentation_revision == expected_presentation_revision,
            )
        };
        if !presentation_matches {
            self.pending_object_filters.remove(&viewport_id);
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                viewport_id.as_str(),
                operation_generation,
                "Object filter superseded by newer viewport presentation",
            );
            return None;
        }
        self.pending_object_filters.remove(&viewport_id);
        let dataset = self.dataset.as_mut()?;
        let active_before = dataset.workspace.active().state.clone();
        let viewport = dataset.workspace.get_mut(&viewport_id)?;
        set_object_filter_model(&mut viewport.state.objects, result.model);
        viewport.state.object_filter_indices = result.matching_indices;
        viewport.state.object_filter_active = result.active;
        viewport.state.object_filter_revision =
            viewport.state.object_filter_revision.wrapping_add(1).max(1);
        let snapshot = object_filter_snapshot(&viewport.state, total_count);
        let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        let active_changed =
            presentation_changed(&active_before, &dataset.workspace.active().state);
        let response = viewport_response(
            &dataset.workspace,
            &viewport_id,
            snapshot,
            vec![viewport_id.clone()],
            active_changed,
        );
        self.readiness.finish_scoped(
            OperationKind::ObjectFilter,
            viewport_id.as_str(),
            operation_generation,
            "Object filter ready",
        );
        Some(response)
    }

    fn get_object_filter(&self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let dataset = self.dataset()?;
        let viewport = dataset
            .workspace
            .get(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        Ok(viewport_response(
            &dataset.workspace,
            &viewport_id,
            object_filter_snapshot(
                &viewport.state,
                dataset
                    .object_resource
                    .as_ref()
                    .map_or(0, |resource| resource.features.len()),
            ),
            vec![viewport_id.clone()],
            false,
        ))
    }

    fn clear_object_filter(&mut self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let total_count = self
            .dataset()?
            .object_resource
            .as_ref()
            .map_or(0, |resource| resource.features.len());
        if let Some(generation) = self.pending_object_filters.remove(&viewport_id) {
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                viewport_id.as_str(),
                generation,
                "Object filter cleared",
            );
        }
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        set_object_filter_model(&mut viewport.state.objects, default_object_filter_model());
        viewport.state.object_filter_indices = Arc::new(Vec::new());
        viewport.state.object_filter_active = false;
        viewport.state.object_filter_revision =
            viewport.state.object_filter_revision.wrapping_add(1).max(1);
        let snapshot = object_filter_snapshot(&viewport.state, total_count);
        let _ = workspace.bump_presentation_revision(&viewport_id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            snapshot,
            vec![viewport_id.clone()],
            active_changed,
        ))
    }

    fn native_layer_id<'a>(params: &'a Value) -> Result<&'a str, ControlError> {
        params
            .get("layer_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("layer_id is required"))
    }

    fn effective_native_layers(viewport: &ViewportModel) -> Vec<Value> {
        let mut layers = viewport.native_layers.snapshots();
        for layer in &mut layers {
            let Some(layer_id) = layer.get("layer_id").and_then(Value::as_str) else {
                continue;
            };
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
                && let Some(channel) = viewport.channels.get(index)
            {
                layer["visible"] = Value::Bool(channel.visible);
                layer["presentation"] = json!({
                    "visible": channel.visible,
                    "color_rgb": channel.color_rgb,
                    "window": channel.window.map(|(min,max)| json!({"min":min,"max":max})),
                });
                layer["offset_world"] = json!(channel.offset_world);
                layer["order"] = json!(
                    viewport
                        .channel_order
                        .iter()
                        .position(|candidate| *candidate == index)
                        .unwrap_or(index)
                );
            } else if layer_id == "segmentation_objects" {
                layer["visible"] = Value::Bool(
                    viewport
                        .objects
                        .get("visible")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                );
                layer["presentation"] = viewport.objects.clone();
            } else if layer_id == "segmentation_labels" {
                layer["visible"] = Value::Bool(viewport.segmentation_labels_visible);
                if let Some(presentation) = layer["presentation"].as_object_mut() {
                    presentation.insert(
                        "visible".to_string(),
                        Value::Bool(viewport.segmentation_labels_visible),
                    );
                }
            } else if layer_id == "segmentation_geojson" {
                layer["visible"] = Value::Bool(viewport.segmentation_geojson_visible);
                if let Some(presentation) = layer["presentation"].as_object_mut() {
                    presentation.insert(
                        "visible".to_string(),
                        Value::Bool(viewport.segmentation_geojson_visible),
                    );
                }
            }
        }
        let mut channels = layers
            .iter()
            .filter(|layer| layer.get("stack").and_then(Value::as_str) == Some("channels"))
            .cloned()
            .collect::<Vec<_>>();
        channels.sort_by_key(|layer| {
            layer
                .get("order")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX)
        });
        let overlays = layers
            .into_iter()
            .filter(|layer| layer.get("stack").and_then(Value::as_str) == Some("overlays"));
        channels.extend(overlays);
        channels
    }

    fn native_layers_for(&self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let viewport = workspace
            .get(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        Ok(viewport_response(
            workspace,
            &viewport_id,
            Value::Array(Self::effective_native_layers(&viewport.state)),
            vec![viewport_id.clone()],
            false,
        ))
    }

    fn native_layer_for(&self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let layer_id = Self::native_layer_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let viewport = workspace
            .get(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let layer = Self::effective_native_layers(&viewport.state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id))
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
        Ok(viewport_response(
            workspace,
            &viewport_id,
            layer,
            vec![viewport_id.clone()],
            false,
        ))
    }

    fn apply_native_layer_visibility(
        state: &mut ViewportModel,
        layer_id: &str,
        visible: bool,
    ) -> Result<bool, ControlError> {
        let mut changed = state.native_layers.set_visibility(layer_id, visible)?;
        if let Some(index) = layer_id
            .strip_prefix("channel:")
            .and_then(|value| value.parse::<usize>().ok())
        {
            let channel = state
                .channels
                .get_mut(index)
                .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
            changed |= channel.visible != visible;
            channel.visible = visible;
        } else if layer_id == "segmentation_objects" {
            let previous = state
                .objects
                .get("visible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            changed |= previous != visible;
            state
                .objects
                .as_object_mut()
                .expect("object presentation is normalized")
                .insert("visible".to_string(), Value::Bool(visible));
        } else if layer_id == "segmentation_labels" {
            changed |= state.segmentation_labels_visible != visible;
            state.segmentation_labels_visible = visible;
        } else if layer_id == "segmentation_geojson" {
            changed |= state.segmentation_geojson_visible != visible;
            state.segmentation_geojson_visible = visible;
        }
        Ok(changed)
    }

    fn apply_native_layer_presentation(
        state: &mut ViewportModel,
        layer_id: &str,
        params: &Value,
    ) -> Result<bool, ControlError> {
        let presentation = params.get("presentation").unwrap_or(params);
        let mut changed = state
            .native_layers
            .set_presentation(layer_id, presentation)?;
        if let Some(index) = layer_id
            .strip_prefix("channel:")
            .and_then(|value| value.parse::<usize>().ok())
        {
            let channel = state
                .channels
                .get_mut(index)
                .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
            let before = channel.clone();
            if let Some(visible) = presentation.get("visible").and_then(Value::as_bool) {
                channel.visible = visible;
            }
            if let Some(color) = presentation.get("color_rgb") {
                let values = color
                    .as_array()
                    .filter(|values| values.len() == 3)
                    .ok_or_else(|| {
                        invalid("color_rgb must contain three integers from 0 to 255")
                    })?;
                channel.color_rgb = [to_u8(&values[0])?, to_u8(&values[1])?, to_u8(&values[2])?];
            }
            if let Some(window) = presentation.get("window").filter(|value| !value.is_null()) {
                let (min, max) = if let Some(values) = window.as_array().filter(|v| v.len() == 2) {
                    (values[0].as_f64(), values[1].as_f64())
                } else {
                    (
                        window.get("min").and_then(Value::as_f64),
                        window.get("max").and_then(Value::as_f64),
                    )
                };
                let (Some(min), Some(max)) = (min, max) else {
                    return Err(invalid(
                        "window must be [min, max] or an object containing min and max",
                    ));
                };
                if !min.is_finite() || !max.is_finite() || max <= min {
                    return Err(invalid(
                        "window values must be finite and max must be greater than min",
                    ));
                }
                channel.window = Some((min as f32, max as f32));
            }
            changed |= *channel != before;
        } else if layer_id == "segmentation_objects" {
            changed |= apply_object_style_patch(&mut state.objects, presentation)?;
        } else if layer_id == "segmentation_labels" {
            if let Some(visible) = presentation.get("visible").and_then(Value::as_bool) {
                changed |= state.segmentation_labels_visible != visible;
                state.segmentation_labels_visible = visible;
            }
        } else if layer_id == "segmentation_geojson" {
            if let Some(visible) = presentation.get("visible").and_then(Value::as_bool) {
                changed |= state.segmentation_geojson_visible != visible;
                state.segmentation_geojson_visible = visible;
            }
        }
        Ok(changed)
    }

    fn mutate_native_layer(
        &mut self,
        params: &Value,
        operation: impl FnOnce(&mut ViewportModel, &str) -> Result<bool, ControlError>,
    ) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let layer_id = Self::native_layer_id(params)?.to_string();
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let changed = operation(&mut viewport.state, &layer_id)?;
        if changed {
            let _ = workspace.bump_presentation_revision(&viewport_id);
        }
        let layer = Self::effective_native_layers(&workspace.get(&viewport_id).unwrap().state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id.as_str()))
            .expect("mutated native layer remains present");
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            json!({"changed":changed,"layer":layer}),
            vec![viewport_id.clone()],
            active_changed,
        ))
    }

    fn set_native_layer_visibility(&mut self, params: &Value) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible is required"))?;
        self.mutate_native_layer(params, |state, layer_id| {
            Self::apply_native_layer_visibility(state, layer_id, visible)
        })
    }

    fn set_native_layer_presentation(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.mutate_native_layer(params, |state, layer_id| {
            Self::apply_native_layer_presentation(state, layer_id, params)
        })
    }

    fn set_native_layer_active(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.mutate_native_layer(params, |state, layer_id| {
            state.native_layers.set_active(layer_id)
        })
    }

    fn set_native_layer_order(&mut self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let stack = params
            .get("stack")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("stack is required"))?;
        let layers = params
            .get("layers")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("layers is required"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("layer IDs must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let changed = viewport.state.native_layers.set_order(stack, &layers)?;
        if stack == "channels" {
            viewport.state.channel_order = layers
                .iter()
                .map(|id| {
                    id.strip_prefix("channel:")
                        .and_then(|value| value.parse::<usize>().ok())
                        .ok_or_else(|| invalid("channels stack accepts only channel layers"))
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        if changed {
            let _ = workspace.bump_presentation_revision(&viewport_id);
        }
        let snapshots = Self::effective_native_layers(&workspace.get(&viewport_id).unwrap().state);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            json!({"changed":changed,"layers":snapshots}),
            vec![viewport_id.clone()],
            active_changed,
        ))
    }

    fn replace_native_layers(&mut self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let state = params
            .get("state")
            .or_else(|| params.get("layers"))
            .ok_or_else(|| invalid("state is required"))?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let changed = viewport.state.native_layers.replace(state)?;
        let snapshots = viewport.state.native_layers.snapshots();
        for layer in &snapshots {
            let Some(layer_id) = layer.get("layer_id").and_then(Value::as_str) else {
                continue;
            };
            if let Some(presentation) = layer.get("presentation") {
                Self::apply_native_layer_presentation(&mut viewport.state, layer_id, presentation)?;
            }
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
                && let Some(offset) = layer
                    .get("offset_world")
                    .and_then(Value::as_array)
                    .filter(|values| values.len() == 2)
                    .and_then(|values| {
                        Some([values[0].as_f64()? as f32, values[1].as_f64()? as f32])
                    })
                && let Some(channel) = viewport.state.channels.get_mut(index)
            {
                channel.offset_world = offset;
            }
        }
        let channel_order = snapshots
            .iter()
            .filter(|layer| layer.get("stack").and_then(Value::as_str) == Some("channels"))
            .filter_map(|layer| {
                Some((
                    layer.get("order").and_then(Value::as_u64)?,
                    layer
                        .get("layer_id")
                        .and_then(Value::as_str)?
                        .strip_prefix("channel:")?
                        .parse::<usize>()
                        .ok()?,
                ))
            })
            .collect::<Vec<_>>();
        if channel_order.len() == viewport.state.channels.len() {
            let mut channel_order = channel_order;
            channel_order.sort_by_key(|(order, _)| *order);
            viewport.state.channel_order =
                channel_order.into_iter().map(|(_, index)| index).collect();
        }
        if let Some(index) = viewport
            .state
            .native_layers
            .active_layer_id()
            .and_then(|layer_id| layer_id.strip_prefix("channel:"))
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|index| *index < viewport.state.channels.len())
        {
            viewport.state.active_channel = index;
        }
        if changed {
            let _ = workspace.bump_presentation_revision(&viewport_id);
        }
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            json!({"changed":changed,"layers":Self::effective_native_layers(&workspace.get(&viewport_id).unwrap().state)}),
            vec![viewport_id.clone()],
            active_changed,
        ))
    }

    fn active_scoped_native_params(&self, params: &Value) -> Result<Value, ControlError> {
        self.active_scoped_params(params)
    }

    fn native_layers_global(&self) -> Result<Value, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        Ok(json!({"mode":"single","layers":Self::effective_native_layers(viewport)}))
    }

    fn native_layer_global(&self, params: &Value) -> Result<Value, ControlError> {
        let layer_id = Self::native_layer_id(params)?;
        let layer = Self::effective_native_layers(&self.dataset()?.workspace.active().state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id))
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
        Ok(json!({"mode":"single","layer":layer}))
    }

    fn unwrap_native_global_result(
        &mut self,
        method: &str,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let scoped = self.active_scoped_native_params(params)?;
        let response = match method {
            "viewer.native_layers.set_active" => self.set_native_layer_active(&scoped)?,
            "viewer.native_layers.set_visibility" => self.set_native_layer_visibility(&scoped)?,
            "viewer.native_layers.set_order" => self.set_native_layer_order(&scoped)?,
            _ => unreachable!("global native layer mutation was checked"),
        };
        Ok(json!({"mode":"single","result":response["result"].clone()}))
    }

    fn set_native_layer_offset_global(
        &mut self,
        params: &Value,
        reset: bool,
    ) -> Result<Value, ControlError> {
        let layer_id = Self::native_layer_id(params)?.to_string();
        let offset = if reset {
            None
        } else {
            Some(
                optional_finite_pair(params, "offset_world")?
                    .ok_or_else(|| invalid("offset_world is required"))?,
            )
        };
        let dataset = self.dataset_mut()?;
        let mut changed = false;
        let viewport_ids = dataset
            .workspace
            .viewports()
            .iter()
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        for viewport in dataset.workspace.viewports_mut() {
            let layer_changed = if let Some(offset) = offset {
                viewport.state.native_layers.set_offset(&layer_id, offset)?
            } else {
                viewport.state.native_layers.reset_offset(&layer_id)?
            };
            changed |= layer_changed;
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
                && let Some(channel) = viewport.state.channels.get_mut(index)
            {
                let effective = viewport
                    .state
                    .native_layers
                    .get(&layer_id)
                    .expect("offset native layer remains present")
                    .offset_world;
                changed |= channel.offset_world != effective;
                channel.offset_world = effective;
            }
        }
        if changed {
            for viewport_id in &viewport_ids {
                let _ = dataset.workspace.bump_presentation_revision(viewport_id);
            }
        }
        let layer = Self::effective_native_layers(&dataset.workspace.active().state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id.as_str()))
            .expect("offset native layer remains present");
        Ok(json!({"mode":"single","result":{"changed":changed,"layer":layer}}))
    }

    fn project_view_spec(viewport: &ViewportModel, has_objects: bool) -> Value {
        let color_property = viewport
            .objects
            .get("color_property")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let overrides = viewport
            .objects
            .get("color_level_overrides")
            .and_then(Value::as_object);
        let hidden_cell_types = overrides
            .into_iter()
            .flatten()
            .filter(|(_, style)| style.get("visible").and_then(Value::as_bool) == Some(false))
            .map(|(value, _)| value.clone())
            .collect::<Vec<_>>();
        let visible_cell_types = if hidden_cell_types.is_empty() {
            Vec::new()
        } else {
            overrides
                .into_iter()
                .flatten()
                .filter(|(_, style)| style.get("visible").and_then(Value::as_bool) != Some(false))
                .map(|(value, _)| value.clone())
                .collect::<Vec<_>>()
        };
        let uses_objects = has_objects
            || color_property.is_some()
            || viewport
                .objects
                .get("fill_cells")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let active = viewport
            .channels
            .get(viewport.active_channel)
            .or_else(|| viewport.channels.first());
        let mut spec = json!({
            "channel_ref": active.map(|channel| json!({"label":channel.name,"alias":""})),
            "visible_channel_refs": viewport.channels.iter().filter(|channel| channel.visible).map(|channel| json!({"label":channel.name,"alias":""})).collect::<Vec<_>>(),
            "camera": {
                "center_world_lvl0": viewport.center,
                "zoom_screen_per_lvl0_px": viewport.zoom,
            },
        });
        if uses_objects {
            spec["segmentation_source"] = Value::String("geoparquet".to_string());
            spec["load_labels"] = Value::Bool(false);
            spec["cell_color_by"] =
                color_property.map_or(Value::Null, |value| Value::String(value.to_string()));
            spec["visible_cell_types"] = json!(visible_cell_types);
            spec["hidden_cell_types"] = json!(hidden_cell_types);
            spec["fill_cells"] = viewport
                .objects
                .get("fill_cells")
                .cloned()
                .unwrap_or(Value::Bool(false));
            spec["show_selection_overlay"] = viewport
                .objects
                .get("show_selection_overlay")
                .cloned()
                .unwrap_or(Value::Bool(true));
        }
        spec
    }

    fn capture_project_view(&mut self, params: &Value) -> Result<Value, ControlError> {
        let name = required_nonempty_string(params, &["name"], "name")?;
        let viewport_id = params
            .get("viewport_id")
            .and_then(Value::as_str)
            .map(ViewportId::new)
            .transpose()
            .map_err(|error| invalid(error.to_string()))?
            .unwrap_or_else(|| self.dataset().unwrap().workspace.active_id().clone());
        let spec = {
            let dataset = self.dataset()?;
            let viewport = dataset
                .workspace
                .get(&viewport_id)
                .ok_or_else(|| not_found(&viewport_id))?;
            Self::project_view_spec(&viewport.state, dataset.object_resource.is_some())
        };
        let view = self
            .project
            .dispatch("project.views.create", &json!({"name":name,"spec":spec}))?;
        self.project_initialized = true;
        Ok(json!({
            "captured":true,
            "viewport_id":params.get("viewport_id").cloned().unwrap_or(Value::Null),
            "view":view,
        }))
    }

    fn saved_view_channel_index(
        channels: &[ModelChannel],
        spec: &Value,
    ) -> Result<Option<usize>, ControlError> {
        let candidates = spec
            .get("channel_ref")
            .and_then(Value::as_object)
            .map(|reference| {
                ["alias", "label"]
                    .into_iter()
                    .filter_map(|name| reference.get(name).and_then(Value::as_str))
                    .chain(spec.get("channel").and_then(Value::as_str))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                spec.get("channel")
                    .and_then(Value::as_str)
                    .into_iter()
                    .collect()
            });
        for candidate in candidates {
            if candidate.trim().is_empty() {
                continue;
            }
            if let Ok(index) = resolve_channel(channels, &Value::String(candidate.to_string())) {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    fn saved_view_visible_channel_indices(
        channels: &[ModelChannel],
        spec: &Value,
    ) -> Result<Vec<usize>, ControlError> {
        let mut indices = Vec::new();
        if let Some(references) = spec.get("visible_channel_refs").and_then(Value::as_array) {
            for reference in references {
                let mut found = None;
                for candidate in ["alias", "label"]
                    .into_iter()
                    .filter_map(|name| reference.get(name).and_then(Value::as_str))
                {
                    if let Ok(index) =
                        resolve_channel(channels, &Value::String(candidate.to_string()))
                    {
                        found = Some(index);
                        break;
                    }
                }
                if let Some(index) = found
                    && !indices.contains(&index)
                {
                    indices.push(index);
                }
            }
        }
        if let Some(names) = spec.get("visible_channels").and_then(Value::as_array) {
            for name in names {
                let index = resolve_channel(channels, name)?;
                if !indices.contains(&index) {
                    indices.push(index);
                }
            }
        }
        Ok(indices)
    }

    fn apply_project_view(&mut self, params: &Value) -> Result<Value, ControlError> {
        let view = self.project.dispatch("project.views.get", params)?;
        let spec = view.get("spec").cloned().unwrap_or_else(|| json!({}));
        let dataset = self.dataset_mut()?;
        let viewport_id = dataset.workspace.active_id().clone();
        let before = dataset.workspace.active().state.clone();
        let viewport = &mut dataset.workspace.active_mut().state;

        if let Some(index) = Self::saved_view_channel_index(&viewport.channels, &spec)? {
            viewport.active_channel = index;
        }
        let visible = Self::saved_view_visible_channel_indices(&viewport.channels, &spec)?;
        if !visible.is_empty() {
            for (index, channel) in viewport.channels.iter_mut().enumerate() {
                channel.visible = visible.contains(&index);
            }
        }
        if let Some(hidden) = spec.get("hidden_channels").and_then(Value::as_array) {
            for selector in hidden {
                let index = resolve_channel(&viewport.channels, selector)?;
                viewport.channels[index].visible = false;
            }
        }
        if let Some(value) = spec.get("cell_color_by") {
            viewport
                .objects
                .as_object_mut()
                .expect("object presentation is normalized")
                .insert("color_property".to_string(), value.clone());
        }
        for (name, visible) in [("visible_cell_types", true), ("hidden_cell_types", false)] {
            if let Some(values) = spec.get(name).and_then(Value::as_array) {
                let overrides = viewport
                    .objects
                    .as_object_mut()
                    .expect("object presentation is normalized")
                    .entry("color_level_overrides")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .expect("object legend overrides are an object");
                for value in values.iter().filter_map(Value::as_str) {
                    overrides
                        .entry(value.to_string())
                        .or_insert_with(|| json!({}))["visible"] = Value::Bool(visible);
                }
            }
        }
        for name in ["fill_cells", "show_selection_overlay"] {
            if let Some(value) = spec.get(name).and_then(Value::as_bool) {
                viewport
                    .objects
                    .as_object_mut()
                    .expect("object presentation is normalized")
                    .insert(name.to_string(), Value::Bool(value));
            }
        }
        if let Some(camera) = spec.get("camera") {
            if let Some(center) = camera
                .get("center_world_lvl0")
                .and_then(Value::as_array)
                .filter(|values| values.len() == 2)
            {
                viewport.center = [
                    center[0]
                        .as_f64()
                        .ok_or_else(|| invalid("saved view camera x is invalid"))?
                        as f32,
                    center[1]
                        .as_f64()
                        .ok_or_else(|| invalid("saved view camera y is invalid"))?
                        as f32,
                ];
            }
            if let Some(zoom) = camera
                .get("zoom_screen_per_lvl0_px")
                .and_then(Value::as_f64)
            {
                if !zoom.is_finite() || zoom <= 0.0 {
                    return Err(invalid(
                        "saved view camera zoom must be positive and finite",
                    ));
                }
                viewport.zoom = zoom as f32;
            }
        }
        let after = viewport.clone();
        let navigation_changed = after.center != before.center || after.zoom != before.zoom;
        let presentation_changed = presentation_changed(&before, &after);
        if navigation_changed {
            let _ = dataset.workspace.bump_navigation_revision(&viewport_id);
            if dataset.workspace.links().camera {
                propagate_camera(&mut dataset.workspace, &viewport_id, &after);
            }
        }
        if presentation_changed {
            let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        }
        Ok(json!({"applied":true,"view":view}))
    }

    pub fn render_workspace_snapshot(&self) -> Option<Value> {
        self.workspace_snapshot().ok()
    }

    pub fn set_mode(&mut self, mode: ModelMode) {
        self.mode = mode;
        if mode != ModelMode::Single {
            self.dataset = None;
        }
        self.readiness
            .cancel_all_pending("Superseded by application mode change");
    }

    pub fn bootstrap_mode_from_renderer(&mut self, mode: ModelMode) {
        self.document_generation = self.document_generation.wrapping_add(1).max(1);
        self.set_mode(mode);
    }

    pub fn mark_projection_dirty(&mut self) -> u64 {
        self.projection_revision = self.projection_revision.wrapping_add(1).max(1);
        self.projection_revision
    }

    pub fn mark_projection_presented(&mut self, revision: u64) {
        self.presented_projection_revision = self.presented_projection_revision.max(revision);
    }

    pub fn report_viewport_geometry(&mut self, viewport_id: &str, width: f32, height: f32) {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return;
        }
        let Ok(id) = ViewportId::new(viewport_id) else {
            return;
        };
        if let Some(dataset) = self.dataset.as_mut()
            && let Some(viewport) = dataset.workspace.get_mut(&id)
        {
            viewport.state.logical_size = [width, height];
            self.measured_viewports.insert(id);
            if dataset
                .workspace
                .viewports()
                .iter()
                .all(|viewport| self.measured_viewports.contains(&viewport.id))
            {
                dataset.logical_workspace_size = observed_workspace_size(&dataset.workspace);
                dataset.geometry_source = GeometrySource::Observed;
            }
        }
    }

    pub fn begin_dataset_open(&mut self, source: impl Into<String>) -> u64 {
        for kind in [
            OperationKind::Document,
            OperationKind::Labels,
            OperationKind::Objects,
            OperationKind::ObjectFilter,
            OperationKind::MaskIo,
        ] {
            self.readiness
                .cancel_kind_pending(kind, "Superseded by newer document request");
        }
        self.object_resource_pending = false;
        self.pending_object_filters.clear();
        self.pending_object_selection_filter = None;
        if let Some(dataset) = self.dataset.as_mut() {
            dataset.label_pending = false;
        }
        self.document_generation = self.document_generation.wrapping_add(1).max(1);
        self.mode = ModelMode::Transition;
        self.readiness.begin(
            OperationKind::Document,
            self.document_generation,
            format!("Opening {}", source.into()),
        );
        self.document_generation
    }

    pub fn begin_dataset_inspection(&mut self, path: &Path) -> (u64, String) {
        self.dataset_inspection_operation_generation = self
            .dataset_inspection_operation_generation
            .wrapping_add(1)
            .max(1);
        let generation = self.dataset_inspection_operation_generation;
        let scope = path.to_string_lossy().into_owned();
        self.readiness.begin_scoped(
            OperationKind::DatasetInspection,
            scope.clone(),
            generation,
            format!("Inspecting {scope}"),
        );
        (generation, scope)
    }

    pub fn finish_dataset_inspection(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .finish_scoped(OperationKind::DatasetInspection, scope, generation, status)
    }

    pub fn cancel_dataset_inspection(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .cancel_scoped(OperationKind::DatasetInspection, scope, generation, status)
    }

    pub fn begin_deep_link_resolution(&mut self, scope: String) -> u64 {
        self.deep_link_resolve_operation_generation = self
            .deep_link_resolve_operation_generation
            .wrapping_add(1)
            .max(1);
        let generation = self.deep_link_resolve_operation_generation;
        self.readiness.begin_scoped(
            OperationKind::DeepLinkResolve,
            scope,
            generation,
            "Resolving deep link",
        );
        generation
    }

    pub fn finish_deep_link_resolution(&mut self, scope: &str, generation: u64) -> bool {
        self.readiness.finish_scoped(
            OperationKind::DeepLinkResolve,
            scope,
            generation,
            "Deep link resolved",
        )
    }

    pub fn cancel_deep_link_resolution(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .cancel_scoped(OperationKind::DeepLinkResolve, scope, generation, status)
    }

    pub fn fail_dataset_open(&mut self, message: impl Into<String>) {
        self.mode = ModelMode::Project;
        self.dataset = None;
        self.readiness
            .fail(OperationKind::Document, self.document_generation, message);
    }

    pub fn fail_dataset_open_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if generation != self.document_generation {
            return false;
        }
        self.fail_dataset_open(message);
        true
    }

    pub fn install_dataset(&mut self, dataset: &OmeZarrDataset) {
        let available = dataset
            .is_root_label_mask()
            .then(|| vec![LabelZarrDataset::root_label_name(dataset)])
            .unwrap_or_default();
        self.install_dataset_with_labels(dataset, available, None);
    }

    pub fn install_dataset_with_labels(
        &mut self,
        dataset: &OmeZarrDataset,
        mut label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) {
        if dataset.is_root_label_mask() {
            let root = LabelZarrDataset::root_label_name(dataset);
            if !label_available.contains(&root) {
                label_available.push(root);
            }
        }
        self.install_document_descriptor_with_labels(
            DocumentDescriptor::from_ome_zarr(dataset),
            label_available,
            root_label_resource,
        );
    }

    pub fn install_document_descriptor_with_labels(
        &mut self,
        descriptor: DocumentDescriptor,
        mut label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) {
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        self.installed_object_resource_generation = self.object_resource_generation;
        self.object_resource_pending = false;
        self.pending_object_filters.clear();
        self.pending_object_selection_filter = None;
        self.readiness.cancel_kind_pending(
            OperationKind::ObjectFilter,
            "Object filters superseded by dataset replacement",
        );
        let retained_geometry = self
            .dataset
            .as_ref()
            .map(|dataset| (dataset.logical_workspace_size, dataset.geometry_source));
        let level = descriptor.levels.first();
        let world_size = level
            .and_then(|level| {
                Some([
                    *level.shape.get(descriptor.dims.x)? as f32,
                    *level.shape.get(descriptor.dims.y)? as f32,
                ])
            })
            .unwrap_or([1.0, 1.0]);
        let plane_extents = level
            .map(|level| {
                [
                    descriptor
                        .dims
                        .z
                        .and_then(|dimension| level.shape.get(dimension).copied())
                        .unwrap_or(1),
                    level.shape.get(descriptor.dims.y).copied().unwrap_or(1),
                    level.shape.get(descriptor.dims.x).copied().unwrap_or(1),
                ]
            })
            .unwrap_or([1, 1, 1]);
        let mut viewport = ViewportModel::new(&descriptor.channels);
        let (logical_workspace_size, geometry_source) =
            retained_geometry.unwrap_or((DEFAULT_LOGICAL_CANVAS, GeometrySource::Bootstrap));
        viewport.logical_size = logical_workspace_size;
        fit_camera(&mut viewport, world_size);
        label_available.sort();
        label_available.dedup();
        let label_selected = if label_available.iter().any(|name| name == "cells") {
            "cells".to_string()
        } else {
            label_available.first().cloned().unwrap_or_default()
        };
        let loaded_root = root_label_resource
            .as_ref()
            .map(|resource| resource.dataset.label_name.clone());
        if root_label_resource.is_some() {
            viewport.native_layers.set_segmentation_labels(true, true);
            viewport.segmentation_labels_visible = true;
        }
        let source_key = descriptor.source.source_key();
        self.dataset = Some(DatasetModel {
            orthogonal_planes: descriptor.dims.z.is_some(),
            descriptor,
            world_size,
            plane_extents,
            logical_workspace_size,
            geometry_source,
            show_left_panel: true,
            show_right_panel: true,
            right_tab: "properties".to_string(),
            shared_resources: default_shared_resources(source_key),
            performance: default_performance_snapshot(),
            object_resource: None,
            label_available,
            label_selected: loaded_root.clone().unwrap_or(label_selected),
            label_loaded: loaded_root.clone(),
            label_resource: root_label_resource,
            label_status: loaded_root
                .map(|name| format!("Opened top-level label mask '{name}'."))
                .unwrap_or_default(),
            label_generation: 1,
            label_pending: false,
            label_actor_owned: true,
            object_selection: ObjectSelectionModel::default(),
            masks: MaskModel::default(),
            workspace: ViewportWorkspace::new(viewport),
        });
        self.measured_viewports
            .retain(|id| id.as_str() == "viewport-1");
        self.mode = ModelMode::Single;
        self.readiness
            .mark_ready(OperationKind::Document, self.document_generation, "Ready");
    }

    pub fn install_dataset_for_generation(
        &mut self,
        generation: u64,
        dataset: &OmeZarrDataset,
        label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) -> bool {
        let mut label_available = label_available;
        if dataset.is_root_label_mask() {
            let root = LabelZarrDataset::root_label_name(dataset);
            if !label_available.contains(&root) {
                label_available.push(root);
            }
        }
        self.install_document_for_generation(
            generation,
            DocumentDescriptor::from_ome_zarr(dataset),
            label_available,
            root_label_resource,
        )
    }

    pub fn install_document_for_generation(
        &mut self,
        generation: u64,
        descriptor: DocumentDescriptor,
        label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) -> bool {
        if generation != self.document_generation {
            return false;
        }
        self.install_document_descriptor_with_labels(
            descriptor,
            label_available,
            root_label_resource,
        );
        true
    }

    pub fn set_renderer_gpu_available(&mut self, available: bool) {
        self.renderer_gpu_available = available;
    }

    pub fn bootstrap_dataset_from_renderer(
        &mut self,
        dataset: &OmeZarrDataset,
        workspace: &Value,
    ) -> Result<(), ControlError> {
        self.document_generation = self.document_generation.wrapping_add(1).max(1);
        self.install_dataset(dataset);
        self.restore_renderer_workspace(workspace)
    }

    /// Merge renderer-owned compatibility data into the canonical model.
    ///
    /// The renderer is a projection consumer, not an authority for fields handled by
    /// [`Self::dispatch`]. In particular, channel metadata, transforms, presentation, panels,
    /// cameras, planes, and workspace structure must come back as typed native commands. A
    /// delayed renderer snapshot may be based on an older projection revision, so copying those
    /// fields here would allow a frame rendered after an occlusion to undo newer Python work.
    pub fn observe_renderer_workspace(
        &mut self,
        snapshot: &Value,
        based_on_projection_revision: u64,
    ) -> bool {
        if based_on_projection_revision > self.projection_revision {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        let observed_source = snapshot
            .get("shared_resources")
            .and_then(|resources| resources.get("dataset_source"))
            .and_then(Value::as_str);
        let canonical_source = dataset
            .shared_resources
            .get("dataset_source")
            .and_then(Value::as_str);
        if observed_source.is_some()
            && canonical_source.is_some()
            && observed_source != canonical_source
        {
            return false;
        }

        // These fields are still produced by renderer-side compatibility domains. They do not
        // overlap any actor-owned mutation and are therefore safe to merge even when the
        // observation is based on an older projection revision.
        if let Some(shared_resources) = snapshot.get("shared_resources") {
            dataset.shared_resources = shared_resources.clone();
        }
        if let Some(performance) = snapshot.get("performance") {
            dataset.performance = performance.clone();
        }
        if let Some(projected) = snapshot.get("viewports").and_then(Value::as_array) {
            for value in projected {
                let Some(id) = value
                    .get("viewport_id")
                    .and_then(Value::as_str)
                    .and_then(|id| ViewportId::new(id).ok())
                else {
                    continue;
                };
                let Some(native_layers) = value.get("native_layers") else {
                    continue;
                };
                if let Some(viewport) = dataset.workspace.get_mut(&id) {
                    // Compatibility renderers may discover a resource that has not yet moved to
                    // the actor. Adopt only missing descriptors: a delayed frame must never
                    // overwrite presentation already committed by Python or native commands.
                    let _ = viewport.state.native_layers.merge_missing(native_layers);
                }
            }
        }
        true
    }

    fn restore_renderer_workspace(&mut self, snapshot: &Value) -> Result<(), ControlError> {
        if let Some(right_tab) = snapshot
            .get("ui")
            .and_then(|ui| ui.get("right_tab"))
            .and_then(Value::as_str)
        {
            self.dataset_mut()?.right_tab = right_tab.to_string();
        }
        if let Some(labels) = snapshot.get("labels") {
            let available = labels
                .get("available")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>();
            let dataset = self.dataset_mut()?;
            dataset.label_available = available;
            dataset.label_selected = labels
                .get("selected")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            dataset.label_loaded = labels
                .get("loaded")
                .and_then(Value::as_str)
                .map(str::to_string);
            dataset.label_status = labels
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            dataset.label_pending = labels.get("busy").and_then(Value::as_bool).unwrap_or(false);
            dataset.label_generation = labels
                .get("generation")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .max(1);
            dataset.label_actor_owned = labels
                .get("actor_owned")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let Some(gpu_available) = labels.get("gpu_available").and_then(Value::as_bool) {
                self.renderer_gpu_available = gpu_available;
            }
        }
        if let Some(masks) = snapshot.get("masks") {
            self.dataset_mut()?.masks.restore_projection(masks)?;
        }
        if let Some(selection) = snapshot.get("object_selection") {
            let object_count = self
                .dataset()?
                .object_resource
                .as_ref()
                .map_or(0, |resource| resource.features.len());
            self.dataset_mut()?
                .object_selection
                .restore_projection(selection, object_count)?;
        }
        let projected = snapshot
            .get("viewports")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("renderer workspace has no viewport array"))?;
        if projected.is_empty() || projected.len() > crate::viewports::MAX_VIEWPORTS {
            return Err(invalid("renderer workspace has an invalid viewport count"));
        }
        let default_channels = self.dataset()?.workspace.active().state.channels.clone();
        let mut measured = HashSet::new();
        let mut slots = Vec::with_capacity(projected.len());
        for value in projected {
            let id = value
                .get("viewport_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("renderer viewport has no ID"))
                .and_then(|id| ViewportId::new(id).map_err(|error| invalid(error.to_string())))?;
            let title = value
                .get("title")
                .and_then(Value::as_str)
                .filter(|title| !title.trim().is_empty())
                .ok_or_else(|| invalid(format!("renderer viewport '{id}' has no title")))?
                .to_string();
            let mut state = ViewportModel {
                center: [0.0, 0.0],
                zoom: 1.0,
                logical_size: DEFAULT_LOGICAL_CANVAS,
                plane_mode: "xy".to_string(),
                plane_slices: [0, 0, 0],
                channels: default_channels.clone(),
                active_channel: 0,
                channel_order: default_channels
                    .iter()
                    .map(|channel| channel.index)
                    .collect(),
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
                native_layers: NativeLayersModel::channels(
                    &default_channels
                        .iter()
                        .map(|channel| ChannelInfo {
                            index: channel.index,
                            name: channel.name.clone(),
                            visible: channel.visible,
                            color_rgb: channel.color_rgb,
                            window: channel.window,
                            note: channel.note.clone(),
                        })
                        .collect::<Vec<_>>(),
                ),
            };
            apply_renderer_viewport(&mut state, value)?;
            if renderer_viewport_size(value).is_some() {
                measured.insert(id.clone());
            }
            slots.push(crate::viewports::ViewportSlot {
                id,
                title,
                state,
                navigation_revision: value
                    .get("navigation_revision")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .max(1),
                presentation_revision: value
                    .get("presentation_revision")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .max(1),
            });
        }
        let active = snapshot
            .get("active_viewport_id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("renderer workspace has no active viewport ID"))
            .and_then(|id| ViewportId::new(id).map_err(|error| invalid(error.to_string())))?;
        let layout = snapshot
            .get("layout")
            .and_then(Value::as_str)
            .and_then(ViewportLayout::parse)
            .ok_or_else(|| invalid("renderer workspace has an invalid layout"))?;
        let links = snapshot
            .get("links")
            .map(|links| ViewportLinks {
                camera: links.get("camera").and_then(Value::as_bool).unwrap_or(true),
                plane: links.get("plane").and_then(Value::as_bool).unwrap_or(true),
                selection: links
                    .get("selection")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            })
            .unwrap_or_default();
        let ratio = snapshot.get("ratio").and_then(Value::as_f64).unwrap_or(0.5) as f32;
        let revision = snapshot
            .get("revision")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        let workspace =
            ViewportWorkspace::restore_projection(slots, active, layout, links, ratio, revision)
                .map_err(|error| invalid(error.to_string()))?;
        let mut workspace = workspace;
        apply_renderer_channel_metadata(&mut workspace, snapshot);
        apply_renderer_channel_transforms(&mut workspace, snapshot);
        if let Some(presentation) = snapshot.get("channel_presentation") {
            let active = workspace.active_mut();
            if let Some(search) = presentation.get("search").and_then(Value::as_str) {
                active.state.channel_search = search.to_string();
            }
            if let Some(sort) = presentation
                .get("sort")
                .and_then(Value::as_str)
                .and_then(canonical_channel_sort)
            {
                active.state.channel_sort = sort.to_string();
            }
        }
        if let Some(panels) = snapshot.get("panels") {
            let dataset = self.dataset_mut()?;
            if let Some(left) = panels.get("left").and_then(Value::as_bool) {
                dataset.show_left_panel = left;
            }
            if let Some(right) = panels.get("right").and_then(Value::as_bool) {
                dataset.show_right_panel = right;
            }
        }
        let all_measured = workspace
            .viewports()
            .iter()
            .all(|viewport| measured.contains(&viewport.id));
        let dataset = self.dataset_mut()?;
        dataset.workspace = workspace;
        if let Some(shared_resources) = snapshot.get("shared_resources") {
            dataset.shared_resources = shared_resources.clone();
        }
        if let Some(performance) = snapshot.get("performance") {
            dataset.performance = performance.clone();
        }
        if all_measured {
            dataset.logical_workspace_size = observed_workspace_size(&dataset.workspace);
            dataset.geometry_source = GeometrySource::Observed;
        } else {
            update_logical_geometry(dataset);
        }
        self.measured_viewports = measured;
        Ok(())
    }

    pub fn loading_state(&self) -> Value {
        let canvas_ready = self.dataset.as_ref().is_some_and(|dataset| {
            !dataset.workspace.viewports().is_empty()
                && dataset
                    .workspace
                    .viewports()
                    .iter()
                    .all(|viewport| self.measured_viewports.contains(&viewport.id))
        });
        let geometry = self.dataset.as_ref().map(|dataset| {
            json!({
                "source": dataset.geometry_source.as_str(),
                "confidence": match dataset.geometry_source {
                    GeometrySource::Observed => "exact",
                    GeometrySource::Derived => "stable_estimate",
                    GeometrySource::Bootstrap => "bootstrap",
                },
                "workspace_size_points": dataset.logical_workspace_size,
            })
        });
        let resources_busy = self.readiness.any_pending();
        let loading = json!({
            "busy": resources_busy,
            "status": self.readiness.aggregate_status(),
            "model_ready": self.mode != ModelMode::Transition,
            "resources_ready": !resources_busy,
            "geometry_ready": self.dataset.is_some(),
            "presentation_ready": self.presented_projection_revision >= self.projection_revision,
            "canvas_ready": canvas_ready,
            "geometry": geometry,
            "projection_revision": self.projection_revision,
            "presented_projection_revision": self.presented_projection_revision,
            "operations": self.readiness.snapshot(),
        });
        let mut response = json!({
            "mode": self.mode.as_str(),
            "pending_deep_link": false,
            "loading": loading,
        });
        let object = response
            .as_object_mut()
            .expect("loading state response is an object");
        match self.mode {
            ModelMode::Project => {
                object.insert("busy".to_string(), Value::Bool(false));
                object.insert(
                    "note".to_string(),
                    Value::String("No dataset viewer is currently open.".to_string()),
                );
            }
            ModelMode::Transition => {
                object.insert("busy".to_string(), Value::Bool(true));
                object.insert("reasons".to_string(), json!(["transition"]));
            }
            ModelMode::Single | ModelMode::Mosaic => {}
        }
        response
    }

    fn application_state(&self) -> Value {
        let view = (self.mode == ModelMode::Single)
            .then(|| self.dataset.as_ref())
            .flatten()
            .map(|dataset| {
                let viewport = dataset.workspace.active();
                let active_channel = viewport
                    .state
                    .channels
                    .get(viewport.state.active_channel)
                    .map(|channel| json!({"index":channel.index,"name":channel.name}));
                let level0 = dataset.descriptor.levels.first();
                json!({
                    "dataset": dataset.descriptor.source.source_key(),
                    "dataset_descriptor": {
                        "source": dataset.descriptor.source.source_key(),
                        "kind": dataset.descriptor.kind,
                        "axes": dataset.descriptor.axes.iter().map(|axis| json!({
                            "name":axis.name,
                            "unit":axis.unit,
                        })).collect::<Vec<_>>(),
                        "shape": level0.map(|level| level.shape.clone()),
                        "chunks": level0.map(|level| level.chunks.clone()),
                        "dtype": level0.map(|level| level.dtype.clone()),
                        "scale": level0.map(|level| level.scale.clone()),
                        "translation": level0.map(|level| level.translation.clone()),
                        "pyramid_levels": dataset.descriptor.levels.len(),
                        "render_kind": match dataset.descriptor.render_kind {
                            DatasetRenderKind::Image => "image",
                            DatasetRenderKind::LabelMask => "labels",
                        },
                    },
                    "active_channel": active_channel,
                    "channel_count": viewport.state.channels.len(),
                    "visible_channels": viewport.state.channels.iter()
                        .filter(|channel| channel.visible)
                        .map(|channel| channel.name.clone())
                        .collect::<Vec<_>>(),
                })
            });
        json!({
            "mode":self.mode.as_str(),
            "view":view,
            "project":self.project.rois_json(),
        })
    }

    fn rendering_state(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        Ok(json!({
            "mode":"single",
            "gpu_available":self.renderer_gpu_available,
            "renderer":if self.renderer_gpu_available { "opengl" } else { "cpu" },
            "compositing":"additive",
            "smooth_pixels":dataset.workspace.active().state.smooth_pixels,
            "deterministic_capture":{
                "method":"viewer.screenshot.capture",
                "readiness":self.loading_state(),
            },
        }))
    }

    fn show_project(&mut self) -> Result<Value, ControlError> {
        if self.mode == ModelMode::Transition {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon is currently transitioning between views",
            ));
        }
        let changed = self.mode != ModelMode::Project;
        self.mode = ModelMode::Project;
        Ok(json!({"mode":"project","changed":changed}))
    }

    pub fn dispatch(
        &mut self,
        method: &str,
        params: &Value,
    ) -> Option<Result<ModelDispatch, ControlError>> {
        let supported = matches!(
            method,
            "app.get_state"
                | "app.settings.get"
                | "app.recent_projects.list"
                | "app.lifecycle.get"
                | "deep_links.parse"
                | "deep_links.filters.get"
                | "deep_links.generate"
                | "app.get_loading_state"
                | "get_loading_state"
                | "app.get_method_availability"
                | "app.navigation.show_project"
                | "project.rois.list"
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
                | "project.views.capture"
                | "project.views.apply"
                | "viewer.channels.list"
                | "viewer.channels.list_visible"
                | "viewer.channels.get_active"
                | "viewer.channels.set_active"
                | "viewer.channels.set_visible"
                | "viewer.channels.get_contrast"
                | "viewer.channels.set_contrast"
                | "viewer.channels.set_color"
                | "viewer.channels.set_note"
                | "viewer.channels.get_transform"
                | "viewer.channels.set_transform"
                | "viewer.channels.reset_transform"
                | "viewer.channels.set_order"
                | "viewer.channels.presentation.get"
                | "viewer.channels.presentation.set"
                | "viewer.channels.list_groups"
                | "viewer.channels.set_group"
                | "viewer.camera.get"
                | "viewer.camera.set"
                | "viewer.camera.zoom_in"
                | "viewer.camera.zoom_out"
                | "viewer.camera.fit"
                | "viewer.planes.get"
                | "viewer.planes.set"
                | "viewer.planes.next"
                | "viewer.planes.previous"
                | "viewer.planes.operation_availability"
                | "viewer.rendering.get_smooth_pixels"
                | "viewer.rendering.set_smooth_pixels"
                | "viewer.rendering.get_state"
                | "viewer.scale_bar.get"
                | "viewer.scale_bar.set"
                | "viewer.panels.get"
                | "viewer.panels.set"
                | "viewer.ui.set_right_tab"
                | "viewer.workspace.get"
                | "viewer.viewports.list"
                | "viewer.workspace.layout.get"
                | "viewer.workspace.layout.set"
                | "viewer.workspace.swap"
                | "viewer.viewports.get"
                | "viewer.viewports.create"
                | "viewer.viewports.clone"
                | "viewer.viewports.rename"
                | "viewer.viewports.remove"
                | "viewer.viewports.set_active"
                | "viewer.viewport_links.get"
                | "viewer.viewport_links.list"
                | "viewer.viewport_links.set"
                | "viewer.viewport_links.create"
                | "viewer.viewport_links.update"
                | "viewer.viewport_links.remove"
                | "viewer.viewports.camera.get"
                | "viewer.viewports.camera.set"
                | "viewer.viewports.camera.fit"
                | "viewer.viewports.planes.get"
                | "viewer.viewports.planes.set"
                | "viewer.viewports.channels.get"
                | "viewer.viewports.channels.set_visible"
                | "viewer.viewports.channels.set"
                | "viewer.viewports.channels.set_active"
                | "viewer.viewports.channels.set_color"
                | "viewer.viewports.channels.set_contrast"
                | "viewer.viewports.channels.set_order"
                | "viewer.viewports.channels.list_groups"
                | "viewer.viewports.channels.set_group"
                | "viewer.viewports.objects.style.get"
                | "viewer.viewports.objects.style.set"
                | "viewer.viewports.objects.legend.set"
                | "viewer.viewports.objects.filter.get"
                | "viewer.viewports.objects.filter.set"
                | "viewer.viewports.objects.filter.clear"
                | "viewer.viewports.layers.list"
                | "viewer.viewports.layers.get"
                | "viewer.viewports.layers.set"
                | "viewer.viewports.layers.set_visibility"
                | "viewer.viewports.layers.set_order"
                | "viewer.viewports.layers.set_active"
                | "viewer.viewports.layers.state.replace"
                | "viewer.objects.get_state"
                | "viewer.objects.get_visibility"
                | "viewer.objects.set_visibility"
                | "viewer.objects.style.get"
                | "viewer.objects.style.set"
                | "viewer.objects.legend.set"
                | "viewer.objects.rendering.get_fast"
                | "viewer.objects.rendering.set_fast"
                | "viewer.objects.source.clear"
                | "viewer.objects.source.cancel_load"
                | "viewer.objects.properties.list"
                | "viewer.objects.properties.load"
                | "viewer.objects.properties.values"
                | "viewer.objects.get_selection"
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
                | "viewer.objects.selection.state.replace"
                | "viewer.objects.get_filter"
                | "viewer.objects.set_filter"
                | "viewer.objects.clear_filter"
                | "viewer.objects.filters.set_model"
                | "viewer.objects.filters.get_revision"
                | "viewer.labels.list"
                | "viewer.labels.get"
                | "viewer.labels.load"
                | "viewer.labels.unload"
                | "viewer.labels.set_visibility"
                | "viewer.thresholds.levels.list"
                | "viewer.native_layers.list"
                | "viewer.native_layers.get"
                | "viewer.native_layers.set_active"
                | "viewer.native_layers.set_visibility"
                | "viewer.native_layers.set_order"
                | "viewer.native_layers.set_offset"
                | "viewer.native_layers.reset_offset"
                | "viewer.masks.layers.list"
                | "viewer.masks.layers.get"
                | "viewer.masks.layers.create"
                | "viewer.masks.layers.update"
                | "viewer.masks.layers.delete"
                | "viewer.masks.polygons.list"
                | "viewer.masks.polygons.add"
                | "viewer.masks.polygons.update"
                | "viewer.masks.polygons.remove"
                | "viewer.masks.selection.get"
                | "viewer.masks.selection.set"
                | "viewer.masks.selection.clear"
                | "viewer.masks.undo"
                | "viewer.masks.state.replace"
                | "viewer.masks.persistence.get"
                | "viewer.masks.persistence.sync"
                | "viewer.viewports.rendering.get"
                | "viewer.viewports.rendering.set"
        );
        if !supported {
            return None;
        }
        if matches!(method, "app.get_loading_state" | "get_loading_state") {
            return Some(Ok(ModelDispatch {
                response: self.loading_state(),
                present: false,
            }));
        }
        if method == "app.get_state" {
            if self.mode == ModelMode::Mosaic {
                return None;
            }
            return Some(Ok(ModelDispatch {
                response: self.application_state(),
                present: false,
            }));
        }
        if method == "app.settings.get" {
            return Some(Ok(ModelDispatch {
                response: self.settings_snapshot(),
                present: false,
            }));
        }
        if method == "app.recent_projects.list" {
            return Some(Ok(ModelDispatch {
                response: self.recent_projects_snapshot(),
                present: false,
            }));
        }
        if method == "app.lifecycle.get" {
            return Some(Ok(ModelDispatch {
                response: self.lifecycle_state(),
                present: false,
            }));
        }
        if method == "deep_links.parse" {
            return Some(Self::parse_deep_link(params).map(|response| ModelDispatch {
                response,
                present: false,
            }));
        }
        if method == "deep_links.filters.get" {
            return Some(
                Self::deep_link_filters(params).map(|response| ModelDispatch {
                    response,
                    present: false,
                }),
            );
        }
        if method == "deep_links.generate" {
            return Some(
                self.generate_deep_link(params)
                    .map(|response| ModelDispatch {
                        response,
                        present: false,
                    }),
            );
        }
        if method == "app.navigation.show_project" {
            return Some(self.show_project().map(|response| ModelDispatch {
                response,
                present: true,
            }));
        }
        if method == "app.get_method_availability" {
            let requested = params
                .get("methods")
                .and_then(Value::as_array)
                .map(|methods| {
                    methods
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                });
            return Some(Ok(ModelDispatch {
                response: crate::control::registry::availability_catalog(
                    self.mode.as_str(),
                    requested.as_deref(),
                ),
                present: false,
            }));
        }
        if matches!(method, "project.views.capture" | "project.views.apply")
            && self.project_operation_pending
        {
            return Some(Err(ControlError::new(
                ControlErrorKind::NotReady,
                format!("{method} cannot run while a project persistence transaction is active"),
            )));
        }
        if is_project_model_method(method) {
            if self.project_operation_pending {
                return Some(Err(ControlError::new(
                    ControlErrorKind::NotReady,
                    format!(
                        "{method} cannot run while a project persistence transaction is active"
                    ),
                )
                .with_data(json!({
                    "method": method,
                    "required_readiness": ["project"],
                    "loading": self.loading_state()["loading"],
                }))));
            }
            if self.mode == ModelMode::Transition {
                return Some(Err(ControlError::new(
                    ControlErrorKind::NotReady,
                    format!("{method} requires the project model to leave transition state"),
                )
                .with_data(json!({
                    "method": method,
                    "required_readiness": ["model"],
                    "loading": self.loading_state()["loading"],
                }))));
            }
            let result = self.project.dispatch(method, params);
            if result.is_ok()
                && !matches!(
                    method,
                    "project.rois.list"
                        | "project.get"
                        | "project.rois.get"
                        | "project.rois.get_selection"
                        | "project.views.list"
                        | "project.views.get"
                )
            {
                self.project_initialized = true;
            }
            if result.is_ok() && method == "project.create" {
                self.set_mode(ModelMode::Project);
            }
            let present = !matches!(
                method,
                "project.rois.list"
                    | "project.get"
                    | "project.rois.get"
                    | "project.rois.get_selection"
                    | "project.views.list"
                    | "project.views.get"
            );
            return Some(result.map(|response| ModelDispatch { response, present }));
        }
        if matches!(self.mode, ModelMode::Project | ModelMode::Mosaic) {
            return None;
        }
        if self.mode == ModelMode::Transition
            && !matches!(method, "app.get_loading_state" | "get_loading_state")
        {
            return Some(Err(ControlError::new(
                ControlErrorKind::NotReady,
                format!("{method} requires the dataset open to reach model/resource readiness"),
            )
            .with_data(json!({
                "method": method,
                "required_readiness": ["model", "resources"],
                "loading": self.loading_state()["loading"],
            }))));
        }
        if let Err(error) = self.check_viewport_revision(params) {
            return Some(Err(error));
        }
        let result = (|| -> Result<Value, ControlError> {
            Ok(match method {
                "app.get_loading_state" | "get_loading_state" | "app.get_method_availability" => {
                    unreachable!("mode-independent queries return before single-view dispatch")
                }
                "viewer.channels.list" => self.channels_snapshot()?,
                "viewer.channels.list_visible" => self.visible_channels_snapshot()?,
                "viewer.channels.get_active" => self.active_channel_snapshot()?,
                "viewer.channels.set_active" => self.set_active_channel_global(params)?,
                "viewer.channels.set_visible" => self.set_visible_channels_global(params)?,
                "viewer.channels.get_contrast" => self.get_channel_contrast_global(params)?,
                "viewer.channels.set_contrast" => self.set_channel_contrast_global(params)?,
                "viewer.channels.set_color" => self.set_channel_color_global(params)?,
                "viewer.channels.set_note" => self.set_channel_note_global(params)?,
                "viewer.channels.get_transform" => self.get_channel_transform(params)?,
                "viewer.channels.set_transform" => self.set_channel_transform(params)?,
                "viewer.channels.reset_transform" => self.reset_channel_transform(params)?,
                "viewer.channels.set_order" => self.set_channel_order_global(params)?,
                "viewer.channels.presentation.get" => self.channel_presentation_global()?,
                "viewer.channels.presentation.set" => {
                    self.set_channel_presentation_global(params)?
                }
                "viewer.channels.list_groups" => self.channel_groups_global()?,
                "viewer.channels.set_group" => self.set_channel_group_global(params)?,
                "viewer.camera.get" => self.get_camera_global()?,
                "viewer.camera.set" => self.set_camera_global(params)?,
                "viewer.camera.zoom_in" => self.zoom_camera_global(params, true)?,
                "viewer.camera.zoom_out" => self.zoom_camera_global(params, false)?,
                "viewer.camera.fit" => self.fit_camera_global()?,
                "viewer.planes.get" => self.get_plane_global()?,
                "viewer.planes.set" => self.set_plane_global(params)?,
                "viewer.planes.next" => self.step_plane_global(params, true)?,
                "viewer.planes.previous" => self.step_plane_global(params, false)?,
                "viewer.planes.operation_availability" => self.plane_operation_availability()?,
                "viewer.rendering.get_smooth_pixels" => self.get_smooth_pixels_global()?,
                "viewer.rendering.set_smooth_pixels" => self.set_smooth_pixels_global(params)?,
                "viewer.rendering.get_state" => self.rendering_state()?,
                "viewer.scale_bar.get" => self.get_scale_bar_global()?,
                "viewer.scale_bar.set" => self.set_scale_bar_global(params)?,
                "viewer.panels.get" => self.get_panels()?,
                "viewer.panels.set" => self.set_panels(params)?,
                "viewer.ui.set_right_tab" => self.set_right_tab(params)?,
                "project.views.capture" => self.capture_project_view(params)?,
                "project.views.apply" => self.apply_project_view(params)?,
                "viewer.workspace.get" | "viewer.viewports.list" => self.workspace_snapshot()?,
                "viewer.workspace.layout.get" => self.layout_snapshot()?,
                "viewer.workspace.layout.set" => self.set_layout(params)?,
                "viewer.workspace.swap" => self.swap_viewports()?,
                "viewer.viewports.get" => self.viewport_snapshot_for(params)?,
                "viewer.viewports.create" | "viewer.viewports.clone" => {
                    self.create_viewport(params)?
                }
                "viewer.viewports.rename" => self.rename_viewport(params)?,
                "viewer.viewports.remove" => self.remove_viewport(params)?,
                "viewer.viewports.set_active" => self.set_active_viewport(params)?,
                "viewer.viewport_links.get" => self.links_snapshot()?,
                "viewer.viewport_links.list" => self.link_groups_snapshot()?,
                "viewer.viewport_links.set" => self.set_links(params, LinkRequestKind::Direct)?,
                "viewer.viewport_links.create" => {
                    self.set_links(params, LinkRequestKind::Create)?
                }
                "viewer.viewport_links.update" => {
                    self.set_links(params, LinkRequestKind::Update)?
                }
                "viewer.viewport_links.remove" => self.remove_links(params)?,
                "viewer.viewports.camera.get" => self.get_camera(params)?,
                "viewer.viewports.camera.set" => self.set_camera(params)?,
                "viewer.viewports.camera.fit" => self.fit_viewport(params)?,
                "viewer.viewports.planes.get" => self.get_plane(params)?,
                "viewer.viewports.planes.set" => self.set_plane(params)?,
                "viewer.viewports.channels.get" => self.get_viewport_channels(params)?,
                "viewer.viewports.channels.set_visible" | "viewer.viewports.channels.set" => {
                    self.set_visible_channels(params)?
                }
                "viewer.viewports.channels.set_active" => self.set_active_channel(params)?,
                "viewer.viewports.channels.set_color" => self.set_channel_color(params)?,
                "viewer.viewports.channels.set_contrast" => self.set_channel_contrast(params)?,
                "viewer.viewports.channels.set_order" => self.set_channel_order(params)?,
                "viewer.viewports.channels.list_groups" => self.channel_groups(params)?,
                "viewer.viewports.channels.set_group" => self.set_channel_group(params)?,
                "viewer.viewports.objects.style.get" => self.get_object_style(params)?,
                "viewer.viewports.objects.style.set" => self.set_object_style(params)?,
                "viewer.viewports.objects.legend.set" => self.set_object_legend(params)?,
                "viewer.viewports.objects.filter.get" => self.get_object_filter(params)?,
                "viewer.viewports.objects.filter.set" => {
                    unreachable!("filter evaluation is dispatched to a resource worker")
                }
                "viewer.viewports.objects.filter.clear" => self.clear_object_filter(params)?,
                "viewer.viewports.layers.list" => self.native_layers_for(params)?,
                "viewer.viewports.layers.get" => self.native_layer_for(params)?,
                "viewer.viewports.layers.set" => self.set_native_layer_presentation(params)?,
                "viewer.viewports.layers.set_visibility" => {
                    self.set_native_layer_visibility(params)?
                }
                "viewer.viewports.layers.set_order" => self.set_native_layer_order(params)?,
                "viewer.viewports.layers.set_active" => self.set_native_layer_active(params)?,
                "viewer.viewports.layers.state.replace" => self.replace_native_layers(params)?,
                "viewer.objects.get_state" => json!({
                    "target": "segmentation_objects",
                    "state": self.object_resource_state(),
                }),
                "viewer.objects.get_visibility" => self.object_overlay_visibility_global(params)?,
                "viewer.objects.set_visibility" => {
                    self.set_object_overlay_visibility_global(params)?
                }
                "viewer.objects.style.get" => self.get_object_style_global(params)?,
                "viewer.objects.style.set" => self.set_object_style_global(params)?,
                "viewer.objects.legend.set" => self.set_object_legend_global(params)?,
                "viewer.objects.rendering.get_fast" => {
                    self.get_fast_object_rendering_global(params)?
                }
                "viewer.objects.rendering.set_fast" => {
                    self.set_fast_object_rendering_global(params)?
                }
                "viewer.objects.source.clear" => self.clear_object_resource()?,
                "viewer.objects.source.cancel_load" => self.cancel_object_resource_load(),
                "viewer.objects.properties.list" => self.object_properties_list(params)?,
                "viewer.objects.properties.load" => self.object_property_load(params)?,
                "viewer.objects.properties.values" => self.object_property_values(params)?,
                "viewer.objects.get_selection" => json!({
                    "mode":"single",
                    "objects":self.object_selection_get(params)?,
                }),
                "viewer.objects.query_rect" => json!({
                    "mode":"single",
                    "objects":self.object_selection_query_rect(params)?,
                }),
                "viewer.objects.query_view" => json!({
                    "mode":"single",
                    "objects":self.object_selection_query_view(params)?,
                }),
                "viewer.objects.query_lasso" => self.object_selection_query_lasso(params)?,
                "viewer.objects.select_rect" => json!({
                    "mode":"single",
                    "objects":self.object_selection_select_rect(params)?,
                }),
                "viewer.objects.select_lasso" => self.object_selection_select_lasso(params)?,
                "viewer.objects.clear_selection" => json!({
                    "mode":"single",
                    "objects":self.object_selection_clear(params)?,
                }),
                "viewer.objects.selection.select_ids" => {
                    self.object_selection_select_ids(params)?
                }
                "viewer.objects.selection.select_filtered" => {
                    self.object_selection_select_filtered(params)?
                }
                "viewer.objects.focus.set" => self.object_selection_focus(params)?,
                "viewer.objects.focus.clear" => self.object_selection_clear_focus(params)?,
                "viewer.objects.selection.state.replace" => {
                    self.object_selection_replace(params)?
                }
                "viewer.objects.get_filter" | "viewer.objects.filters.get_revision" => {
                    self.get_object_filter_global(params)?
                }
                "viewer.objects.set_filter" | "viewer.objects.filters.set_model" => {
                    unreachable!("filter evaluation is dispatched to a resource worker")
                }
                "viewer.objects.clear_filter" => self.clear_object_filter_global(params)?,
                "viewer.labels.list" | "viewer.labels.get" => self.labels_snapshot()?,
                "viewer.labels.load" => {
                    unreachable!("label loading is dispatched to a resource worker")
                }
                "viewer.labels.unload" => self.unload_labels()?,
                "viewer.labels.set_visibility" => self.set_labels_visibility(params)?,
                "viewer.thresholds.levels.list" => self.threshold_levels()?,
                "viewer.native_layers.list" => self.native_layers_global()?,
                "viewer.native_layers.get" => self.native_layer_global(params)?,
                "viewer.native_layers.set_active"
                | "viewer.native_layers.set_visibility"
                | "viewer.native_layers.set_order" => {
                    self.unwrap_native_global_result(method, params)?
                }
                "viewer.native_layers.set_offset" => {
                    self.set_native_layer_offset_global(params, false)?
                }
                "viewer.native_layers.reset_offset" => {
                    self.set_native_layer_offset_global(params, true)?
                }
                "viewer.masks.layers.list"
                | "viewer.masks.layers.get"
                | "viewer.masks.layers.create"
                | "viewer.masks.layers.update"
                | "viewer.masks.layers.delete"
                | "viewer.masks.polygons.list"
                | "viewer.masks.polygons.add"
                | "viewer.masks.polygons.update"
                | "viewer.masks.polygons.remove"
                | "viewer.masks.selection.get"
                | "viewer.masks.selection.set"
                | "viewer.masks.selection.clear"
                | "viewer.masks.undo"
                | "viewer.masks.state.replace" => {
                    let dataset = self.dataset_mut()?;
                    let response = dataset.masks.dispatch(method, params)?;
                    if !matches!(
                        method,
                        "viewer.masks.layers.list"
                            | "viewer.masks.layers.get"
                            | "viewer.masks.polygons.list"
                            | "viewer.masks.selection.get"
                    ) {
                        Self::sync_mask_native_layers(dataset);
                    }
                    response
                }
                "viewer.masks.persistence.get" => self.mask_persistence_state()?,
                "viewer.masks.persistence.sync" => self.sync_masks_to_project()?,
                "viewer.viewports.rendering.get" => self.get_rendering(params)?,
                "viewer.viewports.rendering.set" => self.set_rendering(params)?,
                _ => unreachable!("supported method set and dispatch match diverged"),
            })
        })();
        let present = !matches!(
            method,
            "app.get_loading_state"
                | "get_loading_state"
                | "viewer.channels.list"
                | "viewer.channels.list_visible"
                | "viewer.channels.get_active"
                | "viewer.channels.get_contrast"
                | "viewer.channels.get_transform"
                | "viewer.channels.presentation.get"
                | "viewer.channels.list_groups"
                | "viewer.camera.get"
                | "viewer.planes.get"
                | "viewer.planes.operation_availability"
                | "viewer.rendering.get_smooth_pixels"
                | "viewer.rendering.get_state"
                | "viewer.scale_bar.get"
                | "viewer.panels.get"
                | "viewer.workspace.get"
                | "viewer.viewports.list"
                | "viewer.workspace.layout.get"
                | "viewer.viewports.get"
                | "viewer.viewport_links.get"
                | "viewer.viewport_links.list"
                | "viewer.viewports.camera.get"
                | "viewer.viewports.planes.get"
                | "viewer.viewports.channels.get"
                | "viewer.viewports.channels.list_groups"
                | "viewer.viewports.objects.style.get"
                | "viewer.viewports.objects.filter.get"
                | "viewer.viewports.layers.list"
                | "viewer.viewports.layers.get"
                | "viewer.objects.get_state"
                | "viewer.objects.get_visibility"
                | "viewer.objects.style.get"
                | "viewer.objects.rendering.get_fast"
                | "viewer.objects.properties.list"
                | "viewer.objects.properties.load"
                | "viewer.objects.properties.values"
                | "viewer.objects.get_selection"
                | "viewer.objects.query_rect"
                | "viewer.objects.query_view"
                | "viewer.objects.query_lasso"
                | "viewer.objects.get_filter"
                | "viewer.objects.filters.get_revision"
                | "viewer.labels.list"
                | "viewer.labels.get"
                | "viewer.thresholds.levels.list"
                | "viewer.native_layers.list"
                | "viewer.native_layers.get"
                | "viewer.masks.layers.list"
                | "viewer.masks.layers.get"
                | "viewer.masks.polygons.list"
                | "viewer.masks.selection.get"
                | "viewer.masks.persistence.get"
                | "viewer.viewports.rendering.get"
        );
        Some(result.map(|response| ModelDispatch { response, present }))
    }

    fn dataset(&self) -> Result<&DatasetModel, ControlError> {
        self.dataset
            .as_ref()
            .ok_or_else(|| wrong_mode("No dataset viewer is currently open."))
    }

    fn dataset_mut(&mut self) -> Result<&mut DatasetModel, ControlError> {
        self.dataset
            .as_mut()
            .ok_or_else(|| wrong_mode("No dataset viewer is currently open."))
    }

    fn viewport_id(params: &Value) -> Result<ViewportId, ControlError> {
        let id = params
            .get("viewport_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("viewport_id is required"))?;
        ViewportId::new(id).map_err(|error| invalid(error.to_string()))
    }

    fn check_viewport_revision(&self, params: &Value) -> Result<(), ControlError> {
        let navigation = params.get("if_navigation_revision").and_then(Value::as_u64);
        let presentation = params
            .get("if_presentation_revision")
            .and_then(Value::as_u64);
        if navigation.is_none() && presentation.is_none() {
            return Ok(());
        }
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let viewport = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        if let Some(expected) = navigation
            && expected != viewport.navigation_revision
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "viewport navigation revision conflict: expected {expected}, current {}",
                    viewport.navigation_revision
                ),
            )
            .with_data(json!({
                "viewport_id": id.as_str(),
                "expected_revision": expected,
                "current_revision": viewport.navigation_revision,
                "revision_domain": "navigation",
            })));
        }
        if let Some(expected) = presentation
            && expected != viewport.presentation_revision
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "viewport presentation revision conflict: expected {expected}, current {}",
                    viewport.presentation_revision
                ),
            )
            .with_data(json!({
                "viewport_id": id.as_str(),
                "expected_revision": expected,
                "current_revision": viewport.presentation_revision,
                "revision_domain": "presentation",
            })));
        }
        Ok(())
    }

    fn channels_snapshot(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let active = dataset.workspace.active();
        Ok(json!({
            "mode": "single",
            "channels": active.state.channels.iter().enumerate().map(|(index, channel)| {
                let mut value = channel_json(channel, index == active.state.active_channel);
                value.as_object_mut().expect("channel snapshot is an object").insert(
                    "note".to_string(),
                    Value::String(channel.note.clone()),
                );
                value
            }).collect::<Vec<_>>(),
        }))
    }

    fn visible_channels_snapshot(&self) -> Result<Value, ControlError> {
        let active = self.dataset()?.workspace.active();
        Ok(json!({
            "mode": "single",
            "channels": visible_channels_json(&active.state),
        }))
    }

    fn active_channel_snapshot(&self) -> Result<Value, ControlError> {
        let active = self.dataset()?.workspace.active();
        let channel = active
            .state
            .channels
            .get(active.state.active_channel)
            .map(active_channel_json)
            .unwrap_or(Value::Null);
        Ok(json!({"mode": "single", "active_channel": channel}))
    }

    fn active_scoped_params(&self, params: &Value) -> Result<Value, ControlError> {
        let mut params = params.clone();
        params
            .as_object_mut()
            .ok_or_else(|| invalid("params must be an object"))?
            .insert(
                "viewport_id".to_string(),
                Value::String(self.dataset()?.workspace.active_id().as_str().to_string()),
            );
        Ok(params)
    }

    fn get_object_style_global(&self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let params = self.active_scoped_params(params)?;
        Ok(self.get_object_style(&params)?["result"].clone())
    }

    fn object_overlay_visibility_global(&self, params: &Value) -> Result<Value, ControlError> {
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("objects");
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        Ok(json!({
            "mode":"single",
            "overlay":{
                "target":target,
                "segmentation_labels":viewport.segmentation_labels_visible,
                "segmentation_geojson":viewport.segmentation_geojson_visible,
                "segmentation_objects":viewport.objects
                    .get("visible")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "object_count":dataset.object_resource
                    .as_ref()
                    .map_or(0, |resource| resource.features.len()),
            },
        }))
    }

    fn set_object_overlay_visibility_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_object_overlay_visibility requires visible"))?;
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("objects");
        if !matches!(target, "objects" | "labels" | "geojson" | "all") {
            return Err(invalid(format!("unknown overlay target '{target}'")));
        }

        let dataset = self.dataset_mut()?;
        let viewport_id = dataset.workspace.active_id().clone();
        let viewport = &mut dataset.workspace.active_mut().state;
        let mut changed = false;
        if matches!(target, "objects" | "all") {
            let current = viewport
                .objects
                .get("visible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            changed |= current != visible;
            viewport
                .objects
                .as_object_mut()
                .expect("object presentation is normalized")
                .insert("visible".to_string(), Value::Bool(visible));
            if viewport.native_layers.get("segmentation_objects").is_some() {
                changed |= viewport
                    .native_layers
                    .set_visibility("segmentation_objects", visible)?;
            }
        }
        if matches!(target, "labels" | "all") {
            changed |= viewport.segmentation_labels_visible != visible;
            viewport.segmentation_labels_visible = visible;
            if viewport.native_layers.get("segmentation_labels").is_some() {
                changed |= viewport
                    .native_layers
                    .set_visibility("segmentation_labels", visible)?;
            }
        }
        if matches!(target, "geojson" | "all") {
            changed |= viewport.segmentation_geojson_visible != visible;
            viewport.segmentation_geojson_visible = visible;
            if viewport.native_layers.get("segmentation_geojson").is_some() {
                changed |= viewport
                    .native_layers
                    .set_visibility("segmentation_geojson", visible)?;
            }
        }
        if changed {
            let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        }
        self.object_overlay_visibility_global(params)
    }

    fn set_object_style_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let params = self.active_scoped_params(params)?;
        Ok(self.set_object_style(&params)?["result"].clone())
    }

    fn set_object_legend_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let params = self.active_scoped_params(params)?;
        Ok(self.set_object_legend(&params)?["result"].clone())
    }

    fn get_fast_object_rendering_global(&self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let objects = &self.dataset()?.workspace.active().state.objects;
        Ok(json!({
            "enabled":objects.get("fast_rendering").and_then(Value::as_bool).unwrap_or(true),
        }))
    }

    fn set_fast_object_rendering_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let enabled = params
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("enabled is required"))?;
        let mut scoped = self.active_scoped_params(params)?;
        scoped
            .as_object_mut()
            .expect("active params are an object")
            .insert("fast_rendering".to_string(), Value::Bool(enabled));
        let response = self.set_object_style(&scoped)?;
        Ok(json!({
            "enabled":enabled,
            "changed":response["result"]["changed"],
        }))
    }

    fn get_object_filter_global(&self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let params = self.active_scoped_params(params)?;
        Ok(json!({
            "target":"segmentation_objects",
            "filter":self.get_object_filter(&params)?["result"].clone(),
        }))
    }

    fn clear_object_filter_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        Self::require_primary_object_target(params)?;
        let params = self.active_scoped_params(params)?;
        Ok(json!({
            "target":"segmentation_objects",
            "filter":self.clear_object_filter(&params)?["result"].clone(),
        }))
    }

    fn set_active_channel_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_active_channel(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    fn set_visible_channels_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_visible_channels(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    fn get_channel_contrast_global(&self, params: &Value) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        let index = if params.as_object().is_some_and(|object| !object.is_empty()) {
            resolve_channel(&viewport.channels, channel_selector_from_params(params)?)?
        } else {
            viewport.active_channel
        };
        Ok(json!({
            "mode": "single",
            "contrast": contrast_json(
                &viewport.channels[index],
                index,
                dataset.descriptor.abs_max.max(1.0),
            ),
        }))
    }

    fn set_channel_contrast_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_contrast(&params)?;
        Ok(json!({"mode": "single", "contrast": response["result"]}))
    }

    fn set_channel_color_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_color(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    fn set_channel_note_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let selector = channel_selector_from_params(params)?.clone();
        let note = params
            .get("note")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("set_channel_note requires note"))?
            .to_string();
        let dataset = self.dataset_mut()?;
        let index = resolve_channel(&dataset.workspace.active().state.channels, &selector)?;
        let changed = dataset
            .workspace
            .active()
            .state
            .channels
            .get(index)
            .is_some_and(|channel| channel.note != note);
        for slot in dataset.workspace.viewports_mut() {
            if let Some(channel) = slot.state.channels.get_mut(index) {
                channel.note.clone_from(&note);
            }
        }
        let channel = full_channel_json(
            &dataset.workspace.active().state.channels[index],
            index == dataset.workspace.active().state.active_channel,
        );
        Ok(json!({"changed": changed, "channel": channel}))
    }

    fn get_channel_transform(&self, params: &Value) -> Result<Value, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        let index = resolve_channel(&viewport.channels, channel_selector_from_params(params)?)?;
        Ok(channel_transform_json(&viewport.channels[index], index))
    }

    fn set_channel_transform(&mut self, params: &Value) -> Result<Value, ControlError> {
        let selector = channel_selector_from_params(params)?.clone();
        let offset = optional_finite_pair(params, "offset_world")?;
        let scale = optional_finite_pair(params, "scale")?;
        if let Some([x, y]) = scale
            && (!(0.01..=100.0).contains(&x) || !(0.01..=100.0).contains(&y))
        {
            return Err(invalid("scale values must be between 0.01 and 100"));
        }
        let rotation = match params.get("rotation_rad") {
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| invalid("rotation_rad must be a finite number"))?
                    as f32,
            ),
            None => None,
        };
        let dataset = self.dataset_mut()?;
        let index = resolve_channel(&dataset.workspace.active().state.channels, &selector)?;
        let before = dataset.workspace.active().state.channels[index].clone();
        for slot in dataset.workspace.viewports_mut() {
            let channel = &mut slot.state.channels[index];
            if let Some(offset) = offset {
                channel.offset_world = offset;
            }
            if let Some(scale) = scale {
                channel.scale = scale;
            }
            if let Some(rotation) = rotation {
                channel.rotation_rad = rotation;
            }
        }
        let channel = &dataset.workspace.active().state.channels[index];
        let changed = before.offset_world != channel.offset_world
            || before.scale != channel.scale
            || before.rotation_rad != channel.rotation_rad;
        Ok(json!({
            "changed": changed,
            "transform": channel_transform_json(channel, index),
        }))
    }

    fn reset_channel_transform(&mut self, params: &Value) -> Result<Value, ControlError> {
        let mut reset = params.clone();
        let object = reset
            .as_object_mut()
            .ok_or_else(|| invalid("params must be an object"))?;
        object.insert("offset_world".to_string(), json!([0.0, 0.0]));
        object.insert("scale".to_string(), json!([1.0, 1.0]));
        object.insert("rotation_rad".to_string(), json!(0.0));
        self.set_channel_transform(&reset)
    }

    fn set_channel_order_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_order(&params)?;
        Ok(response["result"].clone())
    }

    fn channel_presentation_global(&self) -> Result<Value, ControlError> {
        Ok(channel_presentation_json(
            &self.dataset()?.workspace.active().state,
        ))
    }

    fn set_channel_presentation_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_presentation(&params)?;
        Ok(response["result"].clone())
    }

    fn channel_groups_global(&self) -> Result<Value, ControlError> {
        Ok(json!({
            "mode": "single",
            "groups": channel_groups_json(&self.dataset()?.workspace.active().state),
        }))
    }

    fn set_channel_group_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_group(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    fn get_camera_global(&self) -> Result<Value, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        Ok(json!({"mode": "single", "camera": control_camera_json(viewport)}))
    }

    fn set_camera_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_camera(&params)?;
        Ok(json!({"mode": "single", "camera": response["result"]}))
    }

    fn zoom_camera_global(&mut self, params: &Value, zoom_in: bool) -> Result<Value, ControlError> {
        let raw_factor = params.get("factor").and_then(Value::as_f64).unwrap_or(1.5);
        let factor = if zoom_in {
            raw_factor
        } else if raw_factor > 0.0 {
            1.0 / raw_factor
        } else {
            raw_factor
        };
        if !factor.is_finite() || factor <= 0.0 {
            return Err(invalid("zoom factor must be finite and > 0"));
        }
        let current = self.dataset()?.workspace.active().state.zoom;
        self.set_camera_global(&json!({"zoom": current as f64 * factor}))
    }

    fn fit_camera_global(&mut self) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(&json!({}))?;
        let response = self.fit_viewport(&params)?;
        Ok(json!({"mode": "single", "camera": response["result"]}))
    }

    fn get_plane_global(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        Ok(json!({
            "mode": "single",
            "plane": control_plane_json(
                &dataset.workspace.active().state,
                dataset.plane_extents,
                dataset.orthogonal_planes,
            ),
        }))
    }

    fn set_plane_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_plane(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    fn step_plane_global(&mut self, params: &Value, forward: bool) -> Result<Value, ControlError> {
        let step = params.get("step").and_then(Value::as_u64).unwrap_or(1);
        let wrap = params.get("wrap").and_then(Value::as_bool).unwrap_or(false);
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        let current = current_plane_slice(viewport);
        let extent = dataset.plane_extents[plane_mode_index(&viewport.plane_mode)].max(1);
        let last = extent.saturating_sub(1);
        let next = if wrap {
            let offset = step % extent;
            if forward {
                (current + offset) % extent
            } else {
                (current + extent - offset) % extent
            }
        } else if forward {
            current.saturating_add(step).min(last)
        } else {
            current.saturating_sub(step)
        };
        self.set_plane_global(&json!({"slice": next}))
    }

    fn plane_operation_availability(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let plane = control_plane_json(
            &dataset.workspace.active().state,
            dataset.plane_extents,
            dataset.orthogonal_planes,
        );
        let xy = plane["mode"] == "xy";
        let operation = |requires_xy: bool| {
            json!({
                "available": !requires_xy || xy,
                "reason": (requires_xy && !xy).then_some("operation requires the XY view plane"),
            })
        };
        Ok(json!({
            "plane": plane,
            "operations": {
                "measurements": operation(true),
                "memory_pin": operation(true),
                "channel_max": operation(true),
                "threshold_preview": operation(true),
                "object_selection": operation(false),
            }
        }))
    }

    fn get_smooth_pixels_global(&self) -> Result<Value, ControlError> {
        Ok(json!({
            "mode": "single",
            "smooth_pixels": {"smooth": self.dataset()?.workspace.active().state.smooth_pixels},
        }))
    }

    fn set_smooth_pixels_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let smooth = params
            .get("smooth")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_smooth_pixels requires smooth"))?;
        let params = self.active_scoped_params(&json!({"smooth_pixels": smooth}))?;
        let response = self.set_rendering(&params)?;
        Ok(json!({
            "mode": "single",
            "result": {
                "changed": response["result"]["changed"],
                "smooth_pixels": {"smooth": smooth},
            }
        }))
    }

    fn get_scale_bar_global(&self) -> Result<Value, ControlError> {
        Ok(json!({
            "visible":self.dataset()?.workspace.active().state.show_scale_bar,
            "supported":true,
        }))
    }

    fn set_scale_bar_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible must be a boolean"))?;
        let params = self.active_scoped_params(&json!({"show_scale_bar":visible}))?;
        self.set_rendering(&params)?;
        Ok(json!({"visible":visible,"supported":true}))
    }

    fn threshold_levels(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let levels = dataset
            .descriptor
            .levels
            .iter()
            .map(|level| {
                let width = level.shape.get(dataset.descriptor.dims.x).copied();
                let height = level.shape.get(dataset.descriptor.dims.y).copied();
                let pixel_count = width.zip(height).and_then(|(width, height)| {
                    width.checked_mul(height)
                });
                json!({
                    "index":level.index,
                    "downsample":level.downsample,
                    "width":width,
                    "height":height,
                    "pixel_count":pixel_count,
                    "interactive":pixel_count.is_some_and(|pixels| pixels <= THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS),
                })
            })
            .collect::<Vec<_>>();
        let default_full_level = levels.iter().find_map(|level| {
            level
                .get("interactive")
                .and_then(Value::as_bool)
                .filter(|interactive| *interactive)
                .and_then(|_| level.get("index"))
                .and_then(Value::as_u64)
        });
        Ok(json!({
            "max_interactive_pixels":THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS,
            "default_full_level":default_full_level,
            "levels":levels,
        }))
    }

    fn get_panels(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        Ok(json!({
            "mode": "single",
            "panels": {
                "left": dataset.show_left_panel,
                "right": dataset.show_right_panel,
            },
        }))
    }

    fn set_panels(&mut self, params: &Value) -> Result<Value, ControlError> {
        let left = params
            .get("left")
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| invalid("left must be a boolean"))
            })
            .transpose()?;
        let right = params
            .get("right")
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| invalid("right must be a boolean"))
            })
            .transpose()?;
        if left.is_none() && right.is_none() {
            return Err(invalid("set_side_panels requires left and/or right"));
        }
        let dataset = self.dataset_mut()?;
        let before_left = dataset.show_left_panel;
        let before_right = dataset.show_right_panel;
        if let Some(left) = left {
            dataset.show_left_panel = left;
        }
        if let Some(right) = right {
            dataset.show_right_panel = right;
        }
        let changed =
            before_left != dataset.show_left_panel || before_right != dataset.show_right_panel;
        if changed {
            if before_left != dataset.show_left_panel {
                let delta = if dataset.show_left_panel {
                    -DEFAULT_LEFT_PANEL_WIDTH
                } else {
                    DEFAULT_LEFT_PANEL_WIDTH
                };
                dataset.logical_workspace_size[0] =
                    (dataset.logical_workspace_size[0] + delta).max(1.0);
            }
            if before_right != dataset.show_right_panel {
                let delta = if dataset.show_right_panel {
                    -DEFAULT_RIGHT_PANEL_WIDTH
                } else {
                    DEFAULT_RIGHT_PANEL_WIDTH
                };
                dataset.logical_workspace_size[0] =
                    (dataset.logical_workspace_size[0] + delta).max(1.0);
            }
            update_logical_geometry(dataset);
        }
        Ok(json!({
            "mode": "single",
            "result": {
                "changed": changed,
                "panels": {
                    "left": dataset.show_left_panel,
                    "right": dataset.show_right_panel,
                },
            },
        }))
    }

    fn set_right_tab(&mut self, params: &Value) -> Result<Value, ControlError> {
        let tab = params
            .get("tab")
            .or_else(|| params.get("right_tab"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|tab| !tab.is_empty())
            .ok_or_else(|| invalid("set_right_tab requires tab"))?;
        if !matches!(
            tab,
            "properties" | "views" | "analysis" | "measurements" | "memory" | "roi_selector"
        ) {
            return Err(invalid(
                "unknown right tab; expected properties, views, analysis, measurements, memory, or roi_selector",
            ));
        }
        self.dataset_mut()?.right_tab = tab.to_string();
        Ok(json!({"mode":"single","tab":{"right_tab":tab}}))
    }

    fn workspace_snapshot(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let workspace = &dataset.workspace;
        let active = workspace.active_id();
        Ok(json!({
            "revision": workspace.revision(),
            "layout": workspace.layout().as_str(),
            "ratio": workspace.split_ratio(),
            "active_viewport_id": active.as_str(),
            "max_viewports": crate::viewports::MAX_VIEWPORTS,
            "shared_resources": dataset.shared_resources,
            "object_resource": dataset.object_resource.as_ref().map_or_else(
                || self.object_resource_state(),
                |resource| resource.descriptor_json(self.installed_object_resource_generation),
            ),
            "labels": self.labels_snapshot()?,
            "masks": dataset.masks.projection_json(),
            "object_selection": dataset.object_selection.projection_json(),
            "panels": {
                "left": dataset.show_left_panel,
                "right": dataset.show_right_panel,
            },
            "ui":{"right_tab":dataset.right_tab},
            "channel_metadata": workspace.active().state.channels.iter().map(|channel| json!({
                "index": channel.index,
                "name": channel.name,
                "note": channel.note,
            })).collect::<Vec<_>>(),
            "channel_transforms": workspace.active().state.channels.iter().enumerate().map(
                |(index, channel)| channel_transform_json(channel, index)
            ).collect::<Vec<_>>(),
            "channel_presentation": channel_presentation_json(&workspace.active().state),
            "performance": dataset.performance,
            "links": links_json(workspace.links()),
            "viewports": workspace.viewports().iter().map(|slot| viewport_json(slot, slot.id == *active)).collect::<Vec<_>>(),
        }))
    }

    fn layout_snapshot(&self) -> Result<Value, ControlError> {
        let workspace = &self.dataset()?.workspace;
        Ok(json!({
            "revision": workspace.revision(),
            "layout": workspace.layout().as_str(),
            "ratio": workspace.split_ratio(),
            "viewport_ids": workspace.viewports().iter().map(|v| v.id.as_str()).collect::<Vec<_>>(),
        }))
    }

    fn set_layout(&mut self, params: &Value) -> Result<Value, ControlError> {
        let value = params
            .get("layout")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("layout is required"))?;
        let layout = ViewportLayout::parse(value)
            .ok_or_else(|| invalid("layout must be 'single', 'horizontal', or 'vertical'"))?;
        if let Some(requested) = params
            .get("viewports")
            .or_else(|| params.get("viewport_ids"))
        {
            validate_viewport_order(&self.dataset()?.workspace, requested)?;
        }
        let result = {
            let dataset = self.dataset_mut()?;
            let mut changed = dataset
                .workspace
                .set_layout(layout)
                .map_err(|e| invalid(e.to_string()))?;
            if let Some(ratio) = params.get("ratio").and_then(Value::as_f64) {
                changed |= dataset
                    .workspace
                    .set_split_ratio(ratio as f32)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            update_logical_geometry(dataset);
            json!({"changed": changed, "layout": layout.as_str(), "ratio": dataset.workspace.split_ratio()})
        };
        self.measured_viewports.clear();
        Ok(result)
    }

    fn swap_viewports(&mut self) -> Result<Value, ControlError> {
        {
            let dataset = self.dataset_mut()?;
            if !dataset.workspace.swap_order() {
                return Err(invalid("swapping requires exactly two viewports"));
            }
            update_logical_geometry(dataset);
        }
        self.measured_viewports.clear();
        Ok(json!({"changed": true, "workspace": self.workspace_snapshot()?}))
    }

    fn viewport_snapshot_for(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_json(slot, id == *workspace.active_id()))
    }

    fn create_viewport(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = {
            let dataset = self.dataset_mut()?;
            let workspace = &mut dataset.workspace;
            let source = params
                .get("source_viewport_id")
                .or_else(|| params.get("viewport_id"))
                .and_then(Value::as_str)
                .map(ViewportId::new)
                .transpose()
                .map_err(|e| invalid(e.to_string()))?
                .unwrap_or_else(|| workspace.active_id().clone());
            let layout = match params.get("layout").and_then(Value::as_str) {
                Some(value) => match ViewportLayout::parse(value) {
                    Some(layout @ (ViewportLayout::Horizontal | ViewportLayout::Vertical)) => {
                        layout
                    }
                    Some(ViewportLayout::Single) => {
                        return Err(invalid(
                            "creating a second viewport requires a split layout",
                        ));
                    }
                    None => return Err(invalid("layout must be 'horizontal' or 'vertical'")),
                },
                None => ViewportLayout::Horizontal,
            };
            let title = params
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_string);
            let activate = params
                .get("activate")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let previous = workspace.active_id().clone();
            let id = workspace
                .clone_viewport(&source, title, layout)
                .map_err(|e| invalid(e.to_string()))?;
            if let Some(ratio) = params.get("ratio").and_then(Value::as_f64) {
                workspace
                    .set_split_ratio(ratio as f32)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            if !activate {
                workspace
                    .set_active(&previous)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            update_logical_geometry(dataset);
            id
        };
        self.measured_viewports.clear();
        Ok(
            json!({"created": true, "viewport_id": id.as_str(), "workspace": self.workspace_snapshot()?}),
        )
    }

    fn rename_viewport(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("title is required"))?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let changed = workspace
            .rename(&id, title.to_string())
            .map_err(|e| invalid(e.to_string()))?;
        if changed {
            let _ = workspace.bump_presentation_revision(&id);
        }
        let revision = workspace
            .get(&id)
            .map(|slot| slot.presentation_revision)
            .unwrap_or(0);
        Ok(
            json!({"changed": changed, "viewport_id": id.as_str(), "title": title.trim(), "presentation_revision": revision}),
        )
    }

    fn remove_viewport(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        {
            let dataset = self.dataset_mut()?;
            dataset
                .workspace
                .remove(&id)
                .map_err(|e| invalid(e.to_string()))?;
            update_logical_geometry(dataset);
        }
        self.measured_viewports.clear();
        Ok(
            json!({"removed": true, "viewport_id": id.as_str(), "workspace": self.workspace_snapshot()?}),
        )
    }

    fn set_active_viewport(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let changed = workspace
            .set_active(&id)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(
            json!({"changed": changed, "active_viewport_id": id.as_str(), "viewport": self.viewport_snapshot_for(params)?}),
        )
    }

    fn links_snapshot(&self) -> Result<Value, ControlError> {
        let workspace = &self.dataset()?.workspace;
        Ok(
            json!({"links": links_json(workspace.links()), "viewport_ids": workspace.viewports().iter().map(|v| v.id.as_str()).collect::<Vec<_>>() }),
        )
    }

    fn link_groups_snapshot(&self) -> Result<Value, ControlError> {
        Ok(json!({"link_groups": [self.link_group()?]}))
    }

    fn link_group(&self) -> Result<Value, ControlError> {
        let workspace = &self.dataset()?.workspace;
        let links = workspace.links();
        let mut fields = Vec::new();
        if links.camera {
            fields.push("camera");
        }
        if links.plane {
            fields.push("plane");
        }
        fields.push("selection");
        Ok(
            json!({"link_group_id": "comparison-navigation", "viewport_ids": workspace.viewports().iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), "fields": fields}),
        )
    }

    fn set_links(&mut self, params: &Value, kind: LinkRequestKind) -> Result<Value, ControlError> {
        if params
            .get("link_group_id")
            .is_some_and(|v| v.as_str() != Some("comparison-navigation"))
        {
            return Err(invalid("link_group_id must be 'comparison-navigation'"));
        }
        let workspace = &self.dataset()?.workspace;
        if kind == LinkRequestKind::Create
            && params
                .get("viewports")
                .or_else(|| params.get("viewport_ids"))
                .is_none()
        {
            return Err(invalid("viewports must identify both workspace viewports"));
        }
        if let Some(requested) = params
            .get("viewports")
            .or_else(|| params.get("viewport_ids"))
        {
            validate_viewport_set(workspace, requested)?;
        }
        let current = workspace.links();
        let links = if kind != LinkRequestKind::Direct {
            let fields = params
                .get("fields")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid("fields must be an array containing camera, plane, and/or selection")
                })?;
            let fields = fields
                .iter()
                .map(Value::as_str)
                .collect::<Option<HashSet<_>>>()
                .ok_or_else(|| invalid("fields must contain only strings"))?;
            for field in &fields {
                if !matches!(*field, "camera" | "plane" | "selection") {
                    return Err(invalid(format!("unknown viewport link field '{field}'")));
                }
            }
            ViewportLinks {
                camera: fields.contains("camera"),
                plane: fields.contains("plane"),
                selection: true,
            }
        } else {
            ViewportLinks {
                camera: params
                    .get("camera")
                    .and_then(Value::as_bool)
                    .unwrap_or(current.camera),
                plane: params
                    .get("plane")
                    .and_then(Value::as_bool)
                    .unwrap_or(current.plane),
                selection: params
                    .get("selection")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            }
        };
        if !links.selection {
            return Err(invalid(
                "selection is document-shared in the two-viewport milestone",
            ));
        }
        let mut response = self.apply_links(links)?;
        if kind != LinkRequestKind::Direct {
            response["link_group"] = self.link_group()?;
        }
        Ok(response)
    }

    fn apply_links(&mut self, links: ViewportLinks) -> Result<Value, ControlError> {
        let workspace = &mut self.dataset_mut()?.workspace;
        let before_revisions = workspace
            .viewports()
            .iter()
            .map(|viewport| (viewport.id.clone(), viewport.navigation_revision))
            .collect::<Vec<_>>();
        let active_id = workspace.active_id().clone();
        let active_state = workspace.active().state.clone();
        let other_ids = workspace
            .viewports()
            .iter()
            .filter(|viewport| viewport.id != active_id)
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        for id in other_ids {
            let mut navigation_changed = false;
            if let Some(viewport) = workspace.get_mut(&id) {
                if links.camera
                    && (viewport.state.center != active_state.center
                        || viewport.state.zoom != active_state.zoom)
                {
                    viewport.state.center = active_state.center;
                    viewport.state.zoom = active_state.zoom;
                    navigation_changed = true;
                }
                if links.plane
                    && (viewport.state.plane_mode != active_state.plane_mode
                        || current_plane_slice(&viewport.state)
                            != current_plane_slice(&active_state))
                {
                    viewport.state.plane_mode = active_state.plane_mode.clone();
                    viewport.state.plane_slices = active_state.plane_slices;
                    navigation_changed = true;
                }
            }
            if navigation_changed {
                let _ = workspace.bump_navigation_revision(&id);
            }
        }
        let changed = workspace.set_links(links);
        let affected_viewport_ids = workspace
            .viewports()
            .iter()
            .filter(|viewport| {
                before_revisions
                    .iter()
                    .find(|(id, _)| *id == viewport.id)
                    .is_none_or(|(_, revision)| *revision != viewport.navigation_revision)
            })
            .map(|viewport| viewport.id.as_str().to_string())
            .collect::<Vec<_>>();
        Ok(json!({
            "changed": changed,
            "links": links_json(links),
            "affected_viewport_ids": affected_viewport_ids,
            "workspace": self.workspace_snapshot()?,
        }))
    }

    fn remove_links(&mut self, params: &Value) -> Result<Value, ControlError> {
        if params
            .get("link_group_id")
            .is_some_and(|v| v.as_str() != Some("comparison-navigation"))
        {
            return Err(invalid("link_group_id must be 'comparison-navigation'"));
        }
        let links = ViewportLinks {
            camera: false,
            plane: false,
            selection: true,
        };
        let mut response = self.apply_links(links)?;
        response["removed"] = Value::Bool(true);
        response["link_group"] = self.link_group()?;
        Ok(response)
    }

    fn get_camera(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            control_camera_json(&slot.state),
            vec![id.clone()],
            false,
        ))
    }

    fn set_camera(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let links = workspace.links();
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let before = target.state.clone();
        let mut state = before.clone();
        if let Some(center) = params
            .get("center_world_lvl0")
            .and_then(Value::as_array)
            .filter(|v| v.len() == 2)
        {
            state.center = [
                center[0]
                    .as_f64()
                    .ok_or_else(|| invalid("camera center must be numeric"))?
                    as f32,
                center[1]
                    .as_f64()
                    .ok_or_else(|| invalid("camera center must be numeric"))?
                    as f32,
            ];
        }
        if let Some(x) = params.get("center_x").and_then(Value::as_f64) {
            if !x.is_finite() {
                return Err(invalid("camera center_x must be finite"));
            }
            state.center[0] = x as f32;
        }
        if let Some(y) = params.get("center_y").and_then(Value::as_f64) {
            if !y.is_finite() {
                return Err(invalid("camera center_y must be finite"));
            }
            state.center[1] = y as f32;
        }
        if let Some(zoom) = params
            .get("zoom_screen_per_lvl0_px")
            .or_else(|| params.get("zoom"))
            .and_then(Value::as_f64)
        {
            if !zoom.is_finite() || zoom <= 0.0 {
                return Err(invalid("zoom must be finite and greater than zero"));
            }
            state.zoom = (zoom as f32).clamp(0.000_01, 5000.0);
        }
        if !state.center.iter().all(|value| value.is_finite()) {
            return Err(invalid("camera center must be finite"));
        }
        target.state = state.clone();
        let changed = camera_changed(&before, &state);
        let _ = workspace.bump_navigation_revision(&id);
        if links.camera && changed {
            propagate_camera(workspace, &id, &state);
        }
        let affected = if links.camera && changed {
            workspace
                .viewports()
                .iter()
                .map(|slot| slot.id.clone())
                .collect()
        } else {
            vec![id.clone()]
        };
        let active_after = workspace.active().state.clone();
        Ok(viewport_response(
            workspace,
            &id,
            control_camera_json(&state),
            affected,
            camera_changed(&active_before, &active_after),
        ))
    }

    fn fit_viewport(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let dataset = self.dataset_mut()?;
        let links = dataset.workspace.links();
        let active_before = dataset.workspace.active().state.clone();
        let target = dataset
            .workspace
            .get_mut(&id)
            .ok_or_else(|| not_found(&id))?;
        let before = target.state.clone();
        fit_camera(&mut target.state, dataset.world_size);
        let state = target.state.clone();
        let changed = camera_changed(&before, &state);
        let _ = dataset.workspace.bump_navigation_revision(&id);
        if links.camera && changed {
            propagate_camera(&mut dataset.workspace, &id, &state);
        }
        let affected = if links.camera && changed {
            dataset
                .workspace
                .viewports()
                .iter()
                .map(|slot| slot.id.clone())
                .collect()
        } else {
            vec![id.clone()]
        };
        let active_after = dataset.workspace.active().state.clone();
        Ok(viewport_response(
            &dataset.workspace,
            &id,
            control_camera_json(&state),
            affected,
            camera_changed(&active_before, &active_after),
        ))
    }

    fn get_plane(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let dataset = self.dataset()?;
        let workspace = &dataset.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            control_plane_json(
                &slot.state,
                dataset.plane_extents,
                dataset.orthogonal_planes,
            ),
            vec![id.clone()],
            false,
        ))
    }

    fn set_plane(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let requested_mode = params
            .get("mode")
            .and_then(Value::as_str)
            .map(normalize_plane_mode)
            .transpose()?;
        let requested_slice = params.get("slice").and_then(Value::as_u64);
        let dataset = self.dataset_mut()?;
        if requested_mode.is_some_and(|mode| mode != "xy") && !dataset.orthogonal_planes {
            return Err(invalid(format!(
                "{} view is not available for this dataset",
                requested_mode
                    .expect("checked as some")
                    .to_ascii_uppercase()
            )));
        }
        let plane_extents = dataset.plane_extents;
        let orthogonal_planes = dataset.orthogonal_planes;
        let workspace = &mut dataset.workspace;
        let links = workspace.links();
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let before = target.state.clone();
        if let Some(mode) = requested_mode {
            target.state.plane_mode = mode.to_string();
        }
        if let Some(slice) = requested_slice {
            set_current_plane_slice(&mut target.state, slice, plane_extents);
        } else {
            clamp_current_plane_slice(&mut target.state, plane_extents);
        }
        let state = target.state.clone();
        let changed = plane_changed(&before, &state);
        let _ = workspace.bump_navigation_revision(&id);
        if links.plane && changed {
            for slot in workspace
                .viewports()
                .iter()
                .filter(|slot| slot.id != id)
                .map(|slot| slot.id.clone())
                .collect::<Vec<_>>()
            {
                if let Some(other) = workspace.get_mut(&slot) {
                    other.state.plane_mode = state.plane_mode.clone();
                    other.state.plane_slices = state.plane_slices;
                }
                let _ = workspace.bump_navigation_revision(&slot);
            }
        }
        let affected = if links.plane && changed {
            workspace
                .viewports()
                .iter()
                .map(|slot| slot.id.clone())
                .collect()
        } else {
            vec![id.clone()]
        };
        let active_after = workspace.active().state.clone();
        Ok(viewport_response(
            workspace,
            &id,
            json!({
                "changed": changed,
                "plane": control_plane_json(&state, plane_extents, orthogonal_planes),
            }),
            affected,
            plane_changed(&active_before, &active_after),
        ))
    }

    fn get_viewport_channels(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            Value::Array(full_channels_json(&slot.state)),
            vec![id.clone()],
            false,
        ))
    }

    fn set_visible_channels(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("channels must be an array"))?
            .clone();
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let indices = resolve_channels(&target.state.channels, &selectors)?;
        let mode = params.get("mode").and_then(Value::as_str).unwrap_or("only");
        if !matches!(mode, "only" | "show" | "hide" | "add" | "remove") {
            return Err(invalid(format!("unknown visibility mode '{mode}'")));
        }
        for channel in &mut target.state.channels {
            channel.visible = match mode {
                "show" | "add" => channel.visible || indices.contains(&channel.index),
                "hide" | "remove" => channel.visible && !indices.contains(&channel.index),
                "only" => indices.contains(&channel.index),
                _ => unreachable!("visibility mode validated above"),
            };
        }
        if let Some(first) = selectors.first() {
            target.state.active_channel = resolve_channel(&target.state.channels, first)?;
        }
        let visible_channels = visible_channels_json(&target.state);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "mode": canonical_visibility_mode(mode), "visible_channels": visible_channels}),
            vec![id.clone()],
            active_changed,
        ))
    }

    fn set_active_channel(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selector = channel_selector_from_params(params)?.clone();
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        target.state.active_channel = resolve_channel(&target.state.channels, &selector)?;
        let active_channel = target.state.active_channel;
        let active_channel = active_channel_json(&target.state.channels[active_channel]);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "active_channel": active_channel}),
            vec![id.clone()],
            active_changed,
        ))
    }

    fn set_channel_color(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selector = channel_selector_from_params(params)?.clone();
        let color = params
            .get("color_rgb")
            .or_else(|| params.get("color"))
            .and_then(Value::as_array)
            .filter(|v| v.len() == 3)
            .ok_or_else(|| invalid("color_rgb must contain three integers"))?;
        let rgb = [to_u8(&color[0])?, to_u8(&color[1])?, to_u8(&color[2])?];
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let index = resolve_channel(&target.state.channels, &selector)?;
        let changed = target.state.channels[index].color_rgb != rgb;
        target.state.channels[index].color_rgb = rgb;
        let channel = full_channel_json(
            &target.state.channels[index],
            index == target.state.active_channel,
        );
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": changed, "channel": channel}),
            vec![id.clone()],
            active_changed,
        ))
    }

    fn set_channel_contrast(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selector = channel_selector_from_params(params)?.clone();
        let min = params
            .get("min")
            .or_else(|| params.get("lo"))
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("min is required"))? as f32;
        let max = params
            .get("max")
            .or_else(|| params.get("hi"))
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("max is required"))? as f32;
        if !min.is_finite() || !max.is_finite() || max <= min {
            return Err(invalid("contrast max must be greater than min"));
        }
        let dataset = self.dataset_mut()?;
        let abs_max = dataset.descriptor.abs_max.max(1.0);
        let workspace = &mut dataset.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let index = resolve_channel(&target.state.channels, &selector)?;
        target.state.channels[index].window = Some((min, max));
        let channel = &target.state.channels[index];
        let result = json!({
            "index": index,
            "name": channel.name,
            "min": min,
            "max": max,
            "abs_max": abs_max,
        });
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            result,
            vec![id.clone()],
            active_changed,
        ))
    }

    fn set_channel_order(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let requested_sort = params
            .get("sort")
            .map(|value| {
                value
                    .as_str()
                    .and_then(canonical_channel_sort)
                    .ok_or_else(|| {
                        invalid(format!(
                            "unknown channel sort mode '{}'",
                            value.as_str().unwrap_or_default()
                        ))
                    })
            })
            .transpose()?;
        let selectors = params.get("channels").and_then(Value::as_array);
        if requested_sort.is_none() && selectors.is_none() {
            return Err(invalid("set_channel_order requires channels or sort"));
        }
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let response = if let Some(sort) = requested_sort {
            target.state.channel_sort = sort.to_string();
            json!({
                "changed": true,
                "sort": sort,
                "order": channel_order_json(&target.state),
            })
        } else {
            let selectors = selectors.expect("validated above");
            let indices = resolve_channel_list_ordered(&target.state.channels, selectors)?;
            let mode = params
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("listed_first");
            match mode {
                "listed_first" => {
                    let pinned = indices.iter().copied().collect::<HashSet<_>>();
                    let mut next = indices;
                    next.extend(
                        target
                            .state
                            .channel_order
                            .iter()
                            .copied()
                            .filter(|index| !pinned.contains(index)),
                    );
                    for index in 0..target.state.channels.len() {
                        if !next.contains(&index) {
                            next.push(index);
                        }
                    }
                    target.state.channel_order = next;
                    target.state.channel_sort = "manual".to_string();
                }
                "exact" => {
                    if indices.len() != target.state.channels.len() {
                        return Err(invalid(
                            "exact channel order must include every channel exactly once",
                        ));
                    }
                    target.state.channel_order = indices;
                    target.state.channel_sort = "manual".to_string();
                }
                other => return Err(invalid(format!("unknown channel order mode '{other}'"))),
            }
            json!({
                "changed": true,
                "mode": mode,
                "sort": target.state.channel_sort,
                "order": channel_order_json(&target.state),
            })
        };
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            response,
            vec![id.clone()],
            active_changed,
        ))
    }

    fn channel_groups(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let target = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            channel_groups_json(&target.state),
            vec![id.clone()],
            false,
        ))
    }

    fn set_channel_group(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        if params.get("replace_all").and_then(Value::as_bool) == Some(true) {
            let groups = params
                .get("groups")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("replace_all requires groups"))?;
            let workspace = &mut self.dataset_mut()?.workspace;
            let active_before = workspace.active().state.clone();
            let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
            let replacement = parse_channel_groups_snapshot(groups, &target.state.channels)?;
            let changed = replacement != target.state.channel_groups;
            target.state.channel_groups = replacement;
            let groups = channel_groups_json(&target.state);
            let _ = workspace.bump_presentation_revision(&id);
            let active_changed = presentation_changed(&active_before, &workspace.active().state);
            return Ok(viewport_response(
                workspace,
                &id,
                json!({"changed": changed, "group_id": Value::Null, "groups": groups}),
                vec![id.clone()],
                active_changed,
            ));
        }
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("set_channel_group requires channels"))?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let indices = resolve_channel_list_ordered(&target.state.channels, selectors)?;
        if indices.is_empty() {
            return Err(invalid("no channels resolved"));
        }
        let requested_group_id = params.get("group_id").and_then(Value::as_u64);
        let requested_name = params
            .get("group")
            .or_else(|| params.get("name"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty());
        let color = optional_rgb(params, "color_rgb")?;
        let group_id = ensure_model_channel_group(
            &mut target.state.channel_groups,
            requested_group_id,
            requested_name,
            color,
        );
        if params
            .get("replace_group_members")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            target
                .state
                .channel_groups
                .channel_members
                .retain(|_, member| member.group_id != group_id);
        }
        let inherit_color = params
            .get("inherit_color")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        for index in indices {
            let name = target.state.channels[index].name.clone();
            target.state.channel_groups.channel_members.insert(
                name,
                ProjectChannelGroupMember {
                    group_id,
                    inherit_color,
                },
            );
        }
        let groups = channel_groups_json(&target.state);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "group_id": group_id, "groups": groups}),
            vec![id.clone()],
            active_changed,
        ))
    }

    fn get_object_style(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let target = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            object_style_json(&target.state.objects),
            vec![id.clone()],
            false,
        ))
    }

    fn set_object_style(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let changed = apply_object_style_patch(&mut target.state.objects, params)?;
        let style = object_style_json(&target.state.objects);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": changed, "style": style}),
            vec![id.clone()],
            active_changed,
        ))
    }

    fn set_object_legend(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        apply_object_legend_patch(&mut target.state.objects, params)?;
        let style = object_style_json(&target.state.objects);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "style": style}),
            vec![id.clone()],
            active_changed,
        ))
    }

    fn set_channel_presentation(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let search = params
            .get("search")
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("search must be a string"))
            })
            .transpose()?;
        let sort = params
            .get("sort")
            .map(|value| {
                value
                    .as_str()
                    .and_then(canonical_channel_sort)
                    .map(str::to_string)
                    .ok_or_else(|| invalid("unknown channel sort mode"))
            })
            .transpose()?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        if let Some(search) = search {
            target.state.channel_search = search;
        }
        if let Some(sort) = sort {
            target.state.channel_sort = sort;
        }
        let result = channel_presentation_json(&target.state);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            result,
            vec![id.clone()],
            active_changed,
        ))
    }

    fn get_rendering(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            rendering_json(&slot.state),
            vec![id.clone()],
            false,
        ))
    }

    fn set_rendering(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let before = rendering_json(&target.state);
        let mut saw_field = false;
        let mut smooth_pixels = target.state.smooth_pixels;
        let mut show_scale_bar = target.state.show_scale_bar;
        let mut show_hud = target.state.show_hud;
        let mut show_tile_debug = target.state.show_tile_debug;
        set_rendering_bool(
            params,
            &["smooth_pixels", "smooth"],
            "smooth_pixels",
            &mut smooth_pixels,
            &mut saw_field,
        )?;
        set_rendering_bool(
            params,
            &["show_scale_bar"],
            "show_scale_bar",
            &mut show_scale_bar,
            &mut saw_field,
        )?;
        set_rendering_bool(
            params,
            &["show_hud"],
            "show_hud",
            &mut show_hud,
            &mut saw_field,
        )?;
        set_rendering_bool(
            params,
            &["show_tile_debug"],
            "show_tile_debug",
            &mut show_tile_debug,
            &mut saw_field,
        )?;
        if !saw_field {
            return Err(invalid(
                "provide smooth_pixels, show_scale_bar, show_hud, and/or show_tile_debug",
            ));
        }
        target.state.smooth_pixels = smooth_pixels;
        target.state.show_scale_bar = show_scale_bar;
        target.state.show_hud = show_hud;
        target.state.show_tile_debug = show_tile_debug;
        let result = rendering_json(&target.state);
        let changed = before != result;
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": changed, "rendering": result}),
            vec![id.clone()],
            active_changed,
        ))
    }
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

fn apply_renderer_viewport(state: &mut ViewportModel, value: &Value) -> Result<(), ControlError> {
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

fn apply_renderer_channel_transforms(
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

fn apply_renderer_channel_metadata(
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
            "screen_rect": [0.0, 0.0, viewport.logical_size[0], viewport.logical_size[1]],
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

fn object_filter_snapshot(state: &ViewportModel, total_count: usize) -> Value {
    let model = state
        .objects
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
    let visible_count = if state.object_filter_active {
        state.object_filter_indices.len()
    } else {
        total_count
    };
    json!({
        "revision": state.object_filter_revision,
        "active": state.object_filter_active,
        "mode": mode,
        "logic": logic,
        "total_count": total_count,
        "visible_count": visible_count,
        "hidden_count": total_count.saturating_sub(visible_count),
        "simple": {"logic": logic, "clauses": clauses},
        "query": {
            "text": query,
            "applied": mode == "query" && state.object_filter_active,
            "error": Value::Null,
        },
    })
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
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr")
    }

    #[test]
    fn comparison_commands_advance_without_a_renderer() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        let left = model.workspace_snapshot().unwrap()["active_viewport_id"]
            .as_str()
            .unwrap()
            .to_string();
        let created = model.dispatch("viewer.viewports.clone", &json!({"source_viewport_id": left, "layout":"horizontal", "ratio":0.5, "title":"Right"})).unwrap().unwrap().response;
        let right = created["viewport_id"].as_str().unwrap().to_string();
        model
            .dispatch(
                "viewer.viewports.channels.set_visible",
                &json!({"viewport_id":left,"channels":[0],"mode":"only"}),
            )
            .unwrap()
            .unwrap();
        model
            .dispatch(
                "viewer.viewports.channels.set_visible",
                &json!({"viewport_id":right,"channels":[1],"mode":"only"}),
            )
            .unwrap()
            .unwrap();
        let fitted = model
            .dispatch("viewer.viewports.camera.fit", &json!({"viewport_id":right}))
            .unwrap()
            .unwrap()
            .response;
        assert!(
            fitted["result"]["zoom_screen_per_lvl0_px"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        let workspace = model.workspace_snapshot().unwrap();
        assert_eq!(workspace["viewports"].as_array().unwrap().len(), 2);
        assert_eq!(workspace["layout"], "horizontal");
    }

    #[test]
    fn readiness_and_availability_queries_are_background_safe_in_every_mode() {
        let mut model = AppModel::project();
        for mode in [ModelMode::Project, ModelMode::Mosaic, ModelMode::Transition] {
            model.bootstrap_mode_from_renderer(mode);
            let loading = model
                .dispatch("app.get_loading_state", &json!({}))
                .expect("loading state is actor-owned")
                .unwrap()
                .response;
            assert_eq!(loading["mode"], mode.as_str());
            let availability = model
                .dispatch(
                    "app.get_method_availability",
                    &json!({"methods":["app.get_loading_state","viewer.camera.fit"]}),
                )
                .expect("availability is actor-owned")
                .unwrap()
                .response;
            assert_eq!(availability["mode"], mode.as_str());
            assert_eq!(availability["methods"].as_array().unwrap().len(), 2);
        }
    }

    #[test]
    fn concurrent_resource_operations_have_independent_readiness() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);

        let project_generation = model.begin_project_operation("Saving project");
        let (document_generation, label_generation, _) = model
            .begin_label_load(&json!({"name":"cells"}))
            .expect("label operation starts");
        let loading = model.loading_state();
        assert_eq!(loading["loading"]["busy"], true);
        assert_eq!(
            loading["loading"]["operations"]["project_io"]["phase"],
            "pending"
        );
        assert_eq!(
            loading["loading"]["operations"]["labels"]["phase"],
            "pending"
        );

        assert!(model.fail_label_load_for_generation(
            document_generation,
            label_generation,
            "label fixture failed",
        ));
        let loading = model.loading_state();
        assert_eq!(loading["loading"]["busy"], true);
        assert_eq!(loading["loading"]["resources_ready"], false);
        assert_eq!(loading["loading"]["status"], "Saving project");
        assert_eq!(
            loading["loading"]["operations"]["labels"]["phase"],
            "failed"
        );

        assert!(model.finish_project_operation_for_generation(project_generation));
        let loading = model.loading_state();
        assert_eq!(loading["loading"]["busy"], false);
        assert_eq!(loading["loading"]["resources_ready"], true);
    }

    #[test]
    fn settings_writes_cannot_commit_out_of_order() {
        let mut model = AppModel::project();
        model.bootstrap_settings(
            AppSettings::default(),
            Some(PathBuf::from("/tmp/odon-settings-ordering.json")),
            Vec::new(),
        );
        let SettingsMutationOutcome::Persist(first) = model
            .prepare_settings_set(&json!({"fast_object_rendering":false}))
            .unwrap()
        else {
            panic!("first settings change should require persistence")
        };
        let error = model
            .prepare_settings_set(&json!({"auto_contrast":{"enabled_on_open":false}}))
            .unwrap_err();
        assert_eq!(error.kind, ControlErrorKind::NotReady);
        assert!(
            model
                .install_settings_for_generation(first.generation, first.settings, first.response,)
                .is_some()
        );
        let SettingsMutationOutcome::Persist(second) = model
            .prepare_settings_set(&json!({"auto_contrast":{"enabled_on_open":false}}))
            .unwrap()
        else {
            panic!("second settings change should start after the first commits")
        };
        assert!(second.generation > first.generation);
    }

    #[test]
    fn viewport_filters_of_the_same_kind_have_independent_readiness() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        let (document_generation, resource_generation) =
            model.begin_object_resource_load("objects.geojson");
        assert!(model.install_object_resource_for_generation(
            document_generation,
            resource_generation,
            Arc::new(ControlObjectResource {
                source: PathBuf::from("objects.geojson"),
                downsample_factor: 1.0,
                features: Arc::new(Vec::new()),
                property_names: Arc::new(vec!["id".to_string()]),
                renderer_payload: None,
            }),
        ));
        let left = model.workspace_snapshot().unwrap()["active_viewport_id"]
            .as_str()
            .unwrap()
            .to_string();
        let right = model
            .dispatch(
                "viewer.viewports.clone",
                &json!({"source_viewport_id":left,"layout":"horizontal"}),
            )
            .unwrap()
            .unwrap()
            .response["viewport_id"]
            .as_str()
            .unwrap()
            .to_string();

        let left_work = model
            .begin_object_filter_evaluation(
                &json!({"viewport_id":left,"mode":"query","query":"id == 'a'"}),
            )
            .unwrap();
        let right_work = model
            .begin_object_filter_evaluation(
                &json!({"viewport_id":right,"mode":"query","query":"id == 'b'"}),
            )
            .unwrap();
        let loading = model.loading_state();
        assert_eq!(
            loading["loading"]["operations"][format!("object_filter:{left}")]["phase"],
            "pending"
        );
        assert_eq!(
            loading["loading"]["operations"][format!("object_filter:{right}")]["phase"],
            "pending"
        );

        assert!(
            model
                .install_object_filter_for_generation(
                    left_work.0,
                    left_work.1,
                    left_work.2,
                    &left_work.3,
                    left_work.4,
                    ControlObjectFilterResult {
                        model: left_work.6,
                        matching_indices: Arc::new(Vec::new()),
                        active: true,
                    },
                )
                .is_some()
        );
        let loading = model.loading_state();
        assert_eq!(loading["loading"]["busy"], true);
        assert_eq!(
            loading["loading"]["operations"][format!("object_filter:{left}")]["phase"],
            "ready"
        );
        assert_eq!(
            loading["loading"]["operations"][format!("object_filter:{right}")]["phase"],
            "pending"
        );
        assert!(model.fail_object_filter_for_generation(
            &right_work.3,
            right_work.2,
            "Right filter failed",
        ));
        assert_eq!(model.loading_state()["loading"]["busy"], false);
    }

    #[test]
    fn mask_io_readiness_is_scoped_and_cancelled_by_document_replacement() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);

        let (document_generation, mask_generation, import_generation, import_scope) =
            model.begin_mask_import_operation().unwrap();
        let (export_generation, export_scope) = model.begin_mask_export_operation().unwrap();
        assert!(model.finish_mask_io_for_generation(
            &export_scope,
            export_generation,
            "Mask export ready",
        ));
        let loading = model.loading_state();
        assert_eq!(loading["loading"]["busy"], true);
        assert_eq!(
            loading["loading"]["operations"][format!("mask_io:{import_scope}")]["phase"],
            "pending"
        );
        assert_eq!(
            loading["loading"]["operations"][format!("mask_io:{export_scope}")]["phase"],
            "ready"
        );

        model.begin_dataset_open("replacement");
        assert_eq!(
            model.loading_state()["loading"]["operations"][format!("mask_io:{import_scope}")]["phase"],
            "cancelled"
        );
        assert!(
            model
                .install_imported_masks_for_generation(
                    document_generation,
                    mask_generation,
                    import_generation,
                    &import_scope,
                    "stale".to_string(),
                    true,
                    vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
                    PathBuf::from("stale.geojson"),
                )
                .is_none()
        );
    }

    #[test]
    fn actor_model_enforces_scoped_viewport_revision_guards() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        let workspace = model.workspace_snapshot().unwrap();
        let id = workspace["active_viewport_id"].as_str().unwrap();
        let navigation = workspace["viewports"][0]["navigation_revision"]
            .as_u64()
            .unwrap();
        let presentation = workspace["viewports"][0]["presentation_revision"]
            .as_u64()
            .unwrap();

        model
            .dispatch(
                "viewer.viewports.camera.set",
                &json!({"viewport_id":id,"zoom":2.0,"if_navigation_revision":navigation}),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            model
                .dispatch(
                    "viewer.viewports.camera.set",
                    &json!({"viewport_id":id,"zoom":3.0,"if_navigation_revision":navigation}),
                )
                .unwrap()
                .unwrap_err()
                .kind,
            ControlErrorKind::Conflict
        );

        model
            .dispatch(
                "viewer.viewports.channels.set_color",
                &json!({"viewport_id":id,"channel":0,"color_rgb":[1,2,3],"if_presentation_revision":presentation}),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            model
                .dispatch(
                    "viewer.viewports.channels.set_color",
                    &json!({"viewport_id":id,"channel":0,"color_rgb":[3,2,1],"if_presentation_revision":presentation}),
                )
                .unwrap()
                .unwrap_err()
                .kind,
            ControlErrorKind::Conflict
        );
    }

    #[test]
    fn stale_dataset_worker_results_cannot_replace_a_newer_document_request() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        let stale = model.begin_dataset_open("first");
        let loading_error = model
            .dispatch("viewer.workspace.get", &json!({}))
            .expect("actor-owned methods never fall back while loading")
            .unwrap_err();
        assert_eq!(loading_error.kind, ControlErrorKind::NotReady);
        assert_eq!(
            loading_error.data.as_ref().unwrap()["loading"]["resources_ready"],
            false
        );
        let current = model.begin_dataset_open("second");
        assert!(!model.install_dataset_for_generation(stale, &dataset, Vec::new(), None));
        assert_eq!(model.mode(), ModelMode::Transition);
        assert!(model.install_dataset_for_generation(current, &dataset, Vec::new(), None));
        assert_eq!(model.mode(), ModelMode::Single);
    }

    #[test]
    fn unequal_splits_derive_each_viewport_from_the_retained_workspace_geometry() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        let left = model.render_workspace_snapshot().unwrap()["active_viewport_id"]
            .as_str()
            .unwrap()
            .to_string();
        model.report_viewport_geometry(&left, 1200.0, 800.0);
        let right = model
            .dispatch(
                "viewer.viewports.clone",
                &json!({"source_viewport_id":left,"layout":"horizontal","ratio":0.6}),
            )
            .unwrap()
            .unwrap()
            .response["viewport_id"]
            .as_str()
            .unwrap()
            .to_string();
        let workspace = model.render_workspace_snapshot().unwrap();
        let viewport = |id: &str| {
            workspace["viewports"]
                .as_array()
                .unwrap()
                .iter()
                .find(|viewport| viewport["viewport_id"] == id)
                .unwrap()
        };
        assert_eq!(
            viewport(&left)["camera"]["viewport"],
            json!([0.0, 0.0, 720.0, 800.0])
        );
        let right_width = viewport(&right)["camera"]["viewport"][2].as_f64().unwrap();
        assert!((right_width - 480.0).abs() < 1.0e-3);
        assert_eq!(viewport(&right)["camera"]["viewport"][3], json!(800.0));

        model
            .dispatch("viewer.viewports.remove", &json!({"viewport_id":left}))
            .unwrap()
            .unwrap();
        let remaining = model.render_workspace_snapshot().unwrap();
        assert_eq!(
            remaining["viewports"][0]["camera"]["viewport"],
            json!([0.0, 0.0, 1200.0, 800.0])
        );
    }

    #[test]
    fn dataset_replacement_preserves_observed_logical_geometry() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        model.report_viewport_geometry("viewport-1", 1234.0, 777.0);
        assert_eq!(
            model.loading_state()["loading"]["geometry"]["source"],
            "observed"
        );

        model.install_dataset(&dataset);
        let workspace = model.render_workspace_snapshot().unwrap();
        assert_eq!(
            workspace["viewports"][0]["camera"]["viewport"],
            json!([0.0, 0.0, 1234.0, 777.0])
        );
        assert_eq!(
            model.loading_state()["loading"]["geometry"]["source"],
            "observed"
        );
    }

    #[test]
    fn panel_changes_derive_background_geometry_without_a_frame() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        model.report_viewport_geometry("viewport-1", 1000.0, 700.0);

        let hidden = model
            .dispatch("viewer.panels.set", &json!({"left":false,"right":false}))
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(hidden["result"]["changed"], true);
        assert_eq!(
            hidden["result"]["panels"],
            json!({"left":false,"right":false})
        );
        assert_eq!(
            model.render_workspace_snapshot().unwrap()["viewports"][0]["camera"]["viewport"],
            json!([0.0, 0.0, 1740.0, 700.0])
        );
        assert_eq!(
            model.loading_state()["loading"]["geometry"]["source"],
            "derived"
        );

        model
            .dispatch("viewer.camera.fit", &json!({}))
            .unwrap()
            .unwrap();
        let fitted = model
            .dispatch("viewer.camera.get", &json!({}))
            .unwrap()
            .unwrap()
            .response;
        assert!(
            fitted["camera"]["zoom_screen_per_lvl0_px"]
                .as_f64()
                .unwrap()
                > 0.0
        );
    }

    #[test]
    fn renderer_bootstrap_atomically_replaces_workspace_and_supersedes_workers() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut source = AppModel::project();
        source.install_dataset(&dataset);
        let left = source.render_workspace_snapshot().unwrap()["active_viewport_id"]
            .as_str()
            .unwrap()
            .to_string();
        source
            .dispatch(
                "viewer.viewports.clone",
                &json!({"source_viewport_id":left,"layout":"vertical","ratio":0.7,"title":"Native second"}),
            )
            .unwrap()
            .unwrap();
        source
            .dispatch(
                "viewer.viewports.channels.set_visible",
                &json!({"viewport_id":left,"channels":[3],"mode":"only"}),
            )
            .unwrap()
            .unwrap();
        let renderer_workspace = source.render_workspace_snapshot().unwrap();

        let mut target = AppModel::project();
        let stale_generation = target.begin_dataset_open("superseded");
        target
            .bootstrap_dataset_from_renderer(&dataset, &renderer_workspace)
            .expect("native renderer state bootstraps atomically");
        assert!(!target.install_dataset_for_generation(
            stale_generation,
            &dataset,
            Vec::new(),
            None
        ));
        assert_eq!(
            target.render_workspace_snapshot().unwrap(),
            renderer_workspace
        );
    }

    #[test]
    fn plane_commands_retain_per_axis_slices_and_clamp_to_dataset_extents() {
        let (mut dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        dataset.dims.z = Some(1);
        dataset.dims.y = 2;
        dataset.dims.x = 3;
        dataset.dims.ndim = 4;
        for level in &mut dataset.levels {
            level.shape.insert(1, 7);
            level.chunks.insert(1, 1);
            level.scale.insert(1, 1.0);
            level.translation.insert(1, 0.0);
        }
        let mut model = AppModel::project();
        model.install_dataset(&dataset);

        let set = |model: &mut AppModel, mode: &str, slice: u64| {
            model
                .dispatch(
                    "viewer.viewports.planes.set",
                    &json!({"viewport_id":"viewport-1","mode":mode,"slice":slice}),
                )
                .unwrap()
                .unwrap()
                .response
        };
        let xy = set(&mut model, "xy", 99);
        assert_eq!(xy["result"]["plane"]["slice"], 6);
        assert_eq!(xy["result"]["plane"]["extent"], 7);
        assert_eq!(
            xy["result"]["plane"]["supported_modes"],
            json!(["xy", "xz", "yz"])
        );

        let xz = set(&mut model, "xz", 1234);
        assert_eq!(xz["result"]["plane"]["slice"], 511);
        assert_eq!(xz["result"]["plane"]["slice_axis"], "y");
        let yz = set(&mut model, "yz", 42);
        assert_eq!(yz["result"]["plane"]["slice"], 42);
        assert_eq!(yz["result"]["plane"]["slice_axis"], "x");

        let back_to_xy = model
            .dispatch(
                "viewer.viewports.planes.set",
                &json!({"viewport_id":"viewport-1","mode":"xy"}),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(back_to_xy["result"]["plane"]["slice"], 6);
    }

    #[test]
    fn invalid_presentation_commands_are_atomic() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        let before = model.render_workspace_snapshot().unwrap();

        let invalid_mode = model
            .dispatch(
                "viewer.viewports.channels.set_visible",
                &json!({"viewport_id":"viewport-1","channels":[0],"mode":"toggle"}),
            )
            .unwrap()
            .unwrap_err();
        assert_eq!(invalid_mode.kind, ControlErrorKind::InvalidParams);
        assert_eq!(model.render_workspace_snapshot().unwrap(), before);

        let invalid_rendering = model
            .dispatch(
                "viewer.viewports.rendering.set",
                &json!({"viewport_id":"viewport-1","show_hud":"yes"}),
            )
            .unwrap()
            .unwrap_err();
        assert_eq!(invalid_rendering.kind, ControlErrorKind::InvalidParams);
        assert_eq!(model.render_workspace_snapshot().unwrap(), before);
    }

    #[test]
    fn complete_channel_presentation_executes_and_roundtrips_without_a_renderer() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);

        let note = model
            .dispatch(
                "viewer.channels.set_note",
                &json!({"channel": 1, "note": "T-cell marker"}),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(note["channel"]["note"], "T-cell marker");

        let transform = model
            .dispatch(
                "viewer.channels.set_transform",
                &json!({
                    "channel": 1,
                    "offset_world": [12.5, -3.0],
                    "scale": [1.25, 0.75],
                    "rotation_rad": 0.5,
                }),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(transform["changed"], true);
        assert_eq!(transform["transform"]["offset_world"], json!([12.5, -3.0]));

        let order = model
            .dispatch(
                "viewer.viewports.channels.set_order",
                &json!({
                    "viewport_id": "viewport-1",
                    "channels": [4, 3, 2, 1, 0],
                    "mode": "exact",
                }),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(order["result"]["order"][0]["index"], 4);

        let group = model
            .dispatch(
                "viewer.viewports.channels.set_group",
                &json!({
                    "viewport_id": "viewport-1",
                    "channels": [1, 2],
                    "name": "Immune",
                    "color_rgb": [10, 20, 30],
                }),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(group["result"]["groups"][0]["name"], "Immune");
        assert_eq!(
            group["result"]["groups"][0]["members"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        model
            .dispatch(
                "viewer.channels.presentation.set",
                &json!({"search": "CD", "sort": "visible_first"}),
            )
            .unwrap()
            .unwrap();
        let projection = model.render_workspace_snapshot().unwrap();
        assert_eq!(projection["channel_presentation"]["search"], "CD");
        assert_eq!(projection["channel_transforms"][1]["rotation_rad"], 0.5);
        assert_eq!(
            projection["viewports"][0]["channel_order"],
            json!([4, 3, 2, 1, 0])
        );

        let mut restored = AppModel::project();
        restored
            .bootstrap_dataset_from_renderer(&dataset, &projection)
            .expect("complete presentation projection roundtrips");
        assert_eq!(restored.render_workspace_snapshot().unwrap(), projection);

        let before = restored.render_workspace_snapshot().unwrap();
        let invalid = restored
            .dispatch(
                "viewer.channels.set_transform",
                &json!({"channel": 1, "scale": [0.0, 1.0]}),
            )
            .unwrap()
            .unwrap_err();
        assert_eq!(invalid.kind, ControlErrorKind::InvalidParams);
        assert_eq!(restored.render_workspace_snapshot().unwrap(), before);
    }

    #[test]
    fn stale_renderer_observation_cannot_revert_actor_owned_state() {
        let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
        let mut model = AppModel::project();
        model.install_dataset(&dataset);
        let stale = model.render_workspace_snapshot().unwrap();

        model
            .dispatch(
                "viewer.channels.set_note",
                &json!({"channel":1,"note":"new actor value"}),
            )
            .unwrap()
            .unwrap();
        model
            .dispatch(
                "viewer.channels.set_transform",
                &json!({"channel":1,"offset_world":[8.0,-4.0]}),
            )
            .unwrap()
            .unwrap();
        model
            .dispatch(
                "viewer.viewports.channels.set_order",
                &json!({
                    "viewport_id":"viewport-1",
                    "channels":[4,3,2,1,0],
                    "mode":"exact",
                }),
            )
            .unwrap()
            .unwrap();
        model
            .dispatch(
                "viewer.channels.presentation.set",
                &json!({"search":"actor search","sort":"visible_first"}),
            )
            .unwrap()
            .unwrap();
        model
            .dispatch("viewer.panels.set", &json!({"left":false,"right":false}))
            .unwrap()
            .unwrap();
        let current_revision = model.mark_projection_dirty();

        assert!(model.observe_renderer_workspace(&stale, current_revision - 1));
        let current = model.render_workspace_snapshot().unwrap();
        assert_eq!(current["channel_metadata"][1]["note"], "new actor value");
        assert_eq!(
            current["channel_transforms"][1]["offset_world"],
            json!([8.0, -4.0])
        );
        assert_eq!(
            current["viewports"][0]["channel_order"],
            json!([4, 3, 2, 1, 0])
        );
        assert_eq!(current["channel_presentation"]["search"], "actor search");
        assert_eq!(current["panels"], json!({"left":false,"right":false}));

        assert!(!model.observe_renderer_workspace(&stale, current_revision + 1));
        let mut wrong_document = stale;
        wrong_document["shared_resources"]["dataset_source"] = json!("another dataset");
        assert!(!model.observe_renderer_workspace(&wrong_document, current_revision));
        assert_eq!(model.render_workspace_snapshot().unwrap(), current);
    }

    #[test]
    fn project_metadata_and_roi_transactions_execute_without_a_renderer() {
        let mut model = AppModel::project();
        let created = model
            .dispatch("project.create", &json!({"default_dataset":"cohort-a"}))
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(
            created["project"]["metadata"]["default_dataset"],
            "cohort-a"
        );

        for (id, path) in [("roi-a", "/tmp/a.ome.zarr"), ("roi-b", "/tmp/b.ome.zarr")] {
            let added = model
                .dispatch(
                    "project.rois.add",
                    &json!({"id":id,"path":path,"metadata":{"group":"test"}}),
                )
                .unwrap()
                .unwrap()
                .response;
            assert_eq!(added["roi"]["id"], id);
        }
        let selected = model
            .dispatch(
                "project.rois.select",
                &json!({"ids":["roi-b"],"mode":"replace"}),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(selected["selected"], json!(["roi-b"]));
        assert_eq!(selected["focused"], "roi-b");

        let updated = model
            .dispatch(
                "project.rois.update",
                &json!({
                    "target_id":"roi-b",
                    "changes":{"display_name":"B","segmentation_path":"/tmp/b.parquet"},
                }),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(updated["roi"]["display_name"], "B");
        model
            .dispatch("project.rois.reorder", &json!({"ids":["roi-b","roi-a"]}))
            .unwrap()
            .unwrap();
        let rois = model
            .dispatch("project.rois.list", &json!({}))
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(rois["rois"][0]["id"], "roi-b");
        assert_eq!(rois["rois"][0]["selected"], true);

        let before = rois.clone();
        let duplicate = model
            .dispatch(
                "project.rois.add",
                &json!({"id":"roi-a","path":"/tmp/c.ome.zarr"}),
            )
            .unwrap()
            .unwrap_err();
        assert_eq!(duplicate.kind, ControlErrorKind::Conflict);
        assert_eq!(
            model
                .dispatch("project.rois.list", &json!({}))
                .unwrap()
                .unwrap()
                .response,
            before
        );

        let stepped = model
            .dispatch("project.rois.next", &json!({"wrap":true}))
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(stepped["focused"], "roi-a");
    }

    #[test]
    fn renderer_project_bootstrap_cannot_revert_actor_owned_commits() {
        let mut roi = crate::data::project_config::ProjectRoi {
            id: "roi".to_string(),
            display_name: Some("Initial".to_string()),
            ..Default::default()
        };
        roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
            PathBuf::from("/tmp/bootstrap.ome.zarr"),
        ));
        let bootstrap = ProjectModelSnapshot {
            rois: vec![roi],
            ..ProjectModelSnapshot::default()
        };
        let mut model = AppModel::project();
        assert!(model.bootstrap_project_from_renderer(bootstrap.clone()));
        model
            .dispatch(
                "project.rois.update",
                &json!({"target_id":"roi","changes":{"display_name":"Actor"}}),
            )
            .unwrap()
            .unwrap();

        assert!(!model.bootstrap_project_from_renderer(bootstrap));
        assert_eq!(
            model.project_snapshot().rois[0].display_name.as_deref(),
            Some("Actor")
        );
    }

    #[test]
    fn renderer_project_bootstrap_normalizes_persisted_state_and_saved_views() {
        let mut model = AppModel::project();
        assert!(model.bootstrap_project_from_renderer(ProjectModelSnapshot {
            state: json!({
                "browser": "invalid legacy value",
                "view_presets": [{
                    "name": "Actor view",
                    "description": "",
                    "spec": {"channel_ref": {"label": "DAPI"}},
                }],
            }),
            ..ProjectModelSnapshot::default()
        }));

        let snapshot = model.project_snapshot();
        assert!(snapshot.state["browser"].is_object());
        assert_eq!(snapshot.view_count, 1);
        assert_eq!(snapshot.view_presets[0]["name"], "Actor view");

        let replacement = ProjectModelSnapshot {
            state: Value::String("invalid legacy state".to_string()),
            view_count: 99,
            ..ProjectModelSnapshot::default()
        };
        let mut replacement_model = AppModel::project();
        assert!(replacement_model.bootstrap_project_from_renderer(replacement));
        let replacement = replacement_model.project_snapshot();
        assert!(replacement.state.is_object());
        assert_eq!(replacement.view_count, 0);
    }
}
