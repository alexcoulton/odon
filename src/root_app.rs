use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};
use std::{collections::HashMap, collections::HashSet};

use eframe::egui;

use crate::app::{
    LabelPromptSessionPreference, OmeZarrViewerApp, S3DatasetSelection, ViewerRequest,
};
use crate::app_support::menu::{NativeMenu, NativeMenuAction};
use crate::app_support::settings::{AppSettings, AutoContrastMethod, settings_file_path};
use crate::data::dataset_kind::{
    LocalDatasetKind, classify_local_dataset_path, normalize_local_dataset_path,
};
use crate::data::dataset_source::DatasetSource;
use crate::data::ome::OmeZarrDataset;
use crate::data::project_config::ProjectRoi;
use crate::data::remote_store::{
    S3BrowseEntry, S3BrowseListing, S3Browser, S3Store, build_http_store, build_s3_browser,
    build_s3_store, list_s3_prefix,
};
use crate::data::samplesheet::load_samplesheet_csv;
use crate::deep_link::DeepLinkRequest;
use crate::mosaic::{MosaicRequest, MosaicViewerApp};
use crate::objects::{ObjectPreloadMode, ObjectPreloadSettings, PreloadedObjectLayer};
use crate::project::{
    ProjectObjectCacheUiState, ProjectSpace, ProjectSpaceAction, ProjectViewSpec,
};
use crate::spatialdata::{SpatialDataDiscovery, SpatialDataElement, discover_spatialdata};
use crate::ui::top_bar;
use crate::xenium::{TiffPlaneSelection, TiffPyramid, discover_xenium_explorer};
use crate::{log_debug, log_info, log_warn};
use odon::control::{ControlError, ControlErrorKind, TaskState};
use odon::mcp::{OdonControlBridge, OdonControlRequest};
use rfd::FileDialog;

fn control_event_name(method: &str) -> &'static str {
    match method {
        "viewer.camera.set"
        | "viewer.camera.zoom_in"
        | "viewer.camera.zoom_out"
        | "viewer.camera.fit" => "viewer.camera.changed",
        "viewer.channels.set_active"
        | "viewer.channels.set_visible"
        | "viewer.channels.set_color"
        | "viewer.channels.set_note"
        | "viewer.channels.set_contrast"
        | "viewer.channels.set_transform"
        | "viewer.channels.reset_transform"
        | "viewer.channels.set_order"
        | "viewer.channels.set_group" => "viewer.channels.changed",
        "viewer.planes.set" | "viewer.planes.next" | "viewer.planes.previous" => {
            "viewer.planes.changed"
        }
        "viewer.objects.select_rect" | "viewer.objects.clear_selection" => {
            "viewer.selection.changed"
        }
        "viewer.objects.set_visibility"
        | "viewer.objects.set_filter"
        | "viewer.objects.clear_filter"
        | "viewer.native_layers.set_active"
        | "viewer.native_layers.set_visibility"
        | "viewer.native_layers.set_order"
        | "viewer.native_layers.set_offset"
        | "viewer.native_layers.reset_offset" => "viewer.layers.changed",
        "project.open"
        | "datasets.open_ome_zarr"
        | "datasets.open_tiff"
        | "datasets.open_mosaic_samplesheet"
        | "app.navigation.show_project" => "application.mode.changed",
        "project.rois.open" => "project.active_roi.changed",
        "viewer.screenshot.capture"
        | "viewer.workspace.screenshot.capture"
        | "app.screenshot.capture"
        | "project.screenshot.capture" => "viewer.screenshot.completed",
        _ => "application.state.changed",
    }
}

fn control_event_source(method: &str) -> &'static str {
    if method.starts_with("project.") {
        "project:active"
    } else if method.starts_with("app.") || method.starts_with("datasets.") {
        "application"
    } else {
        "viewer:active"
    }
}

fn active_viewport_compatibility_event(method: &str) -> Option<&'static str> {
    match method {
        "viewer.viewports.camera.set" | "viewer.viewports.camera.fit" => {
            Some("viewer.camera.changed")
        }
        "viewer.viewports.planes.set" => Some("viewer.planes.changed"),
        "viewer.viewports.channels.set_visible"
        | "viewer.viewports.channels.set"
        | "viewer.viewports.channels.set_active"
        | "viewer.viewports.channels.set_color"
        | "viewer.viewports.channels.set_contrast"
        | "viewer.viewports.channels.set_order"
        | "viewer.viewports.channels.set_group" => Some("viewer.channels.changed"),
        "viewer.viewports.rendering.set" => Some("viewer.rendering.changed"),
        "viewer.viewports.objects.style.set"
        | "viewer.viewports.objects.legend.set"
        | "viewer.viewports.objects.filter.set"
        | "viewer.viewports.objects.filter.clear"
        | "viewer.viewports.layers.set"
        | "viewer.viewports.layers.set_visibility"
        | "viewer.viewports.layers.set_order"
        | "viewer.viewports.layers.set_active" => Some("viewer.layers.changed"),
        _ => None,
    }
}

fn control_application_error(method: &str, response: &serde_json::Value) -> Option<ControlError> {
    fn find_error(value: &serde_json::Value) -> Option<&str> {
        let object = value.as_object()?;
        if let Some(message) = object.get("error").and_then(serde_json::Value::as_str) {
            return Some(message);
        }
        object
            .values()
            .filter(|value| value.is_object())
            .find_map(find_error)
    }

    let message = find_error(response)?;
    let kind = if message.contains("revision conflict") {
        ControlErrorKind::Conflict
    } else if message.contains("transitioning") {
        ControlErrorKind::NotReady
    } else if message.contains("No dataset viewer")
        || message.contains("No mosaic viewer")
        || message.contains("No single-image viewer")
        || message.contains("requires mosaic mode")
        || message.contains("requires single-image mode")
        || message.contains("available in single-image mode")
        || message.contains("for single-image mode")
        || message.contains("requires a single-image viewer")
        || message.contains("requires a mosaic viewer")
    {
        ControlErrorKind::WrongMode
    } else if message.contains("does not exist")
        || message.contains("not found")
        || message.contains("no channel matches")
        || message.contains("out of range")
    {
        ControlErrorKind::ResourceNotFound
    } else if message.contains("requires ")
        || message.contains("provide ")
        || message.contains("invalid ")
        || message.contains("must ")
    {
        ControlErrorKind::InvalidParams
    } else {
        ControlErrorKind::Application
    };
    Some(
        ControlError::new(kind, message).with_data(serde_json::json!({
            "method": method,
            "application_response": response,
        })),
    )
}

#[derive(Debug, Clone)]
struct SpatialOpenDialog {
    discovery: SpatialDataDiscovery,
    selected_image: usize,
    selected_labels: Option<usize>,
    selected_shapes: Vec<usize>,
    selected_points: Option<usize>,
    points_max: usize,
    status: String,
}

struct ReturnToSingleState {
    dataset_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteMode {
    Http,
    S3,
}

struct RootRemoteS3BrowserState {
    session: S3Browser,
    signature: String,
    current_prefix: String,
    parent_prefix: Option<String>,
    entries: Vec<S3BrowseEntry>,
    current_is_dataset: bool,
    selected_dataset_prefixes: HashSet<String>,
    listing_cache: HashMap<String, S3BrowseListing>,
}

enum RootRemoteAction {
    OpenSingle {
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        runtime: Option<Arc<tokio::runtime::Runtime>>,
    },
    OpenS3Mosaic(Vec<S3DatasetSelection>),
    AddToProject(Vec<DatasetSource>),
}

enum Mode {
    Project {
        project_space: ProjectSpace,
    },
    Single(OmeZarrViewerApp),
    Mosaic {
        mosaic: MosaicViewerApp,
        ret: ReturnToSingleState,
    },
    Transition,
}

fn settings_help_button(ui: &mut egui::Ui, text: &'static str) {
    let _ = ui.small_button("?").on_hover_text(text);
}

struct ProjectObjectPreloadEvent {
    path: PathBuf,
    settings: ObjectPreloadSettings,
    result: Result<PreloadedObjectLayer, String>,
    finished: bool,
}

#[derive(Debug)]
struct ViewportScreenshotRequest {
    path: PathBuf,
    crop_rect_points: Option<egui::Rect>,
}

fn screenshot_crop_bounds(
    image_size: [usize; 2],
    crop_rect_points: Option<egui::Rect>,
    pixels_per_point: f32,
) -> Option<(usize, usize, usize, usize)> {
    let [width, height] = image_size;
    let pixels_per_point = pixels_per_point.max(1e-6);
    let (x0, y0, x1, y1) = crop_rect_points.map_or((0usize, 0usize, width, height), |rect| {
        let x0 = (rect.min.x * pixels_per_point).floor().max(0.0) as usize;
        let y0 = (rect.min.y * pixels_per_point).floor().max(0.0) as usize;
        let x1 = (rect.max.x * pixels_per_point).ceil().max(0.0) as usize;
        let y1 = (rect.max.y * pixels_per_point).ceil().max(0.0) as usize;
        (x0.min(width), y0.min(height), x1.min(width), y1.min(height))
    });
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

fn project_roi_segmentation_path(
    project_space: &ProjectSpace,
    roi: &ProjectRoi,
) -> Option<PathBuf> {
    let segpath = roi.segpath.as_ref()?;
    if segpath.is_absolute() {
        Some(segpath.clone())
    } else {
        project_space
            .project_dir()
            .map(|dir| dir.join(segpath))
            .or_else(|| Some(segpath.clone()))
    }
}

fn resolve_example_project_path(example: &str) -> Option<PathBuf> {
    let normalized = normalize_example_name(example);
    let project_name = match normalized.as_str() {
        "synthetic5ch" | "synthetic" | "demo" => "synthetic_5ch.project.json",
        _ => return None,
    };

    example_dirs()
        .into_iter()
        .map(|dir| dir.join(project_name))
        .find(|path| path.is_file())
}

fn apply_example_defaults(req: &mut DeepLinkRequest, example: &str) {
    let normalized = normalize_example_name(example);
    if !matches!(normalized.as_str(), "synthetic5ch" | "synthetic" | "demo") {
        return;
    }
    if req.roi.is_none() {
        req.roi = Some("synthetic_5ch.ome.zarr".to_string());
    }
    if req.channel.is_none() {
        req.channel = Some("DAPI".to_string());
    }
    if req.visible_channels.is_empty() {
        req.visible_channels = vec!["DAPI".to_string(), "CD3".to_string(), "PanCK".to_string()];
    }
    if req.visible_channel_group.is_none() {
        req.visible_channel_group = Some("Synthetic example".to_string());
    }
    if req.channel_order.is_none() {
        req.channel_order = Some(crate::deep_link::DeepLinkChannelOrder::Listed);
    }
}

fn normalize_example_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn example_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(bin_dir) = exe.parent()
    {
        dirs.push(bin_dir.join("examples"));
        dirs.push(bin_dir.join("../Resources/examples"));
        dirs.push(bin_dir.join("../../Resources/examples"));
    }
    dirs.push(PathBuf::from("/usr/share/odon/examples"));
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("fixtures"));
    }
    dirs
}

fn project_object_segmentation_paths(project_space: &ProjectSpace) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    for roi in &project_space.config().rois {
        let Some(path) = project_roi_segmentation_path(project_space, roi) else {
            continue;
        };
        let supported = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "parquet" | "geoparquet"))
            .unwrap_or(false);
        if supported && path.exists() && seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    paths
}

fn project_object_cache_ui_state(
    project_space: &ProjectSpace,
    cached: usize,
    total: usize,
    done: usize,
    failed: usize,
    loading: bool,
    cached_settings: ObjectPreloadSettings,
) -> ProjectObjectCacheUiState {
    let paths = project_object_segmentation_paths(project_space);
    let on_disk_bytes = paths
        .iter()
        .filter_map(|path| path.metadata().ok().map(|meta| meta.len()))
        .sum::<u64>();
    ProjectObjectCacheUiState {
        available_count: paths.len(),
        on_disk_bytes,
        cached,
        total,
        done,
        failed,
        loading,
        cached_settings,
    }
}

fn expand_control_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}

pub struct RootApp {
    mode: Mode,
    gpu_available: bool,
    close_dialog_open: bool,
    spatial_open: Option<SpatialOpenDialog>,
    pending_open_root: Option<PathBuf>,
    pending_control_open_root: Option<PathBuf>,
    pending_deep_link: Option<DeepLinkRequest>,
    deep_link_rx: Option<Receiver<DeepLinkRequest>>,
    object_preload_project: Option<PathBuf>,
    object_preload_rx: Option<Receiver<ProjectObjectPreloadEvent>>,
    object_preload_cache: HashMap<(PathBuf, ObjectPreloadSettings), Arc<PreloadedObjectLayer>>,
    object_preload_settings: ObjectPreloadSettings,
    object_preload_total: usize,
    object_preload_done: usize,
    object_preload_failed: usize,
    view_show_scale_bar: bool,
    remote_dialog_open: bool,
    remote_mode: RemoteMode,
    remote_http_url: String,
    remote_s3_endpoint: String,
    remote_s3_region: String,
    remote_s3_bucket: String,
    remote_s3_prefix: String,
    remote_s3_access_key: String,
    remote_s3_secret_key: String,
    remote_status: String,
    remote_s3_browser: Option<RootRemoteS3BrowserState>,
    label_prompt_preference: LabelPromptSessionPreference,
    app_settings: AppSettings,
    settings_open: bool,
    settings_status: String,
    active_help_topic: Option<crate::ui::help::HelpTopic>,
    control_bridge: Option<OdonControlBridge>,
    control_external_revision: u64,
    control_project_revision: u64,
    control_observed_state: Option<serde_json::Value>,
    control_mutated_this_frame: bool,
    control_last_observed_at: Instant,
    #[cfg(target_os = "macos")]
    native_menu: Option<NativeMenu>,
}

impl RootApp {
    fn spawn_control_bridge(
        ctx: &egui::Context,
        settings_status: &mut String,
    ) -> Option<OdonControlBridge> {
        match OdonControlBridge::spawn_default(ctx.clone()) {
            Ok(bridge) => Some(bridge),
            Err(err) => {
                let msg = format!("Odon control server unavailable: {err}");
                if settings_status.trim().is_empty() {
                    *settings_status = msg;
                } else {
                    settings_status.push_str("; ");
                    settings_status.push_str(&msg);
                }
                None
            }
        }
    }

    fn load_app_settings() -> (AppSettings, String) {
        match AppSettings::load() {
            Ok(settings) => (settings, String::new()),
            Err(err) => (
                AppSettings::default(),
                format!("Settings load failed: {err}"),
            ),
        }
    }

    fn remote_s3_signature(&self) -> String {
        format!(
            "{}\n{}\n{}\n{}\n{}",
            self.remote_s3_endpoint.trim(),
            self.remote_s3_region.trim(),
            self.remote_s3_bucket.trim(),
            self.remote_s3_access_key.trim(),
            self.remote_s3_secret_key.trim()
        )
    }

    fn clear_remote_s3_browser(&mut self) {
        self.remote_s3_browser = None;
    }

    fn configure_single_app(&self, app: &mut OmeZarrViewerApp) {
        app.set_show_scale_bar(self.view_show_scale_bar);
        app.set_label_prompt_preference(self.label_prompt_preference);
        app.set_auto_contrast_settings(self.app_settings.auto_contrast);
        app.set_fast_object_rendering(self.app_settings.fast_object_rendering);
    }

    fn configure_mosaic_app(&self, mosaic: &mut MosaicViewerApp) {
        mosaic.set_fast_object_rendering(self.app_settings.fast_object_rendering);
    }

    fn apply_app_settings_to_mode(&mut self) {
        match &mut self.mode {
            Mode::Single(app) => {
                app.set_auto_contrast_settings(self.app_settings.auto_contrast);
                app.set_fast_object_rendering(self.app_settings.fast_object_rendering);
            }
            Mode::Mosaic { mosaic, .. } => {
                mosaic.set_fast_object_rendering(self.app_settings.fast_object_rendering);
            }
            Mode::Project { .. } | Mode::Transition => {}
        }
    }

    fn persist_app_settings(&mut self) {
        match self.app_settings.save() {
            Ok(path) => {
                self.settings_status = format!("Saved settings to {}.", path.display());
            }
            Err(err) => {
                self.settings_status = format!("Settings save failed: {err}");
            }
        }
    }

    fn record_recent_project(&mut self, path: &Path) {
        if self.app_settings.record_recent_project(path) {
            self.persist_app_settings();
        }
    }

    fn forget_recent_project(&mut self, path: &Path) {
        if self.app_settings.forget_recent_project(path) {
            self.persist_app_settings();
        }
    }

    fn clear_recent_projects(&mut self) {
        if self.app_settings.clear_recent_projects() {
            self.persist_app_settings();
        }
    }

    fn process_control_requests(&mut self, ctx: &egui::Context) {
        const MAX_REQUESTS_PER_FRAME: usize = 32;
        let mut requests = Vec::new();
        if let Some(bridge) = self.control_bridge.as_ref() {
            for _ in 0..MAX_REQUESTS_PER_FRAME {
                match bridge.try_recv() {
                    Ok(request) => requests.push(request),
                    Err(_) => break,
                }
            }
            if bridge.pending_len() > 0 {
                ctx.request_repaint();
            }
        }
        for request in requests {
            self.reply_to_control_request(ctx, request);
        }
    }

    fn control_observed_snapshot(&mut self) -> serde_json::Value {
        let workspace = match &mut self.mode {
            Mode::Single(app) => app.control_viewport_workspace_snapshot(),
            _ => serde_json::Value::Null,
        };
        serde_json::json!({
            "view": self.control_current_view(),
            "camera": self.control_get_camera(),
            "channels": self.control_channels(),
            "smooth": self.control_get_smooth_pixels(),
            "loading": self.control_get_loading_state(),
            "workspace": workspace,
            "selection": match &self.mode {
                Mode::Single(app) => app.control_object_selection_signature(),
                _ => serde_json::Value::Null,
            },
        })
    }

    fn publish_observed_control_changes(&mut self) {
        if !self.control_mutated_this_frame
            && self.control_last_observed_at.elapsed() < Duration::from_millis(33)
        {
            return;
        }
        self.control_last_observed_at = Instant::now();
        let snapshot = self.control_observed_snapshot();
        let Some(previous) = self.control_observed_state.replace(snapshot.clone()) else {
            return;
        };
        if self.control_mutated_this_frame {
            return;
        }
        let Some(bridge) = self.control_bridge.as_ref() else {
            return;
        };
        for (field, event, source) in [
            ("view", "application.mode.changed", "application"),
            ("camera", "viewer.camera.changed", "viewer:active"),
            ("channels", "viewer.channels.changed", "viewer:active"),
            ("smooth", "viewer.rendering.changed", "viewer:active"),
            ("loading", "viewer.readiness.changed", "viewer:active"),
            ("workspace", "viewer.workspace.changed", "viewer:workspace"),
            ("selection", "viewer.selection.changed", "viewer:active"),
        ] {
            if previous.get(field) != snapshot.get(field) {
                bridge.publish_native_event(
                    event,
                    source,
                    snapshot
                        .get(field)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
            }
        }
    }

    fn reply_to_control_request(&mut self, ctx: &egui::Context, request: OdonControlRequest) {
        if let Some(task_id) = request.task_id.as_deref() {
            match request.task_registry.get(task_id) {
                Ok(task) if task.state == TaskState::Cancelled => {
                    let _ = request.reply.send(Err(ControlError::new(
                        ControlErrorKind::Cancelled,
                        "task was cancelled",
                    )
                    .with_data(serde_json::json!({"task_id": task_id}))));
                    return;
                }
                Ok(_) => {
                    let _ = request.task_registry.mark_running(task_id);
                }
                Err(error) => {
                    let _ = request.reply.send(Err(error));
                    return;
                }
            }
        }
        let method = request.command.method();
        let mutates = request.command.mutates();
        let current_revision = request.event_hub.revision();
        if let Some(expected) = request.command.if_revision()
            && expected != current_revision
        {
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "state revision conflict: expected {expected}, current revision is {current_revision}"
                ),
            )
            .with_data(serde_json::json!({
                "method": method,
                "expected_revision": expected,
                "current_revision": current_revision,
            }))));
            return;
        }
        let params = request.command.params();
        let mut response = match method {
            "app.get_state" => self.control_current_view(),
            "app.get_method_availability" => self.control_method_availability(params),
            "project.rois.list" => self.control_project_rois(),
            "project.get" => self.control_get_project(),
            "project.create" => self.control_create_project(params),
            "project.save_as" => self.control_save_project_as(params),
            "project.update_metadata" => self.control_update_project_metadata(params),
            "project.samplesheets.inspect" => self.control_inspect_samplesheet(params),
            "project.samplesheets.validate" => self.control_inspect_samplesheet(params),
            "project.samplesheets.import" => self.control_import_samplesheet(params),
            "project.samplesheets.export" => self.control_export_samplesheet(params),
            "project.discovery.add_root" => self.control_add_discovery_root(params),
            "project.objects.preload.get" => self.control_get_project_object_preload(),
            "project.objects.preload.list_sources" => {
                self.control_list_project_object_preload_sources(params)
            }
            "project.objects.preload.start" => self.control_start_project_object_preload(params),
            "project.objects.preload.clear" => self.control_clear_project_object_preload(),
            "project.rois.get" => self.control_get_project_roi(params),
            "project.rois.add" => self.control_add_project_roi(params),
            "project.rois.update" => self.control_update_project_roi(params),
            "project.rois.remove" => self.control_remove_project_roi(params),
            "project.rois.reorder" => self.control_reorder_project_rois(params),
            "project.rois.get_selection" => self.control_get_project_roi_selection(),
            "project.rois.select" => self.control_select_project_rois(params),
            "project.rois.focus" => self.control_focus_project_roi(params),
            "project.rois.next" => self.control_step_project_roi(params, true),
            "project.rois.previous" => self.control_step_project_roi(params, false),
            "project.rois.open_selected_mosaic" => self.control_open_selected_project_mosaic(ctx),
            "viewer.channels.list" => self.control_channels(),
            "viewer.channels.list_visible" => self.control_visible_channels(),
            "viewer.workspace.get" | "viewer.viewports.list" => {
                self.control_viewport_operation("workspace.get", params)
            }
            "viewer.workspace.layout.get" => {
                self.control_viewport_operation("workspace.layout.get", params)
            }
            "viewer.workspace.layout.set" => {
                self.control_viewport_operation("workspace.layout.set", params)
            }
            "viewer.workspace.swap" => self.control_viewport_operation("workspace.swap", params),
            "viewer.viewports.get" => self.control_viewport_operation("viewports.get", params),
            "viewer.viewports.create" => {
                self.control_viewport_operation("viewports.create", params)
            }
            "viewer.viewports.clone" => self.control_viewport_operation("viewports.create", params),
            "viewer.viewports.rename" => {
                self.control_viewport_operation("viewports.rename", params)
            }
            "viewer.viewports.remove" => {
                self.control_viewport_operation("viewports.remove", params)
            }
            "viewer.viewports.set_active" => {
                self.control_viewport_operation("viewports.set_active", params)
            }
            "viewer.viewport_links.set" => {
                self.control_viewport_operation("viewport_links.set", params)
            }
            "viewer.viewport_links.get" => {
                self.control_viewport_operation("viewport_links.get", params)
            }
            "viewer.viewport_links.list" => {
                self.control_viewport_operation("viewport_links.list", params)
            }
            "viewer.viewport_links.create" => {
                self.control_viewport_operation("viewport_links.create", params)
            }
            "viewer.viewport_links.update" => {
                self.control_viewport_operation("viewport_links.update", params)
            }
            "viewer.viewport_links.remove" => {
                self.control_viewport_operation("viewport_links.remove", params)
            }
            "viewer.viewports.camera.get" => {
                self.control_viewport_operation("viewports.camera.get", params)
            }
            "viewer.viewports.camera.set" => {
                self.control_viewport_operation("viewports.camera.set", params)
            }
            "viewer.viewports.camera.fit" => {
                self.control_viewport_operation("viewports.camera.fit", params)
            }
            "viewer.viewports.planes.get" => {
                self.control_viewport_operation("viewports.planes.get", params)
            }
            "viewer.viewports.planes.set" => {
                self.control_viewport_operation("viewports.planes.set", params)
            }
            "viewer.viewports.channels.get" => {
                self.control_viewport_operation("viewports.channels.get", params)
            }
            "viewer.viewports.channels.set_visible" => {
                self.control_viewport_operation("viewports.channels.set_visible", params)
            }
            "viewer.viewports.channels.set" => {
                self.control_viewport_operation("viewports.channels.set_visible", params)
            }
            "viewer.viewports.channels.set_active" => {
                self.control_viewport_operation("viewports.channels.set_active", params)
            }
            "viewer.viewports.channels.set_color" => {
                self.control_viewport_operation("viewports.channels.set_color", params)
            }
            "viewer.viewports.channels.set_contrast" => {
                self.control_viewport_operation("viewports.channels.set_contrast", params)
            }
            "viewer.viewports.channels.set_order" => {
                self.control_viewport_operation("viewports.channels.set_order", params)
            }
            "viewer.viewports.channels.list_groups" => {
                self.control_viewport_operation("viewports.channels.list_groups", params)
            }
            "viewer.viewports.channels.set_group" => {
                self.control_viewport_operation("viewports.channels.set_group", params)
            }
            "viewer.viewports.rendering.get" => {
                self.control_viewport_operation("viewports.rendering.get", params)
            }
            "viewer.viewports.rendering.set" => {
                self.control_viewport_operation("viewports.rendering.set", params)
            }
            "viewer.viewports.objects.style.get" => {
                self.control_viewport_operation("viewports.objects.style.get", params)
            }
            "viewer.viewports.objects.style.set" => {
                self.control_viewport_operation("viewports.objects.style.set", params)
            }
            "viewer.viewports.objects.legend.set" => {
                self.control_viewport_operation("viewports.objects.legend.set", params)
            }
            "viewer.viewports.objects.filter.get" => {
                self.control_viewport_operation("viewports.objects.filter.get", params)
            }
            "viewer.viewports.objects.filter.set" => {
                self.control_viewport_operation("viewports.objects.filter.set", params)
            }
            "viewer.viewports.objects.filter.clear" => {
                self.control_viewport_operation("viewports.objects.filter.clear", params)
            }
            "viewer.viewports.layers.list" => {
                self.control_viewport_operation("viewports.layers.list", params)
            }
            "viewer.viewports.layers.get" => {
                self.control_viewport_operation("viewports.layers.get", params)
            }
            "viewer.viewports.layers.set" => {
                self.control_viewport_operation("viewports.layers.set", params)
            }
            "viewer.viewports.layers.set_visibility" => {
                self.control_viewport_operation("viewports.layers.set_visibility", params)
            }
            "viewer.viewports.layers.set_order" => {
                self.control_viewport_operation("viewports.layers.set_order", params)
            }
            "viewer.viewports.layers.set_active" => {
                self.control_viewport_operation("viewports.layers.set_active", params)
            }
            "viewer.planes.get" => self.control_get_plane(),
            "viewer.planes.set" => self.control_set_plane(params),
            "viewer.planes.next" => self.control_step_plane(params, true),
            "viewer.planes.previous" => self.control_step_plane(params, false),
            "viewer.panels.get" => self.control_get_side_panels(),
            "viewer.panels.set" => self.control_set_side_panels(params),
            "viewer.rendering.get_smooth_pixels" => self.control_get_smooth_pixels(),
            "viewer.rendering.set_smooth_pixels" => self.control_set_smooth_pixels(params),
            "viewer.rendering.get_state" => self.control_get_rendering_state(),
            "viewer.planes.operation_availability" => self.control_plane_operation_availability(),
            "app.get_loading_state" => self.control_get_loading_state(),
            "viewer.channels.get_active" => self.control_active_channel(),
            "viewer.channels.set_active" => self.control_set_active_channel(params),
            "viewer.channels.set_visible" => self.control_set_visible_channels(params),
            "viewer.channels.set_color" => self.control_set_channel_color(params),
            "viewer.channels.set_note" => self.control_set_channel_note(params),
            "viewer.channels.get_transform" => self.control_get_channel_transform(params),
            "viewer.channels.set_transform" => self.control_set_channel_transform(params),
            "viewer.channels.reset_transform" => self.control_reset_channel_transform(params),
            "viewer.native_layers.list" => self.control_native_layers(),
            "viewer.native_layers.get" => self.control_get_native_layer(params),
            "viewer.native_layers.set_active" => self.control_set_active_native_layer(params),
            "viewer.native_layers.set_visibility" => {
                self.control_set_native_layer_visibility(params)
            }
            "viewer.native_layers.set_order" => self.control_set_native_layer_order(params),
            "viewer.native_layers.set_offset" => self.control_set_native_layer_offset(params),
            "viewer.native_layers.reset_offset" => self.control_reset_native_layer_offset(params),
            "project.open" => self.control_open_project(params),
            "datasets.open_ome_zarr" => self.control_open_ome_zarr(params),
            "datasets.inspect" => self.control_inspect_dataset(params),
            "datasets.open_spatialdata" => self.control_open_spatialdata(ctx, params),
            "datasets.open_xenium" => self.control_open_xenium(ctx, params),
            "datasets.open_http" => self.control_open_http(ctx, params),
            "datasets.s3.get_session" => self.control_get_s3_session(),
            "datasets.s3.configure_session" => self.control_configure_s3_session(params),
            "datasets.s3.clear_session" => self.control_clear_s3_session(),
            "datasets.s3.list" => self.control_list_s3(params),
            "datasets.open_s3" => self.control_open_s3(ctx, params),
            "deep_links.parse" => self.control_parse_deep_link(params),
            "deep_links.resolve" => self.control_resolve_deep_link(params),
            "deep_links.filters.get" => self.control_get_deep_link_filters(params),
            "deep_links.generate" => self.control_generate_deep_link(params),
            "deep_links.apply" => self.control_apply_deep_link(params),
            "datasets.open_tiff" => self.control_open_tiff(ctx, params),
            "datasets.open_mosaic_samplesheet" => self.control_open_mosaic_samplesheet(ctx, params),
            "project.rois.open" => self.control_open_roi(params),
            "project.save" => self.control_save_project(),
            "project.views.list" => self.control_list_project_views(),
            "project.views.get" => self.control_get_project_view(params),
            "project.views.create" => self.control_create_project_view(params),
            "project.views.capture" => self.control_capture_project_view(params),
            "project.views.rename" => self.control_rename_project_view(params),
            "project.views.delete" => self.control_delete_project_view(params),
            "project.views.apply" => self.control_apply_project_view(params),
            "viewer.channels.get_contrast" => self.control_get_channel_contrast(params),
            "viewer.channels.set_contrast" => self.control_set_channel_contrast(params),
            "viewer.objects.get_visibility" => self.control_get_object_overlay_visibility(params),
            "viewer.objects.set_visibility" => self.control_set_object_overlay_visibility(params),
            "viewer.objects.get_state" => self.control_get_object_state(params),
            "viewer.objects.source.load" => self.control_load_object_source(params),
            "viewer.objects.source.reload" => self.control_reload_object_source(),
            "viewer.objects.source.clear" => self.control_clear_object_source(),
            "viewer.objects.source.cancel_load" => self.control_cancel_object_source_load(),
            "viewer.objects.style.get" => self.control_get_object_style(params),
            "viewer.objects.style.set" => self.control_set_object_style(params),
            "viewer.objects.legend.set" => self.control_set_object_legend(params),
            "viewer.objects.rendering.get_fast" => self.control_get_fast_object_rendering(params),
            "viewer.objects.rendering.set_fast" => self.control_set_fast_object_rendering(params),
            "viewer.objects.properties.list" => self.control_list_object_properties(params),
            "viewer.objects.properties.load" => self.control_load_object_property(params),
            "viewer.objects.properties.values" => self.control_get_object_property_values(params),
            "viewer.objects.get_selection" => self.control_get_object_selection(params),
            "viewer.objects.query_rect" => self.control_query_object_ids_in_rect(params),
            "viewer.objects.query_view" => self.control_query_object_ids_in_view(params),
            "viewer.objects.query_lasso" => self.control_query_object_ids_in_lasso(params),
            "viewer.objects.select_rect" => self.control_select_object_ids_in_rect(params),
            "viewer.objects.select_lasso" => self.control_select_object_ids_in_lasso(params),
            "viewer.objects.clear_selection" => self.control_clear_object_selection(params),
            "viewer.objects.selection.select_ids" => self.control_select_object_ids(params),
            "viewer.objects.selection.select_filtered" => {
                self.control_select_filtered_objects(params)
            }
            "viewer.objects.focus.set" => self.control_focus_object(params),
            "viewer.objects.focus.clear" => self.control_clear_object_focus(params),
            "viewer.masks.layers.list" => self.control_list_mask_layers(),
            "viewer.masks.layers.get" => self.control_get_mask_layer(params),
            "viewer.masks.layers.create" => self.control_create_mask_layer(params),
            "viewer.masks.layers.update" => self.control_update_mask_layer(params),
            "viewer.masks.layers.delete" => self.control_delete_mask_layer(params),
            "viewer.masks.polygons.list" => self.control_list_mask_polygons(params),
            "viewer.masks.polygons.add" => self.control_add_mask_polygon(params),
            "viewer.masks.polygons.update" => self.control_update_mask_polygon(params),
            "viewer.masks.polygons.remove" => self.control_remove_mask_polygon(params),
            "viewer.masks.selection.get" => self.control_get_mask_selection(),
            "viewer.masks.selection.set" => self.control_set_mask_selection(params),
            "viewer.masks.selection.clear" => self.control_clear_mask_selection(),
            "viewer.masks.undo" => self.control_undo_mask_edit(),
            "viewer.masks.import_geojson" => self.control_import_masks_geojson(params),
            "viewer.masks.export_geojson" => self.control_export_masks_geojson(params),
            "viewer.masks.persistence.get" => self.control_mask_persistence(),
            "viewer.masks.persistence.sync" => self.control_sync_masks_to_project(),
            "viewer.labels.get" | "viewer.labels.list" => self.control_get_labels(),
            "viewer.labels.load" => self.control_load_labels(params),
            "viewer.labels.unload" => self.control_unload_labels(),
            "viewer.labels.set_visibility" => self.control_set_labels_visibility(params),
            "viewer.thresholds.levels.list" => self.control_threshold_levels(),
            "viewer.thresholds.preview.get" => self.control_get_threshold_preview(),
            "viewer.thresholds.preview.configure" => {
                self.control_configure_threshold_preview(ctx, params)
            }
            "viewer.thresholds.preview.start" => self.control_start_threshold_preview(ctx, params),
            "viewer.thresholds.preview.refresh" => self.control_refresh_threshold_preview(ctx),
            "viewer.thresholds.preview.apply" => self.control_apply_threshold_preview(),
            "viewer.thresholds.preview.cancel" => self.control_cancel_threshold_preview(),
            "viewer.analysis.get" => self.control_get_object_analysis(params),
            "viewer.analysis.set" => self.control_set_object_analysis(params),
            "viewer.analysis.histogram" => self.control_object_histogram(params),
            "viewer.analysis.suggest_thresholds" => {
                self.control_object_threshold_suggestions(params)
            }
            "viewer.analysis.warmup.get" => self.control_get_analysis_warmup(params),
            "viewer.analysis.warmup.start" => self.control_start_analysis_warmup(params),
            "viewer.analysis.presets.import" => self.control_import_analysis_preset(params),
            "viewer.analysis.presets.export" => self.control_export_analysis_preset(params),
            "viewer.measurements.get" => self.control_get_measurement_state(params),
            "viewer.measurements.configure" => self.control_configure_measurement(params),
            "viewer.measurements.start" => self.control_start_measurement(params),
            "viewer.measurements.cancel" => self.control_cancel_measurement(params),
            "viewer.measurements.properties.list" => self.control_get_measurement_state(params),
            "exports.objects.columns" => self.control_get_object_export_columns(params),
            "exports.objects.get_state" => self.control_get_object_export_state(params),
            "exports.objects.start" => self.control_start_object_export(params),
            "exports.objects.export_csv" => self.control_start_typed_object_export(params, "csv"),
            "exports.objects.export_geoparquet" => {
                self.control_start_typed_object_export(params, "geoparquet")
            }
            "viewer.objects.get_filter" => self.control_get_object_filter(params),
            "viewer.objects.set_filter" => self.control_set_object_filter_query(params),
            "viewer.objects.filters.set_model" => self.control_set_object_filter_model(params),
            "viewer.objects.filters.get_revision" => self.control_get_object_filter(params),
            "viewer.objects.clear_filter" => self.control_clear_object_filter(params),
            "viewer.channels.intensity_stats" => self.control_get_channel_intensity_stats(params),
            "viewer.channels.set_order" => self.control_set_channel_order(params),
            "viewer.channels.presentation.get" => self.control_get_channel_presentation(),
            "viewer.channels.presentation.set" => self.control_set_channel_presentation(params),
            "viewer.channels.list_groups" => self.control_list_channel_groups(),
            "viewer.channels.set_group" => self.control_set_channel_group(params),
            "viewer.camera.get" => self.control_get_camera(),
            "viewer.camera.set" => self.control_set_camera(params),
            "viewer.camera.zoom_in" => self.control_zoom(params, true),
            "viewer.camera.zoom_out" => self.control_zoom(params, false),
            "viewer.camera.fit" => self.control_fit_to_view(),
            "viewer.ui.set_right_tab" => self.control_set_right_tab(params),
            "mosaic.ui.set_right_tab" => self.control_set_mosaic_right_tab(params),
            "mosaic.layout.configure" => self.control_configure_mosaic_layout(params),
            "mosaic.get_state" => self.control_get_mosaic_state(),
            "mosaic.items.list" => self.control_list_mosaic_items(params),
            "mosaic.selection.get" => self.control_get_mosaic_selection(),
            "mosaic.selection.set" => self.control_set_mosaic_selection(params),
            "mosaic.selection.clear" => self.control_clear_mosaic_selection(),
            "mosaic.focus.get" => self.control_get_mosaic_focus(),
            "mosaic.focus.set" => self.control_set_mosaic_focus(params),
            "mosaic.focus.next" => self.control_step_mosaic_focus(params, true),
            "mosaic.focus.previous" => self.control_step_mosaic_focus(params, false),
            "mosaic.focus.fit" => self.control_fit_mosaic_focus(),
            "mosaic.focus.clear" => self.control_clear_mosaic_focus(),
            "mosaic.fit_all" => self.control_fit_all_mosaic(),
            "mosaic.objects.get_state" => self.control_get_mosaic_object_state(),
            "mosaic.objects.load_selected" => self.control_load_selected_mosaic_objects(),
            "mosaic.objects.cancel_load" => self.control_cancel_mosaic_object_load(),
            "viewer.screenshot.capture" => self.control_capture_screenshot(params),
            "viewer.screenshot.settings.get" => self.control_get_screenshot_settings(),
            "viewer.screenshot.settings.set" => self.control_set_screenshot_settings(params),
            "app.settings.get" => self.control_get_app_settings(),
            "app.settings.set" => self.control_set_app_settings(params),
            "app.recent_projects.list" => self.control_list_recent_projects(),
            "app.recent_projects.forget" => self.control_forget_recent_project(params),
            "app.recent_projects.clear" => self.control_clear_recent_projects(),
            "viewer.scale_bar.get" => self.control_get_scale_bar(),
            "viewer.scale_bar.set" => self.control_set_scale_bar(params),
            "app.lifecycle.get" => self.control_get_lifecycle(),
            "app.lifecycle.request_close" => self.control_request_close(ctx, params, false),
            "app.lifecycle.request_quit" => self.control_request_close(ctx, params, true),
            "memory.get" => self.control_get_memory(),
            "memory.pin" => self.control_pin_memory(params),
            "memory.unpin" => self.control_unpin_memory(params),
            "memory.unpin_all" => self.control_unpin_all_memory(),
            "memory.tiles.get" => self.control_get_tile_loading(),
            "memory.tiles.set" => self.control_set_tile_loading(params),
            "app.screenshot.capture" => self.control_capture_window_screenshot(ctx, params),
            "viewer.workspace.screenshot.capture" => {
                self.control_capture_workspace_screenshot(ctx, params)
            }
            "project.screenshot.capture" => self.control_capture_project_screenshot(ctx, params),
            "app.navigation.show_project" => self.control_show_project_page(),
            method => unreachable!("control registry admitted unknown method {method}"),
        };
        if let Some(error) = control_application_error(method, &response) {
            let _ = request.reply.send(Err(error));
            return;
        }
        let revision = if mutates {
            self.control_mutated_this_frame = true;
            request.event_hub.next_revision()
        } else {
            request.event_hub.revision()
        };
        if let Some(object) = response.as_object_mut() {
            object.insert(
                "_control".to_string(),
                serde_json::json!({"revision": revision}),
            );
        }
        let event_data = response.clone();
        let _ = request.reply.send(Ok(response));
        if mutates {
            let event_source = params
                .get("viewport_id")
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    event_data
                        .get("viewport_id")
                        .and_then(serde_json::Value::as_str)
                })
                .map(|viewport_id| format!("viewport:{viewport_id}"))
                .unwrap_or_else(|| control_event_source(method).to_string());
            let primary_event = request
                .command
                .event_name()
                .unwrap_or_else(|| control_event_name(method));
            request.event_hub.publish(
                primary_event,
                &event_source,
                revision,
                serde_json::json!({"method": method, "result": event_data.clone()}),
                Some(request.session_id.clone()),
                request.request_id.clone(),
            );
            if event_data
                .get("active_viewport_changed")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                && let Some(legacy_event) = active_viewport_compatibility_event(method)
                && legacy_event != primary_event
            {
                request.event_hub.publish(
                    legacy_event,
                    "viewer:active",
                    revision,
                    serde_json::json!({
                        "method": method,
                        "result": event_data,
                        "caused_by_event": primary_event,
                    }),
                    Some(request.session_id),
                    request.request_id,
                );
            }
        }
    }

    fn current_project_space(&self) -> Option<&ProjectSpace> {
        match &self.mode {
            Mode::Project { project_space } => Some(project_space),
            Mode::Single(app) => Some(app.project_space()),
            Mode::Mosaic { mosaic, .. } => Some(mosaic.project_space()),
            Mode::Transition => None,
        }
    }

    fn control_method_availability(&self, params: &serde_json::Value) -> serde_json::Value {
        let mode = match &self.mode {
            Mode::Project { .. } => "project",
            Mode::Single(_) => "single",
            Mode::Mosaic { .. } => "mosaic",
            Mode::Transition => "transition",
        };
        let requested = params
            .get("methods")
            .and_then(serde_json::Value::as_array)
            .map(|methods| {
                methods
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            });
        odon::control::registry::availability_catalog(mode, requested.as_deref())
    }

    fn current_project_space_mut(&mut self) -> Option<&mut ProjectSpace> {
        match &mut self.mode {
            Mode::Project { project_space } => Some(project_space),
            Mode::Single(app) => Some(app.project_space_mut()),
            Mode::Mosaic { mosaic, .. } => Some(mosaic.project_space_mut()),
            Mode::Transition => None,
        }
    }

    fn take_current_project_space(&mut self) -> Result<ProjectSpace, &'static str> {
        match std::mem::replace(&mut self.mode, Mode::Transition) {
            Mode::Project { project_space } => Ok(project_space),
            Mode::Single(mut app) => Ok(app.take_project_space()),
            Mode::Mosaic { mut mosaic, .. } => Ok(mosaic.take_project_space()),
            Mode::Transition => Err("Odon is currently transitioning between views."),
        }
    }

    fn control_project_view_index(
        project_space: &ProjectSpace,
        params: &serde_json::Value,
    ) -> Result<usize, String> {
        if let Some(index) = params.get("index").and_then(serde_json::Value::as_u64) {
            let index = index as usize;
            return (index < project_space.view_presets().len())
                .then_some(index)
                .ok_or_else(|| format!("view preset index {index} is out of range"));
        }
        if let Some(name) = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            return project_space
                .view_presets()
                .iter()
                .position(|preset| preset.name == name)
                .ok_or_else(|| format!("view preset '{name}' was not found"));
        }
        Err("provide a view preset index or name".to_string())
    }

    fn control_list_project_views(&self) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        serde_json::json!({
            "views": project_space
                .view_presets()
                .iter()
                .enumerate()
                .map(|(index, preset)| serde_json::json!({
                    "index": index,
                    "name": preset.name,
                    "description": preset.description,
                    "spec": preset.spec,
                }))
                .collect::<Vec<_>>(),
        })
    }

    fn control_get_project_view(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let index = match Self::control_project_view_index(project_space, params) {
            Ok(index) => index,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let preset = &project_space.view_presets()[index];
        serde_json::json!({
            "index": index,
            "name": preset.name,
            "description": preset.description,
            "spec": preset.spec,
        })
    }

    fn control_create_project_view(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(name) = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
        else {
            return serde_json::json!({"error": "name is required"});
        };
        let spec = match params.get("spec") {
            Some(value) => match serde_json::from_value::<ProjectViewSpec>(value.clone()) {
                Ok(spec) => spec,
                Err(error) => {
                    return serde_json::json!({"error": format!("invalid view spec: {error}")});
                }
            },
            None => ProjectViewSpec::default(),
        };
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        project_space.save_view_preset(name.to_string(), spec);
        let index = project_space
            .view_presets()
            .iter()
            .position(|preset| preset.name == name)
            .unwrap_or_default();
        self.control_get_project_view(&serde_json::json!({"index": index}))
    }

    fn control_capture_project_view(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(name) = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
        else {
            return serde_json::json!({"error": "name is required"});
        };
        let Mode::Single(app) = &mut self.mode else {
            return serde_json::json!({"error": "view capture is available in single-image mode"});
        };
        let spec = if params.get("viewport_id").is_some() {
            match app.control_project_view_spec_for_viewport(params) {
                Ok(spec) => spec,
                Err(error) => return serde_json::json!({"error": error}),
            }
        } else {
            app.control_current_project_view_spec()
        };
        app.project_space_mut()
            .save_view_preset(name.to_string(), spec);
        let index = app
            .project_space()
            .view_presets()
            .iter()
            .position(|preset| preset.name == name)
            .unwrap_or_default();
        let preset = &app.project_space().view_presets()[index];
        serde_json::json!({
            "captured": true,
            "viewport_id": params.get("viewport_id").cloned(),
            "view": {"index": index, "name": preset.name, "description": preset.description, "spec": preset.spec},
        })
    }

    fn control_rename_project_view(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let index = match Self::control_project_view_index(project_space, params) {
            Ok(index) => index,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(new_name) = params.get("new_name").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "new_name is required"});
        };
        if let Err(error) = project_space.rename_view_preset(index, new_name.to_string()) {
            return serde_json::json!({"error": error});
        }
        self.control_get_project_view(&serde_json::json!({"index": index}))
    }

    fn control_delete_project_view(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let index = match Self::control_project_view_index(project_space, params) {
            Ok(index) => index,
            Err(error) => return serde_json::json!({"error": error}),
        };
        match project_space.delete_view_preset(index) {
            Ok(preset) => serde_json::json!({
                "deleted": true,
                "index": index,
                "name": preset.name,
            }),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    fn control_apply_project_view(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Mode::Single(app) = &mut self.mode else {
            return serde_json::json!({"error": "saved views can be applied in single-image mode"});
        };
        let index = match Self::control_project_view_index(app.project_space(), params) {
            Ok(index) => index,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let preset = app.project_space().view_presets()[index].clone();
        app.control_apply_project_view_spec(&preset.spec);
        serde_json::json!({
            "applied": true,
            "view": {"index": index, "name": preset.name, "description": preset.description, "spec": preset.spec},
        })
    }

    fn sync_control_manifest_to_project(&mut self) {
        let Some((revision, resources, layers)) = self.control_bridge.as_ref().map(|bridge| {
            let (resources, layers) = bridge.project_control_manifest();
            (bridge.revision(), resources, layers)
        }) else {
            return;
        };
        if revision == self.control_project_revision {
            return;
        }
        if let Some(project_space) = self.current_project_space_mut() {
            project_space.config_mut().control_resources = resources;
            project_space.config_mut().control_layers = layers;
            self.control_project_revision = revision;
        }
    }

    fn load_control_manifest_from_project(&mut self) {
        let Some((resources, layers)) = self.current_project_space().map(|project_space| {
            (
                project_space.config().control_resources.clone(),
                project_space.config().control_layers.clone(),
            )
        }) else {
            return;
        };
        let result = self
            .control_bridge
            .as_ref()
            .map(|bridge| bridge.replace_project_control_manifest(&resources, &layers));
        if let Some(Err(error)) = result
            && let Some(project_space) = self.current_project_space_mut()
        {
            project_space.set_status(format!("Project external layer restore failed: {error}"));
        }
    }

    fn control_project_rois(&self) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"project": null, "rois": []});
        };
        let selected = project_space
            .selected_rois()
            .into_iter()
            .filter_map(|roi| roi.source_key())
            .collect::<HashSet<_>>();
        let focused = project_space.focused_roi().and_then(ProjectRoi::source_key);
        let rois = project_space
            .rois()
            .iter()
            .map(|roi| {
                let source_key = roi.source_key();
                serde_json::json!({
                    "id": roi.id,
                    "display_name": roi.display_name,
                    "dataset": roi.dataset,
                    "source_key": source_key,
                    "source": roi.source_display(),
                    "segmentation_path": roi.segpath.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "selected": source_key.as_ref().is_some_and(|key| selected.contains(key)),
                    "focused": source_key == focused,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "project_path": project_space
                .saved_project_path()
                .map(|path| path.to_string_lossy().to_string()),
            "roi_count": rois.len(),
            "rois": rois,
        })
    }

    fn control_get_project(&self) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let config = project_space.config();
        serde_json::json!({
            "path": project_space.saved_project_path().map(|path| path.to_string_lossy().to_string()),
            "config_generation": project_space.config_generation(),
            "roi_count": config.rois.len(),
            "view_count": project_space.view_presets().len(),
            "metadata": {
                "default_dataset": config.default_dataset,
                "secondary_dataset": config.secondary_dataset,
                "default_threshold_marker": config.default_threshold_marker,
                "mosaic_segmentation_search_roots": config.mosaic_segmentation_search_roots,
                "dataset_keys": config.datasets.keys().collect::<Vec<_>>(),
            },
        })
    }

    fn control_create_project(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let mut project_space = ProjectSpace::default();
        if let Some(default_dataset) = params
            .get("default_dataset")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            project_space.config_mut().default_dataset = Some(default_dataset.to_string());
        }
        self.mode = Mode::Project { project_space };
        serde_json::json!({"created": true, "project": self.control_get_project()})
    }

    fn control_save_project_as(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        else {
            return serde_json::json!({"error": "path is required"});
        };
        self.sync_control_manifest_to_project();
        let result = match &mut self.mode {
            Mode::Project { project_space } => project_space.save_to_file(&path),
            Mode::Single(app) => {
                let mut project_space = app.take_project_space();
                let result = project_space.save_to_file(&path);
                app.set_project_space(project_space);
                result
            }
            Mode::Mosaic { mosaic, .. } => {
                let mut project_space = mosaic.take_project_space();
                let result = project_space.save_to_file(&path);
                mosaic.set_project_space(project_space);
                result
            }
            Mode::Transition => {
                return serde_json::json!({"error": "Odon is currently transitioning between views."});
            }
        };
        match result {
            Ok(()) => serde_json::json!({"saved": true, "path": path.to_string_lossy()}),
            Err(error) => serde_json::json!({"error": format!("{error}")}),
        }
    }

    fn control_update_project_metadata(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let config = project_space.config_mut();
        if let Some(value) = params.get("default_dataset") {
            config.default_dataset = value.as_str().map(str::to_string);
        }
        if let Some(value) = params.get("secondary_dataset") {
            config.secondary_dataset = value.as_str().map(str::to_string);
        }
        if let Some(value) = params.get("default_threshold_marker") {
            config.default_threshold_marker = value.as_str().map(str::to_string);
        }
        if let Some(values) = params
            .get("mosaic_segmentation_search_roots")
            .and_then(serde_json::Value::as_array)
        {
            let roots = values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(PathBuf::from)
                .collect::<Vec<_>>();
            config.mosaic_segmentation_search_roots = roots;
        }
        project_space.mark_config_changed();
        serde_json::json!({"updated": true, "project": self.control_get_project()})
    }

    fn control_inspect_samplesheet(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        let sheet = match load_samplesheet_csv(&path) {
            Ok(sheet) => sheet,
            Err(error) => {
                return serde_json::json!({
                    "valid": false,
                    "path": path.to_string_lossy(),
                    "error": format!("failed to parse samplesheet: {error}"),
                });
            }
        };
        let offset = params
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let limit = params
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(200) as usize;
        let base_dir = path.parent();
        let mut seen = HashSet::new();
        let duplicate_ids = sheet
            .rows
            .iter()
            .filter_map(|row| (!seen.insert(row.id.clone())).then(|| row.id.clone()))
            .collect::<Vec<_>>();
        let missing_count = sheet
            .rows
            .iter()
            .filter(|row| {
                let resolved = if row.path.is_relative() {
                    base_dir
                        .map(|dir| dir.join(&row.path))
                        .unwrap_or_else(|| row.path.clone())
                } else {
                    row.path.clone()
                };
                !resolved.exists()
            })
            .count();
        let total = sheet.rows.len();
        let rows = sheet
            .rows
            .iter()
            .skip(offset)
            .take(limit)
            .map(|row| {
                let resolved = if row.path.is_relative() {
                    base_dir
                        .map(|dir| dir.join(&row.path))
                        .unwrap_or_else(|| row.path.clone())
                } else {
                    row.path.clone()
                };
                serde_json::json!({
                    "id": row.id,
                    "path": row.path.to_string_lossy(),
                    "resolved_path": resolved.to_string_lossy(),
                    "exists": resolved.exists(),
                    "kind": classify_local_dataset_path(&resolved).map(|kind| match kind {
                        LocalDatasetKind::OmeZarr => "ome_zarr",
                        LocalDatasetKind::Tiff => "tiff",
                        LocalDatasetKind::Xenium => "xenium",
                    }),
                    "metadata": row.meta,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "valid": duplicate_ids.is_empty(),
            "path": path.to_string_lossy(),
            "metadata_columns": sheet.meta_columns,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(rows.len()) < total,
            "missing_source_count": missing_count,
            "duplicate_ids": duplicate_ids,
            "rows": rows,
        })
    }

    fn control_import_samplesheet(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        if let Err(error) = project_space.import_rois_from_csv(&path) {
            return serde_json::json!({
                "error": format!("failed to import samplesheet: {error}"),
                "path": path.to_string_lossy(),
            });
        }
        serde_json::json!({
            "imported": true,
            "path": path.to_string_lossy(),
            "project": self.control_project_rois(),
        })
    }

    fn control_export_samplesheet(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        let overwrite = params
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if path.exists() && !overwrite {
            return serde_json::json!({
                "error": "destination exists; pass overwrite=true to replace it",
                "path": path.to_string_lossy(),
            });
        }
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        if let Err(error) = project_space.export_samplesheet_csv(&path) {
            return serde_json::json!({
                "error": format!("failed to export samplesheet: {error}"),
                "path": path.to_string_lossy(),
            });
        }
        serde_json::json!({
            "exported": true,
            "path": path.to_string_lossy(),
            "bytes": std::fs::metadata(&path).ok().map(|metadata| metadata.len()),
        })
    }

    fn control_add_discovery_root(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let before = project_space.rois().len();
        if let Err(error) = project_space.import_rois_from_root(&path) {
            return serde_json::json!({
                "error": format!("dataset discovery failed: {error}"),
                "path": path.to_string_lossy(),
            });
        }
        let added = project_space.rois().len().saturating_sub(before);
        serde_json::json!({
            "discovered": true,
            "root": path.to_string_lossy(),
            "added": added,
            "project": self.control_project_rois(),
        })
    }

    fn object_preload_mode_key(mode: ObjectPreloadMode) -> &'static str {
        match mode {
            ObjectPreloadMode::FullGeometry => "full_geometry",
            ObjectPreloadMode::CentroidPoints => "centroid_points",
        }
    }

    fn control_get_project_object_preload(&self) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let state = project_object_cache_ui_state(
            project_space,
            self.object_preload_cache.len(),
            self.object_preload_total,
            self.object_preload_done,
            self.object_preload_failed,
            self.object_preload_rx.is_some(),
            self.object_preload_settings,
        );
        serde_json::json!({
            "available_count": state.available_count,
            "on_disk_bytes": state.on_disk_bytes,
            "cached": state.cached,
            "total": state.total,
            "done": state.done,
            "failed": state.failed,
            "loading": state.loading,
            "settings": {
                "mode": Self::object_preload_mode_key(state.cached_settings.mode),
                "lazy_properties": state.cached_settings.lazy_properties,
            },
            "project_path": project_space.saved_project_path().map(|path| path.to_string_lossy().to_string()),
        })
    }

    fn control_list_project_object_preload_sources(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let paths = project_object_segmentation_paths(project_space);
        let offset = params
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let limit = params
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(200) as usize;
        let total = paths.len();
        let sources = paths
            .iter()
            .skip(offset)
            .take(limit)
            .map(|path| {
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "bytes": path.metadata().ok().map(|metadata| metadata.len()),
                    "cached": self.object_preload_cache.contains_key(&(path.clone(), self.object_preload_settings)),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(sources.len()) < total,
            "sources": sources,
        })
    }

    fn control_start_project_object_preload(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        if self.object_preload_rx.is_some() {
            return serde_json::json!({"error": "project object preload is already running"});
        }
        let mode = match params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("full_geometry")
        {
            "full_geometry" => ObjectPreloadMode::FullGeometry,
            "centroid_points" => ObjectPreloadMode::CentroidPoints,
            _ => return serde_json::json!({"error": "unknown object preload mode"}),
        };
        let settings = ObjectPreloadSettings {
            mode,
            lazy_properties: params
                .get("lazy_properties")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
        };
        let Some(project_space) = self.current_project_space().cloned() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        if project_space.saved_project_path().is_none() {
            return serde_json::json!({"error": "save the project before preloading object segmentations"});
        }
        if project_object_segmentation_paths(&project_space).is_empty() {
            return serde_json::json!({"error": "project has no preload-eligible Parquet or GeoParquet segmentation paths"});
        }
        self.start_project_object_preload(&project_space, settings);
        serde_json::json!({
            "started": self.object_preload_rx.is_some(),
            "preload": self.control_get_project_object_preload(),
        })
    }

    fn control_clear_project_object_preload(&mut self) -> serde_json::Value {
        let removed = self.object_preload_cache.len();
        let cancelled = self.object_preload_rx.is_some();
        self.clear_project_object_preload();
        serde_json::json!({
            "cleared": true,
            "removed": removed,
            "cancelled": cancelled,
            "preload": self.control_get_project_object_preload(),
        })
    }

    fn control_get_project_roi(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "id is required"});
        };
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let index = match project_space.roi_index_by_id(id) {
            Ok(index) => index,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let roi = &project_space.rois()[index];
        serde_json::json!({"index": index, "roi": roi})
    }

    fn control_roi_from_params(
        params: &serde_json::Value,
        existing: Option<&ProjectRoi>,
    ) -> Result<ProjectRoi, String> {
        let mut roi = existing.cloned().unwrap_or_default();
        if let Some(id) = params.get("id").and_then(serde_json::Value::as_str) {
            roi.id = id.to_string();
        }
        if let Some(value) = params.get("display_name") {
            roi.display_name = value.as_str().map(str::to_string);
        }
        if let Some(value) = params.get("dataset") {
            roi.dataset = value.as_str().map(str::to_string);
        }
        if let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from)
        {
            roi.set_dataset_source(DatasetSource::Local(path));
        }
        if let Some(value) = params.get("segmentation_path") {
            roi.segpath = value.as_str().map(PathBuf::from);
        }
        if let Some(metadata) = params.get("metadata") {
            let Some(metadata) = metadata.as_object() else {
                return Err("metadata must be an object of string values".to_string());
            };
            roi.meta = metadata
                .iter()
                .map(|(key, value)| {
                    value
                        .as_str()
                        .map(|value| (key.clone(), value.to_string()))
                        .ok_or_else(|| format!("metadata value '{key}' must be a string"))
                })
                .collect::<Result<HashMap<_, _>, _>>()?;
        }
        Ok(roi)
    }

    fn control_add_project_roi(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let roi = match Self::control_roi_from_params(params, None) {
            Ok(roi) => roi,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let index = match project_space.add_roi_record(roi) {
            Ok(index) => index,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let id = project_space.rois()[index].id.clone();
        self.control_get_project_roi(&serde_json::json!({"id": id}))
    }

    fn control_update_project_roi(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(target_id) = params
            .get("target_id")
            .or_else(|| params.get("id"))
            .and_then(serde_json::Value::as_str)
        else {
            return serde_json::json!({"error": "target_id is required"});
        };
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        let index = match project_space.roi_index_by_id(target_id) {
            Ok(index) => index,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let existing = project_space.rois()[index].clone();
        let patch = params.get("changes").unwrap_or(params);
        let roi = match Self::control_roi_from_params(patch, Some(&existing)) {
            Ok(roi) => roi,
            Err(error) => return serde_json::json!({"error": error}),
        };
        if let Err(error) = project_space.update_roi_record(target_id, roi) {
            return serde_json::json!({"error": error});
        }
        let updated_id = project_space.rois()[index].id.clone();
        self.control_get_project_roi(&serde_json::json!({"id": updated_id}))
    }

    fn control_remove_project_roi(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "id is required"});
        };
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        match project_space.remove_roi_by_id(id) {
            Ok(roi) => serde_json::json!({"removed": true, "roi": roi}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    fn control_reorder_project_rois(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(ids) = params.get("ids").and_then(serde_json::Value::as_array) else {
            return serde_json::json!({"error": "ids is required"});
        };
        let ids = ids
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        if let Err(error) = project_space.reorder_rois(&ids) {
            return serde_json::json!({"error": error});
        }
        self.control_project_rois()
    }

    fn control_get_project_roi_selection(&self) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        serde_json::json!({
            "focused": project_space.focused_roi().map(|roi| roi.id.clone()),
            "selected": project_space.selected_rois().into_iter().map(|roi| roi.id).collect::<Vec<_>>(),
        })
    }

    fn control_select_project_rois(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(values) = params.get("ids").and_then(serde_json::Value::as_array) else {
            return serde_json::json!({"error": "ids is required"});
        };
        let ids = values
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mode = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("replace");
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        if let Err(error) = project_space.select_roi_ids(&ids, mode) {
            return serde_json::json!({"error": error});
        }
        self.control_get_project_roi_selection()
    }

    fn control_focus_project_roi(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "id is required"});
        };
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        if let Err(error) = project_space.focus_roi_id(id) {
            return serde_json::json!({"error": error});
        }
        self.control_get_project_roi_selection()
    }

    fn control_step_project_roi(
        &mut self,
        params: &serde_json::Value,
        forward: bool,
    ) -> serde_json::Value {
        let step = params
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as i64;
        let wrap = params
            .get("wrap")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let Some(project_space) = self.current_project_space_mut() else {
            return serde_json::json!({"error": "Odon is currently transitioning between views."});
        };
        if let Err(error) = project_space.step_focused_roi(if forward { step } else { -step }, wrap)
        {
            return serde_json::json!({"error": error});
        }
        self.control_get_project_roi_selection()
    }

    fn control_open_selected_project_mosaic(&mut self, ctx: &egui::Context) -> serde_json::Value {
        let rois = match self.current_project_space() {
            Some(project_space) => project_space.selected_rois(),
            None => {
                return serde_json::json!({"error": "Odon is currently transitioning between views."});
            }
        };
        if rois.len() < 2 {
            return serde_json::json!({"error": "select at least two ROIs to open a mosaic"});
        }
        let prev = std::mem::replace(&mut self.mode, Mode::Transition);
        let project_space = match prev {
            Mode::Project { project_space } => project_space,
            Mode::Single(mut app) => app.take_project_space(),
            Mode::Mosaic { mut mosaic, .. } => mosaic.take_project_space(),
            Mode::Transition => {
                return serde_json::json!({"error": "Odon is currently transitioning between views."});
            }
        };
        let count = rois.len();
        self.open_mosaic_from_project(ctx, rois, project_space);
        if matches!(self.mode, Mode::Mosaic { .. }) {
            serde_json::json!({"opened": true, "mode": "mosaic", "roi_count": count})
        } else {
            serde_json::json!({"error": "failed to open selected project ROIs as a mosaic"})
        }
    }

    fn control_channels(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "channels": app.control_channel_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "channels": mosaic.control_channel_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "mode": "project",
                "channels": [],
                "note": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "mode": "transition",
                "channels": [],
            }),
        }
    }

    fn control_get_channel_presentation(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_channel_presentation_json(),
            Mode::Mosaic { mosaic, .. } => mosaic.control_channel_presentation_json(),
            Mode::Project { .. } => {
                serde_json::json!({"error": "channel presentation requires a dataset viewer"})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_channel_presentation(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_channel_presentation(params),
            Mode::Mosaic { mosaic, .. } => mosaic.control_set_channel_presentation(params),
            Mode::Project { .. } => {
                serde_json::json!({"error": "channel presentation requires a dataset viewer"})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_viewport_operation(
        &mut self,
        operation: &str,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => match operation {
                "workspace.get" => app.control_viewport_workspace_snapshot(),
                "workspace.layout.get" => app.control_get_viewport_layout(),
                "workspace.layout.set" => app.control_set_viewport_layout(params),
                "workspace.swap" => app.control_swap_viewports(),
                "viewports.get" => app.control_get_viewport(params),
                "viewports.create" => app.control_create_viewport(params),
                "viewports.rename" => app.control_rename_viewport(params),
                "viewports.remove" => app.control_remove_viewport(params),
                "viewports.set_active" => app.control_set_active_viewport(params),
                "viewport_links.set" => app.control_set_viewport_links(params),
                "viewport_links.get" => app.control_get_viewport_links(),
                "viewport_links.list" => app.control_list_viewport_link_groups(),
                "viewport_links.create" => app.control_create_viewport_link_group(params),
                "viewport_links.update" => app.control_update_viewport_link_group(params),
                "viewport_links.remove" => app.control_remove_viewport_link_group(params),
                "viewports.camera.get" => app.control_get_viewport_camera(params),
                "viewports.camera.set" => app.control_set_viewport_camera(params),
                "viewports.camera.fit" => app.control_fit_viewport_camera(params),
                "viewports.planes.get" => app.control_get_viewport_plane(params),
                "viewports.planes.set" => app.control_set_viewport_plane(params),
                "viewports.channels.get" => app.control_get_viewport_channels(params),
                "viewports.channels.set_visible" => app.control_set_viewport_channels(params),
                "viewports.channels.set_active" => app.control_set_viewport_active_channel(params),
                "viewports.channels.set_color" => app.control_set_viewport_channel_color(params),
                "viewports.channels.set_contrast" => {
                    app.control_set_viewport_channel_contrast(params)
                }
                "viewports.channels.set_order" => app.control_set_viewport_channel_order(params),
                "viewports.channels.list_groups" => app.control_get_viewport_channel_groups(params),
                "viewports.channels.set_group" => app.control_set_viewport_channel_group(params),
                "viewports.rendering.get" => app.control_get_viewport_rendering(params),
                "viewports.rendering.set" => app.control_set_viewport_rendering(params),
                "viewports.objects.style.get" => app.control_get_viewport_object_style(params),
                "viewports.objects.style.set" => app.control_set_viewport_object_style(params),
                "viewports.objects.legend.set" => app.control_set_viewport_object_legend(params),
                "viewports.objects.filter.get" => app.control_get_viewport_object_filter(params),
                "viewports.objects.filter.set" => app.control_set_viewport_object_filter(params),
                "viewports.objects.filter.clear" => {
                    app.control_clear_viewport_object_filter(params)
                }
                "viewports.layers.list" => app.control_get_viewport_layers(params),
                "viewports.layers.get" => app.control_get_viewport_layer(params),
                "viewports.layers.set" => app.control_set_viewport_layer(params),
                "viewports.layers.set_visibility" => {
                    app.control_set_viewport_layer_visibility(params)
                }
                "viewports.layers.set_order" => app.control_set_viewport_layer_order(params),
                "viewports.layers.set_active" => app.control_set_viewport_active_layer(params),
                _ => serde_json::json!({
                    "error": format!("unknown viewport operation '{operation}'"),
                }),
            },
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "multi-viewport workspaces are currently available in single-image mode",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_plane(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "plane": app.control_plane_snapshot(),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "plane navigation is available in single-image mode",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_plane(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_plane(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "plane navigation is available in single-image mode",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_step_plane(
        &mut self,
        params: &serde_json::Value,
        forward: bool,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_step_plane(params, forward),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "plane navigation is available in single-image mode",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_side_panels(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "panels": app.control_side_panels_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "panels": mosaic.control_side_panels_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_side_panels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_side_panels(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_side_panels(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_smooth_pixels(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "smooth_pixels": app.control_smooth_pixels_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "smooth_pixels": mosaic.control_smooth_pixels_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_rendering_state(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "gpu_available": self.gpu_available,
                "renderer": if self.gpu_available { "opengl" } else { "cpu" },
                "compositing": "additive",
                "smooth_pixels": app.control_smooth_pixels_snapshot(),
                "deterministic_capture": {"method": "viewer.screenshot.capture", "readiness": app.control_loading_state_snapshot()},
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "gpu_available": self.gpu_available,
                "renderer": "opengl",
                "compositing": "additive",
                "smooth_pixels": mosaic.control_smooth_pixels_snapshot(),
                "deterministic_capture": {"method": "viewer.screenshot.capture", "readiness": mosaic.control_loading_state_snapshot()},
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "rendering state requires a dataset viewer"})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_plane_operation_availability(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => {
                let plane = app.control_plane_snapshot();
                let xy = plane.get("mode").and_then(serde_json::Value::as_str) == Some("xy");
                let operation = |requires_xy: bool| {
                    serde_json::json!({
                        "available": !requires_xy || xy,
                        "reason": (requires_xy && !xy).then_some("operation requires the XY view plane"),
                    })
                };
                serde_json::json!({
                    "plane": plane,
                    "operations": {
                        "measurements": operation(true),
                        "memory_pin": operation(true),
                        "channel_max": operation(true),
                        "threshold_preview": operation(true),
                        "object_selection": operation(false),
                    }
                })
            }
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "explicit plane restrictions apply to single-image multidimensional viewing"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "open a dataset viewer to inspect plane restrictions"})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_labels(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_labels_json(),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "NGFF label-group control is currently available in single-image mode"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "open a dataset viewer to inspect labels"})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_load_labels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_load_labels(params),
            _ => serde_json::json!({"error": "loading NGFF labels requires single-image mode"}),
        }
    }

    fn control_unload_labels(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_unload_labels(),
            _ => serde_json::json!({"error": "unloading NGFF labels requires single-image mode"}),
        }
    }

    fn control_set_labels_visibility(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_labels_visibility(params),
            _ => serde_json::json!({"error": "NGFF label visibility requires single-image mode"}),
        }
    }

    fn control_set_smooth_pixels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_smooth_pixels(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_smooth_pixels(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_loading_state(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "pending_deep_link": self.pending_deep_link.is_some(),
                "loading": app.control_loading_state_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "pending_deep_link": self.pending_deep_link.is_some(),
                "loading": mosaic.control_loading_state_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "mode": "project",
                "busy": false,
                "pending_deep_link": self.pending_deep_link.is_some(),
                "note": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "mode": "transition",
                "busy": true,
                "pending_deep_link": self.pending_deep_link.is_some(),
                "reasons": ["transition"],
            }),
        }
    }

    fn control_get_channel_contrast(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "contrast": app.control_get_channel_contrast(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "contrast": mosaic.control_get_channel_contrast(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_channel_contrast(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "contrast": app.control_set_channel_contrast(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "contrast": mosaic.control_set_channel_contrast(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_channel_color(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_channel_color(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_channel_color(params),
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_channel_note(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_channel_note(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_channel_note(params),
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_channel_transform(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "transform": app.control_get_channel_transform(params),
            }),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "channel transforms are available in single-image mode"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_channel_transform(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_channel_transform(params),
            }),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "channel transforms are available in single-image mode"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_reset_channel_transform(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_reset_channel_transform(params),
            }),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "channel transforms are available in single-image mode"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_native_layers(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "layers": app.control_native_layer_snapshot_list(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "layers": mosaic.control_native_layer_snapshot_list(),
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_native_layer(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "layer": app.control_get_native_layer(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "layer": mosaic.control_get_native_layer(params),
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_active_native_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => {
                serde_json::json!({"mode": "single", "result": app.control_set_active_native_layer(params)})
            }
            Mode::Mosaic { mosaic, .. } => {
                serde_json::json!({"mode": "mosaic", "result": mosaic.control_set_active_native_layer(params)})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_native_layer_visibility(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => {
                serde_json::json!({"mode": "single", "result": app.control_set_native_layer_visibility(params)})
            }
            Mode::Mosaic { mosaic, .. } => {
                serde_json::json!({"mode": "mosaic", "result": mosaic.control_set_native_layer_visibility(params)})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_native_layer_order(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => {
                serde_json::json!({"mode": "single", "result": app.control_set_native_layer_order(params)})
            }
            Mode::Mosaic { mosaic, .. } => {
                serde_json::json!({"mode": "mosaic", "result": mosaic.control_set_native_layer_order(params)})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_native_layer_offset(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => {
                serde_json::json!({"mode": "single", "result": app.control_set_native_layer_offset(params)})
            }
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "native layer offsets are available in single-image mode"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_reset_native_layer_offset(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => {
                serde_json::json!({"mode": "single", "result": app.control_reset_native_layer_offset(params)})
            }
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "native layer offsets are available in single-image mode"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_object_overlay_visibility(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "overlay": app.control_get_object_overlay_visibility(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "overlay": mosaic.control_get_object_overlay_visibility(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_object_overlay_visibility(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "overlay": app.control_set_object_overlay_visibility(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "overlay": mosaic.control_set_object_overlay_visibility(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_object_state(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_get_object_state(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "Use mosaic.objects.get_state in mosaic mode.",
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_load_object_source(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_load_object_source(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Use mosaic object loading controls in mosaic mode."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_reload_object_source(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_reload_object_source(),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Use mosaic object loading controls in mosaic mode."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_clear_object_source(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_clear_object_source(),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Use mosaic object loading controls in mosaic mode."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_cancel_object_source_load(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_cancel_object_source_load(),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Use mosaic object loading controls in mosaic mode."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_object_style(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_get_object_style(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Mosaic-wide object styling is not available through this method."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_object_style(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_object_style(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Mosaic-wide object styling is not available through this method."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_object_legend(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_object_legend(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Mosaic-wide object styling is not available through this method."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_fast_object_rendering(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_get_fast_object_rendering(params),
            Mode::Mosaic { mosaic, .. } => mosaic.control_fast_object_rendering_snapshot(),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_fast_object_rendering(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_fast_object_rendering(params),
            Mode::Mosaic { mosaic, .. } => {
                let Some(enabled) = params.get("enabled").and_then(serde_json::Value::as_bool)
                else {
                    return serde_json::json!({"error": "enabled is required"});
                };
                mosaic.set_fast_object_rendering(enabled);
                serde_json::json!({"changed": true, "enabled": enabled, "mode": "mosaic"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_list_object_properties(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "properties": app.control_list_object_properties(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "Object properties currently require a single-image viewer.",
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_load_object_property(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "property": app.control_load_object_property(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "Object properties currently require a single-image viewer.",
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_object_property_values(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "values": app.control_get_object_property_values(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "Object properties currently require a single-image viewer.",
            }),
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_channel_intensity_stats(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "stats": app.control_get_channel_intensity_stats(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "mode": "mosaic",
                "error": "Channel intensity stats currently require a single-image viewer.",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_channel_order(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_channel_order(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_channel_order(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_list_channel_groups(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "groups": app.control_channel_groups_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "groups": mosaic.control_channel_groups_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "mode": "project",
                "groups": [],
                "note": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "mode": "transition",
                "groups": [],
            }),
        }
    }

    fn control_set_channel_group(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_channel_group(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_channel_group(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_object_selection(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_get_object_selection(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object rectangle selection MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_query_object_ids_in_rect(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_query_object_ids_in_rect(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object rectangle selection MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_query_object_ids_in_view(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_query_object_ids_in_view(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object viewport query MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_query_object_ids_in_lasso(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_query_object_ids_in_lasso(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Object selection requires a single-image viewer."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_select_object_ids_in_rect(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_select_object_ids_in_rect(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object rectangle selection MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_select_object_ids_in_lasso(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_select_object_ids_in_lasso(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Object selection requires a single-image viewer."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_clear_object_selection(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_clear_object_selection(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object rectangle selection MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_select_object_ids(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_select_object_ids(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Object selection requires a single-image viewer."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_select_filtered_objects(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_select_filtered_objects(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Object selection requires a single-image viewer."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_focus_object(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_focus_object(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Object focus requires a single-image viewer."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_clear_object_focus(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_clear_object_focus(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Object focus requires a single-image viewer."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "No dataset viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_list_mask_layers(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_list_mask_layers(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_mask_layer(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_get_mask_layer(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_create_mask_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_create_mask_layer(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_update_mask_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_update_mask_layer(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_delete_mask_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_delete_mask_layer(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_list_mask_polygons(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_list_mask_polygons(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_add_mask_polygon(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_add_mask_polygon(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_update_mask_polygon(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_update_mask_polygon(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_remove_mask_polygon(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_remove_mask_polygon(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_undo_mask_edit(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_undo_mask_edit(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_mask_selection(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_get_mask_selection(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_mask_selection(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_mask_selection(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_clear_mask_selection(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_clear_mask_selection(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_import_masks_geojson(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        match &mut self.mode {
            Mode::Single(app) => app.control_import_masks_geojson(&path, params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_export_masks_geojson(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        match &self.mode {
            Mode::Single(app) => app.control_export_masks_geojson(&path, params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask editing requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_mask_persistence(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_mask_persistence(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask persistence requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_sync_masks_to_project(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_sync_masks_to_project(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Mask persistence requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_threshold_levels(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_threshold_levels(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Thresholding requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_threshold_preview(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_threshold_preview_snapshot(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Thresholding requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_configure_threshold_preview(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_configure_threshold_preview(ctx, params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Thresholding requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_start_threshold_preview(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_start_threshold_preview(ctx, params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Thresholding requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_refresh_threshold_preview(&mut self, ctx: &egui::Context) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_refresh_threshold_preview(ctx),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Thresholding requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_apply_threshold_preview(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_apply_threshold_preview(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Thresholding requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_cancel_threshold_preview(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_cancel_threshold_preview(),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Thresholding requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_object_analysis(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_get_object_analysis(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_object_analysis(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_object_analysis(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_object_histogram(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_object_histogram(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_object_threshold_suggestions(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_object_threshold_suggestions(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_analysis_warmup(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_get_analysis_warmup(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_start_analysis_warmup(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_start_analysis_warmup(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_import_analysis_preset(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        match &mut self.mode {
            Mode::Single(app) => app.control_import_analysis_preset(params, &path),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_export_analysis_preset(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        match &mut self.mode {
            Mode::Single(app) => app.control_export_analysis_preset(params, &path),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object analysis requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_measurement_state(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_get_measurement_state(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Measurements require a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_configure_measurement(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_configure_measurement(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Measurements require a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_start_measurement(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_start_measurement(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Measurements require a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_cancel_measurement(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_cancel_measurement(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Measurements require a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_object_export_columns(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_get_object_export_columns(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object export requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_object_export_state(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_get_object_export_state(params),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object export requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_start_object_export(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        match &mut self.mode {
            Mode::Single(app) => app.control_start_object_export(params, path),
            Mode::Mosaic { .. } | Mode::Project { .. } => {
                serde_json::json!({"error": "Object export requires a single-image viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_start_typed_object_export(
        &mut self,
        params: &serde_json::Value,
        format: &str,
    ) -> serde_json::Value {
        let mut params = params.clone();
        if let Some(object) = params.as_object_mut() {
            object.insert("format".to_string(), serde_json::json!(format));
        }
        self.control_start_object_export(&params)
    }

    fn control_get_object_filter(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_get_object_filter(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object filter MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_object_filter_query(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_set_object_filter_query(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object filter MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_object_filter_model(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_set_object_filter_model(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object filters are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_clear_object_filter(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "objects": app.control_clear_object_filter(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "object filter MCP tools are available in single-image mode"
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_camera(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "camera": app.control_camera_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "camera": mosaic.control_camera_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_camera(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "camera": app.control_set_camera(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "camera": mosaic.control_set_camera(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_zoom(&mut self, params: &serde_json::Value, zoom_in: bool) -> serde_json::Value {
        let raw_factor = params
            .get("factor")
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(1.5);
        let factor = if zoom_in {
            raw_factor
        } else if raw_factor > 0.0 {
            1.0 / raw_factor
        } else {
            raw_factor
        };
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "camera": app.control_zoom(factor),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "camera": mosaic.control_zoom(factor),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_fit_to_view(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "camera": app.control_fit_to_view(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "camera": mosaic.control_fit_to_view(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_mosaic_right_tab(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "tab": mosaic.control_set_right_tab(params),
            }),
            Mode::Single(_) => serde_json::json!({
                "error": "set_mosaic_right_tab requires mosaic mode.",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No mosaic viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_configure_mosaic_layout(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "layout": mosaic.control_configure_layout(params),
            }),
            Mode::Single(_) => serde_json::json!({
                "error": "configure_mosaic_layout requires mosaic mode.",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No mosaic viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_mosaic_state(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "mosaic": mosaic.control_mosaic_snapshot(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_list_mosaic_items(&self, params: &serde_json::Value) -> serde_json::Value {
        match &self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_list_items(params),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_mosaic_selection(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "selection": mosaic.control_selection_snapshot(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_mosaic_selection(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "selection": mosaic.control_select_rois(params),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_clear_mosaic_selection(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "selection": mosaic.control_clear_selection(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_mosaic_focus(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "focused": mosaic.control_focus_snapshot(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_mosaic_focus(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_focused_roi(params),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_step_mosaic_focus(
        &mut self,
        params: &serde_json::Value,
        forward: bool,
    ) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_step_focused_roi(params, forward),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_fit_mosaic_focus(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_fit_focused_roi(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_clear_mosaic_focus(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_clear_focus(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_fit_all_mosaic(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_fit_all(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_mosaic_object_state(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "objects": mosaic.control_object_loading_snapshot(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_load_selected_mosaic_objects(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_load_selected_objects(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_cancel_mosaic_object_load(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_cancel_object_load(),
            }),
            Mode::Single(_) | Mode::Project { .. } => {
                serde_json::json!({"error": "No mosaic viewer is currently open."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_right_tab(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "tab": app.control_set_right_tab(params),
            }),
            Mode::Mosaic { .. } => serde_json::json!({
                "error": "set_right_tab is for single-image mode; use set_mosaic_right_tab in mosaic mode.",
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No single-image viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_get_screenshot_settings(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_screenshot_settings_json(),
            Mode::Mosaic { mosaic, .. } => mosaic.control_screenshot_settings_json(),
            Mode::Project { .. } => {
                serde_json::json!({"error": "Screenshot settings require a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_app_settings(&self) -> serde_json::Value {
        serde_json::json!({
            "auto_contrast": self.app_settings.auto_contrast,
            "fast_object_rendering": self.app_settings.fast_object_rendering,
            "settings_path": settings_file_path().ok().map(|path| path.to_string_lossy().into_owned()),
            "status": self.settings_status,
        })
    }

    fn control_set_app_settings(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let mut candidate = self.app_settings.clone();
        if let Some(value) = params.get("fast_object_rendering") {
            let Some(value) = value.as_bool() else {
                return serde_json::json!({"error": "fast_object_rendering must be a boolean"});
            };
            candidate.fast_object_rendering = value;
        }
        if let Some(value) = params.get("auto_contrast") {
            let Some(settings) = value.as_object() else {
                return serde_json::json!({"error": "auto_contrast must be an object"});
            };
            if let Some(value) = settings.get("enabled_on_open") {
                let Some(value) = value.as_bool() else {
                    return serde_json::json!({"error": "auto_contrast.enabled_on_open must be a boolean"});
                };
                candidate.auto_contrast.enabled_on_open = value;
            }
            if let Some(value) = settings.get("method") {
                candidate.auto_contrast.method = match value.as_str() {
                    Some("zero_to_p97") => AutoContrastMethod::ZeroToP97,
                    Some("p1_to_p99") => AutoContrastMethod::P1ToP99,
                    Some("zero_to_max") => AutoContrastMethod::ZeroToMax,
                    _ => {
                        return serde_json::json!({"error": "auto_contrast.method must be zero_to_p97, p1_to_p99, or zero_to_max"});
                    }
                };
            }
            for (key, target) in [
                (
                    "lower_percentile",
                    &mut candidate.auto_contrast.lower_percentile,
                ),
                (
                    "upper_percentile",
                    &mut candidate.auto_contrast.upper_percentile,
                ),
            ] {
                if let Some(value) = settings.get(key) {
                    let Some(value) = value.as_u64().and_then(|value| u8::try_from(value).ok())
                    else {
                        return serde_json::json!({"error": format!("auto_contrast.{key} must be an integer from 0 to 100")});
                    };
                    if value > 100 {
                        return serde_json::json!({"error": format!("auto_contrast.{key} must be an integer from 0 to 100")});
                    }
                    *target = value;
                }
            }
            if candidate.auto_contrast.lower_percentile >= candidate.auto_contrast.upper_percentile
            {
                return serde_json::json!({"error": "auto_contrast.lower_percentile must be less than upper_percentile"});
            }
        }
        let path = match candidate.save() {
            Ok(path) => path,
            Err(error) => {
                return serde_json::json!({"error": format!("settings save failed: {error}")});
            }
        };
        self.app_settings = candidate;
        self.settings_status = format!("Saved settings to {}.", path.display());
        self.apply_app_settings_to_mode();
        self.control_get_app_settings()
    }

    fn control_list_recent_projects(&self) -> serde_json::Value {
        serde_json::json!({
            "projects": self.app_settings.recent_projects.iter().map(|project| serde_json::json!({
                "path": project.path.to_string_lossy(),
                "display_name": project.display_name(),
                "last_opened_unix_ms": project.last_opened_unix_ms,
                "exists": project.path.exists(),
            })).collect::<Vec<_>>(),
        })
    }

    fn control_forget_recent_project(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|path| !path.is_empty())
        else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(path);
        let mut candidate = self.app_settings.clone();
        let forgotten = candidate.forget_recent_project(&path);
        if forgotten {
            if let Err(error) = candidate.save() {
                return serde_json::json!({"error": format!("settings save failed: {error}")});
            }
            self.app_settings = candidate;
        }
        serde_json::json!({"forgotten": forgotten, "path": path.to_string_lossy(), "remaining": self.app_settings.recent_projects.len()})
    }

    fn control_clear_recent_projects(&mut self) -> serde_json::Value {
        let mut candidate = self.app_settings.clone();
        let cleared = candidate.recent_projects.len();
        if candidate.clear_recent_projects() {
            if let Err(error) = candidate.save() {
                return serde_json::json!({"error": format!("settings save failed: {error}")});
            }
            self.app_settings = candidate;
        }
        serde_json::json!({"cleared": cleared})
    }

    fn control_get_scale_bar(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => {
                serde_json::json!({"visible": app.show_scale_bar(), "supported": true})
            }
            Mode::Mosaic { .. } => {
                serde_json::json!({"visible": false, "supported": false, "reason": "mosaic scale bars are not currently rendered"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"visible": self.view_show_scale_bar, "supported": false, "reason": "open a dataset viewer"})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_scale_bar(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(visible) = params.get("visible").and_then(serde_json::Value::as_bool) else {
            return serde_json::json!({"error": "visible must be a boolean"});
        };
        match &mut self.mode {
            Mode::Single(app) => {
                self.view_show_scale_bar = visible;
                app.set_show_scale_bar(visible);
                serde_json::json!({"visible": visible, "supported": true})
            }
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "mosaic scale bars are not currently rendered"})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "open a dataset viewer to set scale-bar visibility"})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_lifecycle(&self) -> serde_json::Value {
        let dirty = match &self.mode {
            Mode::Project { project_space } => project_space.has_unsaved_changes(),
            Mode::Single(app) => app.has_unsaved_changes(),
            Mode::Mosaic { mosaic, .. } => mosaic.has_unsaved_changes(),
            Mode::Transition => false,
        };
        serde_json::json!({
            "dirty": dirty,
            "project_path": self.current_project_space().and_then(ProjectSpace::saved_project_path).map(|path| path.to_string_lossy().into_owned()),
            "can_save": self.current_project_space().and_then(ProjectSpace::saved_project_path).is_some(),
            "mode": match self.mode { Mode::Project { .. } => "project", Mode::Single(_) => "single", Mode::Mosaic { .. } => "mosaic", Mode::Transition => "transition" },
        })
    }

    fn control_request_close(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
        quit: bool,
    ) -> serde_json::Value {
        let decision = params
            .get("save")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("prompt");
        if !matches!(decision, "prompt" | "save" | "discard") {
            return serde_json::json!({"error": "save must be prompt, save, or discard"});
        }
        let lifecycle = self.control_get_lifecycle();
        let dirty = lifecycle
            .get("dirty")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if dirty && decision == "prompt" {
            return serde_json::json!({"confirmation_required": true, "action": if quit { "quit" } else { "close" }, "lifecycle": lifecycle});
        }
        if dirty && decision == "save" {
            if let Mode::Single(app) = &mut self.mode
                && app.has_unsaved_mask_changes()
            {
                let synced = app.control_sync_masks_to_project();
                if synced.get("error").is_some() {
                    return synced;
                }
            }
            let saved = self.control_save_project();
            if saved.get("error").is_some() {
                return saved;
            }
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        serde_json::json!({"accepted": true, "action": if quit { "quit" } else { "close" }, "discarded": dirty && decision == "discard"})
    }

    fn control_set_screenshot_settings(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let mut normalized = params.clone();
        if let Some(raw_path) = params.get("output_dir").and_then(serde_json::Value::as_str) {
            let path = expand_control_path(raw_path);
            if let Some(object) = normalized.as_object_mut() {
                object.insert(
                    "output_dir".to_string(),
                    serde_json::json!(path.to_string_lossy()),
                );
            }
        }
        match &mut self.mode {
            Mode::Single(app) => app.control_set_screenshot_settings_json(&normalized),
            Mode::Mosaic { mosaic, .. } => mosaic.control_set_screenshot_settings_json(&normalized),
            Mode::Project { .. } => {
                serde_json::json!({"error": "Screenshot settings require a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_memory(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_memory_json(),
            Mode::Mosaic { mosaic, .. } => mosaic.control_memory_json(),
            Mode::Project { .. } => {
                serde_json::json!({"error": "Memory control requires a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_pin_memory(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_pin_memory_level(params),
            Mode::Mosaic { mosaic, .. } => mosaic.control_pin_memory(params),
            Mode::Project { .. } => {
                serde_json::json!({"error": "Memory control requires a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_unpin_memory(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_unpin_memory_level(params),
            Mode::Mosaic { mosaic, .. } => mosaic.control_unpin_memory(params),
            Mode::Project { .. } => {
                serde_json::json!({"error": "Memory control requires a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_unpin_all_memory(&mut self) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_unpin_all_memory(),
            Mode::Mosaic { mosaic, .. } => mosaic.control_unpin_all_memory(),
            Mode::Project { .. } => {
                serde_json::json!({"error": "Memory control requires a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_get_tile_loading(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => app.control_tile_loading_json(),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Tile-loader tuning is available in single-image mode."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "Tile-loader tuning requires a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_set_tile_loading(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => app.control_set_tile_loading_json(params),
            Mode::Mosaic { .. } => {
                serde_json::json!({"error": "Tile-loader tuning is available in single-image mode."})
            }
            Mode::Project { .. } => {
                serde_json::json!({"error": "Tile-loader tuning requires a dataset viewer."})
            }
            Mode::Transition => {
                serde_json::json!({"error": "Odon is currently transitioning between views."})
            }
        }
    }

    fn control_capture_screenshot(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let mut normalized = params.clone();
        if let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) {
            let path = expand_control_path(raw_path);
            let overwrite = params
                .get("overwrite")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            if path.exists() && !overwrite {
                return serde_json::json!({"error": "destination exists; pass overwrite=true to replace it"});
            }
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
                && let Err(error) = std::fs::create_dir_all(parent)
            {
                return serde_json::json!({"error": format!("failed to create screenshot directory: {error}")});
            }
            if let Some(object) = normalized.as_object_mut() {
                object.insert(
                    "path".to_string(),
                    serde_json::json!(path.to_string_lossy()),
                );
            }
        }
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "screenshot": app.control_capture_screenshot(&normalized),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "screenshot": mosaic.control_capture_screenshot(&normalized),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_capture_window_screenshot(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_capture_egui_screenshot(ctx, params, None, "capture_window_screenshot")
    }

    fn control_capture_workspace_screenshot(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let crop_rect = match &mut self.mode {
            Mode::Single(app) => app.workspace_canvas_rect(),
            Mode::Mosaic { .. } => {
                return serde_json::json!({
                    "error": "multi-viewport workspace screenshots are available in single-image mode"
                });
            }
            Mode::Project { .. } => {
                return serde_json::json!({"error": "No dataset viewer is currently open."});
            }
            Mode::Transition => {
                return serde_json::json!({"error": "Odon is currently transitioning between views."});
            }
        };
        let Some(crop_rect) = crop_rect else {
            return serde_json::json!({"error": "workspace canvases have not been laid out yet"});
        };
        let mut response = self.control_capture_egui_screenshot(
            ctx,
            params,
            Some(crop_rect),
            "capture_workspace_screenshot",
        );
        if let Some(object) = response.as_object_mut() {
            object.insert("scope".to_string(), serde_json::json!("workspace"));
        }
        response
    }

    fn control_capture_egui_screenshot(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
        crop_rect_points: Option<egui::Rect>,
        operation_name: &str,
    ) -> serde_json::Value {
        let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return serde_json::json!({
                "error": format!("{operation_name} requires path"),
            });
        };
        let path = expand_control_path(path);
        let overwrite = params
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if path.exists() && !overwrite {
            return serde_json::json!({
                "error": "destination exists; pass overwrite=true to replace it",
                "path": path.to_string_lossy(),
            });
        }
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            return serde_json::json!({
                "error": format!("Failed to create screenshot directory: {err}"),
                "path": path.to_string_lossy(),
            });
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
            ViewportScreenshotRequest {
                path: path.clone(),
                crop_rect_points,
            },
        )));
        serde_json::json!({
            "queued": true,
            "path": path.to_string_lossy(),
        })
    }

    fn control_capture_project_screenshot(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let project = self.control_show_project_page();
        if project.get("error").is_some() {
            return project;
        }
        let screenshot = self.control_capture_window_screenshot(ctx, params);
        if screenshot.get("error").is_some() {
            return screenshot;
        }
        serde_json::json!({
            "project": project,
            "screenshot": screenshot,
        })
    }

    fn handle_viewport_screenshot_events(&mut self, ctx: &egui::Context) {
        let events = ctx.input(|input| input.events.clone());
        for event in events {
            let egui::Event::Screenshot {
                user_data, image, ..
            } = event
            else {
                continue;
            };
            let Some(data) = user_data.data else {
                continue;
            };
            let Ok(request) = Arc::downcast::<ViewportScreenshotRequest>(data) else {
                continue;
            };
            let [width, _height] = image.size;
            let Some((x0, y0, x1, y1)) = screenshot_crop_bounds(
                image.size,
                request.crop_rect_points,
                ctx.pixels_per_point(),
            ) else {
                self.settings_status = "Screenshot crop was empty.".to_string();
                continue;
            };
            let capture_width = x1 - x0;
            let capture_height = y1 - y0;
            let mut rgba = Vec::with_capacity(
                capture_width
                    .saturating_mul(capture_height)
                    .saturating_mul(4),
            );
            for y in y0..y1 {
                for x in x0..x1 {
                    rgba.extend_from_slice(&image.pixels[y * width + x].to_array());
                }
            }
            let result = image::save_buffer(
                &request.path,
                &rgba,
                capture_width as u32,
                capture_height as u32,
                image::ColorType::Rgba8,
            );
            match result {
                Ok(()) => {
                    self.settings_status =
                        format!("Saved window screenshot to {}.", request.path.display());
                }
                Err(err) => {
                    self.settings_status = format!("Window screenshot failed: {err}");
                }
            }
        }
    }

    fn control_visible_channels(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "channels": app.control_visible_channel_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "channels": mosaic.control_visible_channel_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "mode": "project",
                "channels": [],
                "note": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "mode": "transition",
                "channels": [],
            }),
        }
    }

    fn control_active_channel(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "active_channel": app.control_active_channel_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "active_channel": mosaic.control_active_channel_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "mode": "project",
                "active_channel": null,
                "note": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "mode": "transition",
                "active_channel": null,
            }),
        }
    }

    fn control_set_active_channel(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_active_channel(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_active_channel(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_set_visible_channels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "result": app.control_set_visible_channels(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "result": mosaic.control_set_visible_channels(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_open_ome_zarr(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_open_local_dataset(
            params,
            "open_ome_zarr",
            LocalDatasetKind::OmeZarr,
            "local OME-Zarr dataset root or metadata file",
            "OME-Zarr",
        )
    }

    fn control_open_tiff(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        let Some(path) = normalize_local_dataset_path(&path) else {
            return serde_json::json!({"error": "path is not a local TIFF / OME-TIFF file"});
        };
        if classify_local_dataset_path(&path) != Some(LocalDatasetKind::Tiff) {
            return serde_json::json!({"error": "path is not a TIFF / OME-TIFF dataset"});
        }
        let z = params
            .get("z")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let t = params
            .get("t")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let mut app = match OmeZarrViewerApp::new_tiff_runtime_with_plane(
            ctx,
            self.gpu_available,
            path.clone(),
            z,
            t,
            self.app_settings.auto_contrast,
        ) {
            Ok(app) => app,
            Err(error) => {
                return serde_json::json!({
                    "error": format!("failed to open TIFF plane Z={z}, T={t}: {error}"),
                    "path": path.to_string_lossy(),
                });
            }
        };
        let project_space = match self.take_current_project_space() {
            Ok(project_space) => project_space,
            Err(error) => return serde_json::json!({"error": error}),
        };
        self.configure_single_app(&mut app);
        app.set_project_space(project_space);
        self.mode = Mode::Single(app);
        serde_json::json!({
            "opened": true,
            "mode": "single",
            "kind": "tiff",
            "path": path.to_string_lossy(),
            "plane": {"z": z, "t": t},
        })
    }

    fn control_inspect_dataset(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        if !path.exists() {
            return serde_json::json!({
                "error": "dataset path does not exist",
                "path": path.to_string_lossy(),
            });
        }
        let normalized = normalize_local_dataset_path(&path).unwrap_or_else(|| path.clone());

        if normalized.is_dir()
            && let Ok(discovery) = discover_spatialdata(&normalized)
        {
            let elements = |kind: &str, values: &[crate::spatialdata::SpatialDataElement]| {
                values
                    .iter()
                    .map(|element| serde_json::json!({
                        "kind": kind,
                        "name": element.name,
                        "path": element.rel_group.to_string_lossy(),
                        "parquet_path": element.rel_parquet.as_ref().map(|path| path.to_string_lossy().to_string()),
                        "transform": {
                            "scale": element.transform.scale,
                            "translation": element.transform.translation,
                        },
                        "feature_key": element.feature_key,
                    }))
                    .collect::<Vec<_>>()
            };
            let mut all = Vec::new();
            all.extend(elements("image", &discovery.images));
            all.extend(elements("label", &discovery.labels));
            all.extend(elements("points", &discovery.points));
            all.extend(elements("shape", &discovery.shapes));
            all.extend(elements("table", &discovery.tables));
            return serde_json::json!({
                "kind": "spatialdata",
                "path": discovery.root.to_string_lossy(),
                "can_open": !discovery.images.is_empty(),
                "elements": all,
            });
        }

        match classify_local_dataset_path(&normalized) {
            Some(LocalDatasetKind::OmeZarr) => match OmeZarrDataset::open_local(&normalized) {
                Ok((dataset, _)) => serde_json::json!({
                    "kind": "ome_zarr",
                    "path": normalized.to_string_lossy(),
                    "can_open": true,
                    "metadata": {
                        "name": dataset.multiscale.name,
                        "axes": dataset.multiscale.axes.iter().map(|axis| serde_json::json!({"name": axis.name, "unit": axis.unit})).collect::<Vec<_>>(),
                        "level_count": dataset.levels.len(),
                        "levels": dataset.levels.iter().map(|level| serde_json::json!({
                            "index": level.index,
                            "path": level.path,
                            "shape": level.shape,
                            "chunks": level.chunks,
                            "dtype": level.dtype,
                            "scale": level.scale,
                            "translation": level.translation,
                        })).collect::<Vec<_>>(),
                        "channels": dataset.channels.iter().map(|channel| serde_json::json!({
                            "index": channel.index,
                            "name": channel.name,
                            "color_rgb": channel.color_rgb,
                            "window": channel.window.map(|(min, max)| [min, max]),
                        })).collect::<Vec<_>>(),
                        "dimensions": {"ndim": dataset.dims.ndim, "c": dataset.dims.c, "z": dataset.dims.z, "y": dataset.dims.y, "x": dataset.dims.x},
                        "absolute_max": dataset.abs_max,
                    },
                }),
                Err(error) => serde_json::json!({
                    "kind": "ome_zarr",
                    "path": normalized.to_string_lossy(),
                    "can_open": false,
                    "error": format!("failed to inspect OME-Zarr: {error}"),
                }),
            },
            Some(LocalDatasetKind::Tiff) => match TiffPyramid::open_with_selection(
                &normalized,
                TiffPlaneSelection { z: 0, t: 0 },
            ) {
                Ok(pyramid) => {
                    let channels = pyramid.default_channels_named("image");
                    serde_json::json!({
                        "kind": "tiff",
                        "path": normalized.to_string_lossy(),
                        "can_open": true,
                        "metadata": {
                            "file_size_bytes": std::fs::metadata(&normalized).ok().map(|metadata| metadata.len()),
                            "pixel_dtype": pyramid.pixel_dtype,
                            "absolute_max": pyramid.abs_max,
                            "channel_count": pyramid.channel_count,
                            "channels": channels.iter().map(|channel| serde_json::json!({
                                "index": channel.index,
                                "name": channel.name,
                                "color_rgb": channel.color_rgb,
                            })).collect::<Vec<_>>(),
                            "planes": {
                                "size_z": pyramid.size_z,
                                "size_t": pyramid.size_t,
                                "default": {"z": 0, "t": 0},
                            },
                            "levels": pyramid.levels.iter().enumerate().map(|(index, level)| serde_json::json!({
                                "index": index,
                                "width": level.width,
                                "height": level.height,
                                "chunk_width": level.chunk_w,
                                "chunk_height": level.chunk_h,
                                "tiles_x": level.tiles_x,
                                "tiles_y": level.tiles_y,
                                "channels": level.channels,
                                "channel_layout": format!("{:?}", level.channel_layout).to_ascii_lowercase(),
                            })).collect::<Vec<_>>(),
                            "ome": pyramid.ome.as_ref().map(|ome| serde_json::json!({
                                "dimension_order": ome.dimension_order,
                                "size_z": ome.size_z,
                                "size_t": ome.size_t,
                                "size_c": ome.size_c,
                                "physical_size_x": ome.physical_size_x,
                                "physical_size_x_unit": ome.physical_size_x_unit,
                                "physical_size_y": ome.physical_size_y,
                                "physical_size_y_unit": ome.physical_size_y_unit,
                                "channels": ome.channels.iter().map(|channel| serde_json::json!({
                                    "name": channel.name,
                                    "color_rgb": channel.color_rgb,
                                })).collect::<Vec<_>>(),
                            })),
                        },
                    })
                }
                Err(error) => serde_json::json!({
                    "kind": "tiff",
                    "path": normalized.to_string_lossy(),
                    "can_open": false,
                    "error": format!("failed to inspect TIFF: {error}"),
                }),
            },
            Some(LocalDatasetKind::Xenium) => match discover_xenium_explorer(&normalized) {
                Ok(discovery) => serde_json::json!({
                    "kind": "xenium",
                    "path": discovery.root.to_string_lossy(),
                    "can_open": discovery.morphology_mip_omezarr.is_some() || discovery.morphology_mip_tiff.is_some(),
                    "metadata": {
                        "pixel_size_um": discovery.pixel_size_um,
                        "morphology_mip_ome_zarr": discovery.morphology_mip_omezarr.map(|path| path.to_string_lossy().to_string()),
                        "morphology_mip_tiff": discovery.morphology_mip_tiff.map(|path| path.to_string_lossy().to_string()),
                        "transcripts_zarr_zip": discovery.transcripts_zarr_zip.map(|path| path.to_string_lossy().to_string()),
                        "cells_zarr_zip": discovery.cells_zarr_zip.map(|path| path.to_string_lossy().to_string()),
                    },
                }),
                Err(error) => serde_json::json!({
                    "kind": "xenium",
                    "path": normalized.to_string_lossy(),
                    "can_open": false,
                    "error": format!("failed to inspect Xenium dataset: {error}"),
                }),
            },
            None => serde_json::json!({
                "kind": "unsupported",
                "path": normalized.to_string_lossy(),
                "can_open": false,
                "error": "path is not a supported OME-Zarr, TIFF, SpatialData, or Xenium source",
            }),
        }
    }

    fn spatial_element(
        elements: &[SpatialDataElement],
        name: &str,
        kind: &str,
    ) -> Result<SpatialDataElement, String> {
        elements
            .iter()
            .find(|element| element.name == name)
            .cloned()
            .ok_or_else(|| format!("SpatialData {kind} element '{name}' was not found"))
    }

    fn control_open_spatialdata(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let Some(image_name) = params.get("image").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "image is required"});
        };
        let root = expand_control_path(raw_path);
        let discovery = match discover_spatialdata(&root) {
            Ok(discovery) => discovery,
            Err(error) => {
                return serde_json::json!({
                    "error": format!("failed to discover SpatialData elements: {error}"),
                    "path": root.to_string_lossy(),
                });
            }
        };
        let image = match Self::spatial_element(&discovery.images, image_name, "image") {
            Ok(element) => element,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let resolve_many = |key: &str,
                            elements: &[SpatialDataElement],
                            kind: &str|
         -> Result<Vec<SpatialDataElement>, String> {
            params
                .get(key)
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_str)
                .map(|name| Self::spatial_element(elements, name, kind))
                .collect()
        };
        let extra_images = match resolve_many("extra_images", &discovery.images, "image") {
            Ok(elements) => elements
                .into_iter()
                .filter(|element| element.name != image.name)
                .collect::<Vec<_>>(),
            Err(error) => return serde_json::json!({"error": error}),
        };
        let shapes = match resolve_many("shapes", &discovery.shapes, "shape") {
            Ok(elements) => elements,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let labels = match params.get("labels").and_then(serde_json::Value::as_str) {
            Some(name) => match Self::spatial_element(&discovery.labels, name, "label") {
                Ok(element) => Some(element),
                Err(error) => return serde_json::json!({"error": error}),
            },
            None => None,
        };
        let points = match params.get("points").and_then(serde_json::Value::as_str) {
            Some(name) => match Self::spatial_element(&discovery.points, name, "points") {
                Ok(element) => Some(element),
                Err(error) => return serde_json::json!({"error": error}),
            },
            None => None,
        };
        let points_max = params
            .get("points_max")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(200_000) as usize;
        let project_space = match self.take_current_project_space() {
            Ok(project_space) => project_space,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let image_root = discovery.root.join(&image.rel_group);
        match OmeZarrDataset::open_local(&image_root) {
            Ok((dataset, store)) => {
                let mut app = OmeZarrViewerApp::new_runtime(
                    ctx,
                    self.gpu_available,
                    dataset,
                    store,
                    self.app_settings.auto_contrast,
                );
                self.configure_single_app(&mut app);
                app.set_project_space(project_space);
                app.attach_spatialdata_layers(
                    discovery.root.clone(),
                    image.transform,
                    extra_images.clone(),
                    labels.clone(),
                    discovery.tables.clone(),
                    shapes.clone(),
                    points.clone().map(|element| (element, points_max)),
                );
                if let Some(viewport) = ctx.input(|input| input.viewport().inner_rect) {
                    app.fit_to_viewport(viewport);
                }
                self.mode = Mode::Single(app);
                serde_json::json!({
                    "opened": true,
                    "mode": "single",
                    "kind": "spatialdata",
                    "path": discovery.root.to_string_lossy(),
                    "image": image.name,
                    "extra_images": extra_images.iter().map(|element| &element.name).collect::<Vec<_>>(),
                    "labels": labels.as_ref().map(|element| &element.name),
                    "shapes": shapes.iter().map(|element| &element.name).collect::<Vec<_>>(),
                    "points": points.as_ref().map(|element| &element.name),
                    "points_max": points_max,
                })
            }
            Err(error) => {
                self.mode = Mode::Project { project_space };
                serde_json::json!({
                    "error": format!("failed to open SpatialData image '{}': {error}", image.name),
                    "path": image_root.to_string_lossy(),
                })
            }
        }
    }

    fn control_open_xenium(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(raw_path) = params.get("path").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = expand_control_path(raw_path);
        let discovery = match discover_xenium_explorer(&path) {
            Ok(discovery) => discovery,
            Err(error) => {
                return serde_json::json!({
                    "error": format!("failed to discover Xenium experiment: {error}"),
                    "path": path.to_string_lossy(),
                });
            }
        };
        let imagery = params
            .get("imagery")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("auto");
        let selected = match imagery {
            "ome_zarr" => discovery
                .morphology_mip_omezarr
                .clone()
                .map(|path| ("ome_zarr", path)),
            "tiff" => discovery
                .morphology_mip_tiff
                .clone()
                .map(|path| ("tiff", path)),
            _ => discovery
                .morphology_mip_omezarr
                .clone()
                .map(|path| ("ome_zarr", path))
                .or_else(|| {
                    discovery
                        .morphology_mip_tiff
                        .clone()
                        .map(|path| ("tiff", path))
                }),
        };
        let Some((imagery_kind, imagery_path)) = selected else {
            return serde_json::json!({
                "error": format!("requested Xenium {imagery} imagery is unavailable"),
                "path": discovery.root.to_string_lossy(),
            });
        };
        let cells = params
            .get("load_cells")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true)
            .then(|| discovery.cells_zarr_zip.clone())
            .flatten();
        let transcripts = params
            .get("load_transcripts")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true)
            .then(|| discovery.transcripts_zarr_zip.clone())
            .flatten();
        let project_space = match self.take_current_project_space() {
            Ok(project_space) => project_space,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let app_result = if imagery_kind == "ome_zarr" {
            OmeZarrDataset::open_local(&imagery_path).map(|(dataset, store)| {
                let mut app = OmeZarrViewerApp::new_runtime(
                    ctx,
                    self.gpu_available,
                    dataset,
                    store,
                    self.app_settings.auto_contrast,
                );
                app.attach_xenium_layers(
                    discovery.root.clone(),
                    cells.clone(),
                    transcripts.clone(),
                    discovery.pixel_size_um,
                );
                app
            })
        } else {
            OmeZarrViewerApp::new_xenium_runtime(
                ctx,
                self.gpu_available,
                discovery.root.clone(),
                imagery_path.clone(),
                cells.clone(),
                transcripts.clone(),
                discovery.pixel_size_um,
                self.app_settings.auto_contrast,
            )
        };
        match app_result {
            Ok(mut app) => {
                self.configure_single_app(&mut app);
                app.set_project_space(project_space);
                self.mode = Mode::Single(app);
                serde_json::json!({
                    "opened": true,
                    "mode": "single",
                    "kind": "xenium",
                    "path": discovery.root.to_string_lossy(),
                    "imagery": imagery_kind,
                    "imagery_path": imagery_path.to_string_lossy(),
                    "cells_loaded": cells.is_some(),
                    "transcripts_loaded": transcripts.is_some(),
                    "pixel_size_um": discovery.pixel_size_um,
                })
            }
            Err(error) => {
                self.mode = Mode::Project { project_space };
                serde_json::json!({
                    "error": format!("failed to open Xenium {imagery_kind} imagery: {error}"),
                    "path": imagery_path.to_string_lossy(),
                })
            }
        }
    }

    fn control_open_http(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(url) = params
            .get("url")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|url| !url.is_empty())
        else {
            return serde_json::json!({"error": "url is required"});
        };
        let url = url.trim_end_matches('/').to_string();
        let store = match build_http_store(&url) {
            Ok(store) => store,
            Err(error) => {
                return serde_json::json!({"error": format!("invalid HTTP source: {error}")});
            }
        };
        let source = DatasetSource::Http {
            base_url: url.clone(),
        };
        let dataset = match OmeZarrDataset::open_with_store(source, store.clone()) {
            Ok(dataset) => dataset,
            Err(error) => {
                return serde_json::json!({
                    "error": format!("failed to open remote OME-Zarr: {error}"),
                    "url": url,
                });
            }
        };
        let project_space = match self.take_current_project_space() {
            Ok(project_space) => project_space,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let mut app = OmeZarrViewerApp::new_runtime(
            ctx,
            self.gpu_available,
            dataset,
            store,
            self.app_settings.auto_contrast,
        );
        self.configure_single_app(&mut app);
        app.set_project_space(project_space);
        self.mode = Mode::Single(app);
        serde_json::json!({"opened": true, "mode": "single", "kind": "http_ome_zarr", "url": url})
    }

    fn control_get_s3_session(&self) -> serde_json::Value {
        let configured = !self.remote_s3_endpoint.trim().is_empty()
            && !self.remote_s3_bucket.trim().is_empty()
            && !self.remote_s3_access_key.trim().is_empty()
            && !self.remote_s3_secret_key.trim().is_empty();
        serde_json::json!({
            "configured": configured,
            "endpoint": configured.then(|| self.remote_s3_endpoint.trim()),
            "region": configured.then(|| self.remote_s3_region.trim()),
            "bucket": configured.then(|| self.remote_s3_bucket.trim()),
            "credentials": if configured { "session_only_redacted" } else { "none" },
            "persisted": false,
        })
    }

    fn control_configure_s3_session(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let string = |name: &str| {
            params
                .get(name)
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        };
        let (Some(mut endpoint), Some(bucket), Some(access_key), Some(secret_key)) = (
            string("endpoint"),
            string("bucket"),
            string("access_key"),
            string("secret_key"),
        ) else {
            return serde_json::json!({"error": "endpoint, bucket, access_key, and secret_key are required"});
        };
        if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
            endpoint = format!("https://{endpoint}");
        }
        let region = string("region").unwrap_or_else(|| "auto".to_string());
        if let Err(error) = build_s3_browser(&endpoint, &region, &bucket, &access_key, &secret_key)
        {
            return serde_json::json!({"error": format!("invalid S3 session configuration: {error}")});
        }
        self.remote_s3_endpoint = endpoint;
        self.remote_s3_region = region;
        self.remote_s3_bucket = bucket;
        self.remote_s3_access_key = access_key;
        self.remote_s3_secret_key = secret_key;
        self.remote_s3_prefix.clear();
        self.clear_remote_s3_browser();
        self.control_get_s3_session()
    }

    fn control_clear_s3_session(&mut self) -> serde_json::Value {
        self.remote_s3_endpoint.clear();
        self.remote_s3_region = "auto".to_string();
        self.remote_s3_bucket.clear();
        self.remote_s3_prefix.clear();
        self.remote_s3_access_key.clear();
        self.remote_s3_secret_key.clear();
        self.clear_remote_s3_browser();
        serde_json::json!({"cleared": true, "configured": false, "persisted": false})
    }

    fn control_list_s3(&self, params: &serde_json::Value) -> serde_json::Value {
        if self.remote_s3_access_key.trim().is_empty()
            || self.remote_s3_secret_key.trim().is_empty()
        {
            return serde_json::json!({"error": "S3 session credentials are not configured"});
        }
        let prefix = params
            .get("prefix")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let browser = match build_s3_browser(
            &self.remote_s3_endpoint,
            &self.remote_s3_region,
            &self.remote_s3_bucket,
            &self.remote_s3_access_key,
            &self.remote_s3_secret_key,
        ) {
            Ok(browser) => browser,
            Err(error) => {
                return serde_json::json!({"error": format!("failed to connect to S3: {error}")});
            }
        };
        match list_s3_prefix(&browser, prefix) {
            Ok(listing) => serde_json::json!({
                "endpoint": self.remote_s3_endpoint,
                "region": self.remote_s3_region,
                "bucket": self.remote_s3_bucket,
                "prefix": listing.prefix,
                "parent_prefix": listing.parent_prefix,
                "current_is_dataset": listing.current_is_dataset,
                "entries": listing.entries.into_iter().map(|entry| serde_json::json!({
                    "name": entry.name,
                    "prefix": entry.prefix,
                    "is_dataset": entry.is_dataset,
                })).collect::<Vec<_>>(),
            }),
            Err(error) => {
                serde_json::json!({"error": format!("failed to list S3 prefix: {error}")})
            }
        }
    }

    fn control_open_s3(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        if self.remote_s3_access_key.trim().is_empty()
            || self.remote_s3_secret_key.trim().is_empty()
        {
            return serde_json::json!({"error": "S3 session credentials are not configured"});
        }
        let prefix = params
            .get("prefix")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .trim()
            .trim_matches('/')
            .to_string();
        let S3Store { store, runtime } = match build_s3_store(
            &self.remote_s3_endpoint,
            &self.remote_s3_region,
            &self.remote_s3_bucket,
            &prefix,
            &self.remote_s3_access_key,
            &self.remote_s3_secret_key,
        ) {
            Ok(store) => store,
            Err(error) => {
                return serde_json::json!({"error": format!("failed to connect to S3: {error}")});
            }
        };
        let source = DatasetSource::S3 {
            endpoint: self.remote_s3_endpoint.clone(),
            region: self.remote_s3_region.clone(),
            bucket: self.remote_s3_bucket.clone(),
            prefix: prefix.clone(),
        };
        let dataset = match OmeZarrDataset::open_with_store(source, store.clone()) {
            Ok(dataset) => dataset,
            Err(error) => {
                return serde_json::json!({"error": format!("failed to open S3 OME-Zarr: {error}")});
            }
        };
        let project_space = match self.take_current_project_space() {
            Ok(project_space) => project_space,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let mut app = OmeZarrViewerApp::new_runtime(
            ctx,
            self.gpu_available,
            dataset,
            store,
            self.app_settings.auto_contrast,
        );
        app.set_remote_runtime(Some(runtime));
        self.configure_single_app(&mut app);
        app.set_project_space(project_space);
        self.mode = Mode::Single(app);
        serde_json::json!({
            "opened": true,
            "mode": "single",
            "kind": "s3_ome_zarr",
            "endpoint": self.remote_s3_endpoint,
            "region": self.remote_s3_region,
            "bucket": self.remote_s3_bucket,
            "prefix": prefix,
        })
    }

    fn control_parse_deep_link(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(url) = params.get("url").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "url is required"});
        };
        match DeepLinkRequest::parse_arg(url) {
            Ok(Some(request)) => serde_json::json!({
                "valid": true,
                "url": request.to_url(),
                "request": request,
            }),
            Ok(None) => serde_json::json!({"error": "url must use the odon: scheme"}),
            Err(error) => serde_json::json!({"error": format!("invalid deep link: {error}")}),
        }
    }

    fn deep_link_request_from_params(
        &self,
        params: &serde_json::Value,
    ) -> Result<DeepLinkRequest, String> {
        if let Some(url) = params.get("url").and_then(serde_json::Value::as_str) {
            return match DeepLinkRequest::parse_arg(url) {
                Ok(Some(request)) => Ok(request),
                Ok(None) => Err("url must use the odon: scheme".to_string()),
                Err(error) => Err(format!("invalid deep link: {error}")),
            };
        }
        if let Some(value) = params.get("request") {
            return serde_json::from_value::<DeepLinkRequest>(value.clone())
                .map_err(|error| format!("invalid deep-link request: {error}"));
        }
        Err("url or request is required".to_string())
    }

    fn prepare_deep_link_request(
        &self,
        mut request: DeepLinkRequest,
    ) -> Result<(DeepLinkRequest, serde_json::Value), String> {
        if let Some(example) = request.example.clone() {
            apply_example_defaults(&mut request, &example);
            if request.project_path.is_none() {
                request.project_path = resolve_example_project_path(&example);
            }
        }
        if let Some(path) = request.project_path.as_deref() {
            let path = expand_control_path(&path.to_string_lossy());
            request.project_path = Some(path);
        }

        let current = self.current_project_space();
        let use_current = request.project_path.as_ref().is_none_or(|path| {
            current.and_then(ProjectSpace::saved_project_path).as_ref() == Some(path)
        });
        let (roi, project_source) = if use_current {
            let project = current.ok_or_else(|| "No project is currently loaded.".to_string())?;
            (
                project.roi_for_link_target(request.roi.as_deref(), request.sample.as_deref())?,
                "current",
            )
        } else {
            let path = request
                .project_path
                .as_deref()
                .ok_or_else(|| "Deep link does not identify a project.".to_string())?;
            if !path.exists() {
                return Err(format!(
                    "Deep-link project does not exist: {}",
                    path.to_string_lossy()
                ));
            }
            let mut project = ProjectSpace::default();
            project
                .load_from_file(path)
                .map_err(|error| format!("Deep-link project could not be loaded: {error}"))?;
            (
                project.roi_for_link_target(request.roi.as_deref(), request.sample.as_deref())?,
                "project_file",
            )
        };
        let resolution = serde_json::json!({
            "project_source": project_source,
            "project_path": request.project_path,
            "roi": roi,
        });
        Ok((request, resolution))
    }

    fn control_resolve_deep_link(&self, params: &serde_json::Value) -> serde_json::Value {
        let request = match self.deep_link_request_from_params(params) {
            Ok(request) => request,
            Err(error) => return serde_json::json!({"error": error}),
        };
        match self.prepare_deep_link_request(request) {
            Ok((request, resolution)) => serde_json::json!({
                "resolved": true,
                "url": request.to_url(),
                "request": request,
                "resolution": resolution,
            }),
            Err(error) => serde_json::json!({"resolved": false, "error": error}),
        }
    }

    fn control_get_deep_link_filters(&self, params: &serde_json::Value) -> serde_json::Value {
        let request = match self.deep_link_request_from_params(params) {
            Ok(request) => request,
            Err(error) => return serde_json::json!({"error": error}),
        };
        serde_json::json!({
            "object_filters": request.object_filters,
            "object_filter_logic": request.object_filter_logic,
            "object_query": request.object_query,
            "visible_cell_types": request.visible_cell_types,
            "hidden_cell_types": request.hidden_cell_types,
        })
    }

    fn control_generate_deep_link(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let explicit = params.get("request").is_some();
        let mut request = if let Some(value) = params.get("request") {
            match serde_json::from_value::<DeepLinkRequest>(value.clone()) {
                Ok(request) => request,
                Err(error) => {
                    return serde_json::json!({
                        "error": format!("invalid deep-link request: {error}")
                    });
                }
            }
        } else {
            match &mut self.mode {
                Mode::Single(app) => app
                    .control_current_project_view_spec()
                    .to_deep_link_request(None),
                Mode::Project { .. } | Mode::Mosaic { .. } => DeepLinkRequest::default(),
                Mode::Transition => {
                    return serde_json::json!({
                        "error": "Odon is currently transitioning between views."
                    });
                }
            }
        };

        if !explicit
            && params
                .get("include_project")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true)
        {
            request.project_path = self
                .current_project_space()
                .and_then(ProjectSpace::saved_project_path);
        }
        if params.get("roi").is_some() {
            request.roi = params
                .get("roi")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
        } else if !explicit {
            request.roi = self
                .current_project_space()
                .and_then(ProjectSpace::focused_roi)
                .map(|roi| roi.id.clone());
        }

        serde_json::json!({
            "url": request.to_url(),
            "request": request,
            "source": if explicit { "request" } else { "current_state" },
        })
    }

    fn control_apply_deep_link(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let request = match self.deep_link_request_from_params(params) {
            Ok(request) => request,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let (request, resolution) = match self.prepare_deep_link_request(request) {
            Ok(prepared) => prepared,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let url = request.to_url();
        self.pending_deep_link = Some(request.clone());
        serde_json::json!({
            "queued": true,
            "settled": false,
            "url": url,
            "request": request,
            "resolution": resolution,
            "note": "Application occurs during the next UI update and may initiate a longer load.",
        })
    }

    fn control_open_project(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return serde_json::json!({"error": "open_project requires path"});
        };
        let path = expand_control_path(path);
        if !path.exists() {
            return serde_json::json!({
                "error": "Project file does not exist.",
                "path": path.to_string_lossy(),
            });
        }
        self.load_project_into_current_mode(&path);
        serde_json::json!({
            "opened": true,
            "path": path.to_string_lossy(),
            "project": self.control_project_rois(),
        })
    }

    fn control_open_mosaic_samplesheet(
        &mut self,
        ctx: &egui::Context,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return serde_json::json!({"error": "open_mosaic_samplesheet requires path"});
        };
        let path = expand_control_path(path);
        let columns = params
            .get("columns")
            .or_else(|| params.get("cols"))
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .filter(|value| *value > 0);

        let prev = std::mem::replace(&mut self.mode, Mode::Transition);
        match MosaicViewerApp::from_samplesheet_runtime(ctx, self.gpu_available, &path, columns) {
            Ok(mut mosaic) => {
                self.configure_mosaic_app(&mut mosaic);
                let roi_count = mosaic.control_view_snapshot()["roi_count"].clone();
                self.mode = Mode::Mosaic {
                    mosaic,
                    ret: ReturnToSingleState { dataset_root: None },
                };
                serde_json::json!({
                    "mode": "mosaic",
                    "path": path.to_string_lossy(),
                    "roi_count": roi_count,
                })
            }
            Err(err) => {
                self.mode = prev;
                serde_json::json!({
                    "error": format!("Open mosaic samplesheet failed: {err}"),
                    "path": path.to_string_lossy(),
                })
            }
        }
    }

    fn control_open_local_dataset(
        &mut self,
        params: &serde_json::Value,
        tool_name: &str,
        expected_kind: LocalDatasetKind,
        input_label: &str,
        output_label: &str,
    ) -> serde_json::Value {
        let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return serde_json::json!({"error": format!("{tool_name} requires path")});
        };
        let path = expand_control_path(path);
        let Some(root) = normalize_local_dataset_path(&path) else {
            return serde_json::json!({
                "error": format!("path is not a {input_label}"),
                "path": path.to_string_lossy(),
            });
        };
        if classify_local_dataset_path(&root) != Some(expected_kind) {
            return serde_json::json!({
                "error": format!("path is not a {output_label} dataset"),
                "path": root.to_string_lossy(),
            });
        }
        self.pending_control_open_root = Some(root.clone());
        serde_json::json!({
            "queued": true,
            "path": root.to_string_lossy(),
            "note": format!("The {output_label} open request was queued and will replace the active viewer on the next UI update."),
        })
    }

    fn control_open_roi(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let roi = params
            .get("roi")
            .or_else(|| params.get("id"))
            .or_else(|| params.get("name"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(roi) = roi else {
            return serde_json::json!({"error": "open_roi requires roi, id, or name"});
        };
        let sample = params
            .get("sample")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"error": "No project is currently loaded."});
        };
        match project_space.roi_for_link_target(Some(roi), sample) {
            Ok(_) => {
                let mut request = DeepLinkRequest::default();
                request.roi = Some(roi.to_string());
                request.sample = sample.map(str::to_string);
                self.pending_deep_link = Some(request);
                serde_json::json!({
                    "queued": true,
                    "roi": roi,
                    "sample": sample,
                })
            }
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    fn control_save_project(&mut self) -> serde_json::Value {
        self.sync_control_manifest_to_project();
        match &mut self.mode {
            Mode::Project { project_space } => {
                let Some(path) = project_space.saved_project_path() else {
                    return serde_json::json!({"error": "Project has no saved path."});
                };
                match project_space.save_to_file(&path) {
                    Ok(()) => serde_json::json!({"saved": true, "path": path.to_string_lossy()}),
                    Err(err) => serde_json::json!({"error": format!("{err}")}),
                }
            }
            Mode::Single(app) => {
                let mut project_space = app.take_project_space();
                let Some(path) = project_space.saved_project_path() else {
                    app.set_project_space(project_space);
                    return serde_json::json!({"error": "Project has no saved path."});
                };
                let result = project_space.save_to_file(&path);
                app.set_project_space(project_space);
                match result {
                    Ok(()) => serde_json::json!({"saved": true, "path": path.to_string_lossy()}),
                    Err(err) => serde_json::json!({"error": format!("{err}")}),
                }
            }
            Mode::Mosaic { mosaic, .. } => {
                let mut project_space = mosaic.take_project_space();
                let Some(path) = project_space.saved_project_path() else {
                    mosaic.set_project_space(project_space);
                    return serde_json::json!({"error": "Project has no saved path."});
                };
                let result = project_space.save_to_file(&path);
                mosaic.set_project_space(project_space);
                match result {
                    Ok(()) => serde_json::json!({"saved": true, "path": path.to_string_lossy()}),
                    Err(err) => serde_json::json!({"error": format!("{err}")}),
                }
            }
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn control_show_project_page(&mut self) -> serde_json::Value {
        let prev = std::mem::replace(&mut self.mode, Mode::Transition);
        match prev {
            Mode::Project { project_space } => {
                self.mode = Mode::Project { project_space };
                serde_json::json!({
                    "mode": "project",
                    "changed": false,
                })
            }
            Mode::Single(mut app) => {
                let project_space = app.take_project_space();
                self.mode = Mode::Project { project_space };
                serde_json::json!({
                    "mode": "project",
                    "changed": true,
                })
            }
            Mode::Mosaic { mut mosaic, .. } => {
                let project_space = mosaic.take_project_space();
                self.mode = Mode::Project { project_space };
                serde_json::json!({
                    "mode": "project",
                    "changed": true,
                })
            }
            Mode::Transition => {
                self.mode = Mode::Transition;
                serde_json::json!({
                    "error": "Odon is currently transitioning between views.",
                })
            }
        }
    }

    fn control_current_view(&self) -> serde_json::Value {
        let (mode, view) = match &self.mode {
            Mode::Project { .. } => ("project", serde_json::Value::Null),
            Mode::Single(app) => ("single", app.control_view_snapshot()),
            Mode::Mosaic { mosaic, .. } => ("mosaic", mosaic.control_view_snapshot()),
            Mode::Transition => ("transition", serde_json::Value::Null),
        };
        serde_json::json!({
            "mode": mode,
            "view": view,
            "project": self.control_project_rois(),
        })
    }

    fn load_project_space_from_file(project_space: &mut ProjectSpace, path: &Path) -> bool {
        match project_space.load_from_file(path) {
            Ok(()) => true,
            Err(err) => {
                project_space.set_status(format!("Load project failed: {err}"));
                false
            }
        }
    }

    fn load_project_into_current_mode(&mut self, path: &Path) {
        let loaded = match &mut self.mode {
            Mode::Project { project_space } => {
                Self::load_project_space_from_file(project_space, path)
            }
            Mode::Single(app) => {
                let mut ps = app.take_project_space();
                let loaded = Self::load_project_space_from_file(&mut ps, path);
                app.set_project_space(ps);
                loaded
            }
            Mode::Mosaic { mosaic, .. } => {
                let mut ps = mosaic.take_project_space();
                let loaded = Self::load_project_space_from_file(&mut ps, path);
                if loaded {
                    mosaic.set_layer_groups(ps.layer_groups().clone());
                }
                mosaic.set_project_space(ps);
                loaded
            }
            Mode::Transition => false,
        };
        if loaded {
            self.load_control_manifest_from_project();
            self.record_recent_project(path);
        }
    }

    fn ui_settings_dialog(&mut self, ctx: &egui::Context) {
        if !self.settings_open {
            return;
        }

        let before = self.app_settings.clone();
        let mut open = self.settings_open;
        egui::Window::new("Settings")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.heading("Auto Contrast");
                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut self.app_settings.auto_contrast.enabled_on_open,
                        "Apply auto contrast when opening a dataset",
                    );
                    settings_help_button(
                        ui,
                        "Automatically sets channel contrast limits after opening a dataset so the image is immediately visible.",
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Method");
                    settings_help_button(
                        ui,
                        "Controls how Odon chooses automatic contrast limits from the image intensity distribution.",
                    );
                    egui::ComboBox::from_id_salt("global_auto_contrast_method")
                        .selected_text(self.app_settings.auto_contrast.method.label())
                        .show_ui(ui, |ui| {
                            for method in crate::app_support::settings::AutoContrastMethod::ALL {
                                ui.selectable_value(
                                    &mut self.app_settings.auto_contrast.method,
                                    method,
                                    method.label(),
                                );
                            }
                        });
                });
                ui.label(self.app_settings.auto_contrast.method.description());

                let settings = &mut self.app_settings.auto_contrast;
                match settings.method {
                    crate::app_support::settings::AutoContrastMethod::ZeroToP97 => {
                        ui.horizontal(|ui| {
                            ui.label("Upper percentile");
                            settings_help_button(
                                ui,
                                "Pixels brighter than this percentile are clipped for display contrast.",
                            );
                            ui.add(
                                egui::DragValue::new(&mut settings.upper_percentile)
                                    .range(1..=100)
                                    .speed(0.2)
                                    .suffix("%"),
                            );
                        });
                    }
                    crate::app_support::settings::AutoContrastMethod::P1ToP99 => {
                        ui.horizontal(|ui| {
                            ui.label("Lower percentile");
                            settings_help_button(
                                ui,
                                "Pixels darker than this percentile are clipped for display contrast.",
                            );
                            ui.add(
                                egui::DragValue::new(&mut settings.lower_percentile)
                                    .range(0..=99)
                                    .speed(0.2)
                                    .suffix("%"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Upper percentile");
                            settings_help_button(
                                ui,
                                "Pixels brighter than this percentile are clipped for display contrast.",
                            );
                            ui.add(
                                egui::DragValue::new(&mut settings.upper_percentile)
                                    .range(1..=100)
                                    .speed(0.2)
                                    .suffix("%"),
                            );
                        });
                    }
                    crate::app_support::settings::AutoContrastMethod::ZeroToMax => {}
                }
                self.app_settings.auto_contrast = self.app_settings.auto_contrast.normalized();

                ui.separator();
                ui.heading("Object Rendering");
                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut self.app_settings.fast_object_rendering,
                        "Fast object rendering",
                    );
                    settings_help_button(
                        ui,
                        "When viewing many polygon objects at low zoom, draws lightweight proxy points until zoomed in enough for full polygons.",
                    );
                });

                ui.add_space(8.0);
                let can_apply_now = matches!(self.mode, Mode::Single(_));
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(can_apply_now, egui::Button::new("Apply To Current Viewer"))
                        .clicked()
                    {
                        if let Mode::Single(app) = &mut self.mode {
                            app.set_auto_contrast_settings(self.app_settings.auto_contrast);
                            app.apply_auto_contrast_now();
                            self.settings_status = format!(
                                "Applied {} to the current viewer.",
                                self.app_settings.auto_contrast.method.label()
                            );
                        }
                    }
                    settings_help_button(
                        ui,
                        "Applies the current auto-contrast method immediately to the open single-image viewer.",
                    );
                });
                if !can_apply_now {
                    ui.label("Open a single dataset viewer to apply these settings immediately.");
                }

                if let Ok(path) = settings_file_path() {
                    ui.add_space(8.0);
                    ui.label(format!("Settings file: {}", path.display()));
                }

                if !self.settings_status.trim().is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(&self.settings_status);
                }
            });
        self.settings_open = open;

        if self.app_settings != before {
            self.apply_app_settings_to_mode();
            self.persist_app_settings();
        }
    }

    fn save_screenshot_via_dialog(&mut self) {
        let default_name = match &self.mode {
            Mode::Single(app) => app.default_screenshot_filename(),
            Mode::Mosaic { mosaic, .. } => mosaic.default_screenshot_filename(),
            _ => "odon.screenshot.png".to_string(),
        };
        if let Some(path) = FileDialog::new()
            .add_filter("PNG", &["png"])
            .set_file_name(&default_name)
            .set_title("Save Screenshot (Canvas PNG)")
            .save_file()
        {
            match &mut self.mode {
                Mode::Single(app) => {
                    app.request_screenshot_png(path);
                }
                Mode::Project { project_space } => {
                    project_space.set_status("Save Screenshot: open a dataset first.".to_string());
                }
                Mode::Mosaic { mosaic, .. } => {
                    mosaic.request_screenshot_png(path);
                }
                Mode::Transition => {}
            }
        }
    }

    fn quick_screenshot(&mut self) {
        let mut fallback_to_dialog = false;
        match &mut self.mode {
            Mode::Single(app) => {
                if app.screenshot_output_dir().is_some() {
                    if let Err(err) = app.request_quick_screenshot_png() {
                        app.set_status(format!("Quick screenshot failed: {err}"));
                    }
                } else {
                    fallback_to_dialog = true;
                }
            }
            Mode::Project { project_space } => {
                project_space.set_status("Save Screenshot: open a dataset first.".to_string());
            }
            Mode::Mosaic { mosaic, .. } => {
                if mosaic.screenshot_output_dir().is_some() {
                    if let Err(err) = mosaic.request_quick_screenshot_png() {
                        mosaic.set_status(format!("Quick screenshot failed: {err}"));
                    }
                } else {
                    fallback_to_dialog = true;
                }
            }
            Mode::Transition => {}
        }
        if fallback_to_dialog {
            self.save_screenshot_via_dialog();
        }
    }

    fn apply_remote_s3_listing(
        &mut self,
        session: S3Browser,
        signature: String,
        listing: S3BrowseListing,
        selected_dataset_prefixes: HashSet<String>,
    ) {
        let current_prefix = listing.prefix.clone();
        let parent_prefix = listing.parent_prefix.clone();
        let entries = listing.entries.clone();
        let current_is_dataset = listing.current_is_dataset;
        self.remote_s3_browser = Some(RootRemoteS3BrowserState {
            session,
            signature,
            current_prefix: current_prefix.clone(),
            parent_prefix,
            entries,
            current_is_dataset,
            selected_dataset_prefixes,
            listing_cache: HashMap::new(),
        });
        if let Some(state) = self.remote_s3_browser.as_mut() {
            state.listing_cache.insert(current_prefix, listing);
        }
    }

    fn connect_remote_s3_browser(&mut self) -> anyhow::Result<()> {
        let browser = build_s3_browser(
            &self.remote_s3_endpoint,
            &self.remote_s3_region,
            &self.remote_s3_bucket,
            &self.remote_s3_access_key,
            &self.remote_s3_secret_key,
        )?;
        let signature = self.remote_s3_signature();
        let browse_prefix = if self.remote_s3_prefix.trim().ends_with(".ome.zarr")
            || self.remote_s3_prefix.trim().ends_with(".zarr")
        {
            self.remote_s3_prefix
                .trim()
                .trim_matches('/')
                .rsplit_once('/')
                .map(|(parent, _)| parent.to_string())
                .unwrap_or_default()
        } else {
            self.remote_s3_prefix.trim().trim_matches('/').to_string()
        };
        let listing = list_s3_prefix(&browser, &browse_prefix)?;
        self.apply_remote_s3_listing(browser, signature, listing, Default::default());
        Ok(())
    }

    fn refresh_remote_s3_browser(&mut self) -> anyhow::Result<()> {
        let Some(state) = self.remote_s3_browser.take() else {
            anyhow::bail!("not connected to S3");
        };
        let listing = list_s3_prefix(&state.session, &state.current_prefix)?;
        let mut selected = state.selected_dataset_prefixes;
        let mut cache = state.listing_cache;
        cache.insert(listing.prefix.clone(), listing.clone());
        self.apply_remote_s3_listing(state.session, state.signature, listing, selected.clone());
        if let Some(next) = self.remote_s3_browser.as_mut() {
            next.selected_dataset_prefixes = std::mem::take(&mut selected);
            next.listing_cache = cache;
        }
        Ok(())
    }

    fn browse_remote_s3_prefix(&mut self, prefix: String) -> anyhow::Result<()> {
        let Some(state) = self.remote_s3_browser.take() else {
            anyhow::bail!("not connected to S3");
        };
        let mut selected = state.selected_dataset_prefixes;
        let mut cache = state.listing_cache;
        let listing = if let Some(cached) = cache.get(&prefix).cloned() {
            cached
        } else {
            let listing = list_s3_prefix(&state.session, &prefix)?;
            cache.insert(prefix.clone(), listing.clone());
            listing
        };
        self.apply_remote_s3_listing(state.session, state.signature, listing, selected.clone());
        if let Some(next) = self.remote_s3_browser.as_mut() {
            next.selected_dataset_prefixes = std::mem::take(&mut selected);
            next.listing_cache = cache;
        }
        Ok(())
    }

    fn selected_remote_s3_datasets(&self) -> Vec<S3DatasetSelection> {
        let Some(state) = self.remote_s3_browser.as_ref() else {
            return Vec::new();
        };
        let endpoint = self.remote_s3_endpoint.trim().to_string();
        let region = self.remote_s3_region.trim().to_string();
        let bucket = self.remote_s3_bucket.trim().to_string();
        let access_key = self.remote_s3_access_key.trim().to_string();
        let secret_key = self.remote_s3_secret_key.trim().to_string();
        let mut prefixes = state
            .selected_dataset_prefixes
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        prefixes.sort();
        prefixes
            .into_iter()
            .map(|prefix| S3DatasetSelection {
                endpoint: endpoint.clone(),
                region: region.clone(),
                bucket: bucket.clone(),
                prefix,
                access_key: access_key.clone(),
                secret_key: secret_key.clone(),
            })
            .collect()
    }

    fn open_remote_dataset_from_dialog(&mut self) -> anyhow::Result<RootRemoteAction> {
        match self.remote_mode {
            RemoteMode::Http => {
                let mut url = self
                    .remote_http_url
                    .trim()
                    .trim_end_matches('/')
                    .to_string();
                if url.is_empty() {
                    anyhow::bail!("URL is empty");
                }
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    url = format!("https://{url}");
                }
                let store = build_http_store(&url)?;
                let source = crate::data::dataset_source::DatasetSource::Http { base_url: url };
                let dataset = OmeZarrDataset::open_with_store(source, store.clone())?;
                Ok(RootRemoteAction::OpenSingle {
                    dataset,
                    store,
                    runtime: None,
                })
            }
            RemoteMode::S3 => {
                let mut endpoint = self.remote_s3_endpoint.trim().to_string();
                let region = self.remote_s3_region.trim().to_string();
                let bucket = self.remote_s3_bucket.trim().to_string();
                let prefix = self.remote_s3_prefix.trim().trim_matches('/').to_string();
                let access_key = self.remote_s3_access_key.trim().to_string();
                let secret_key = self.remote_s3_secret_key.trim().to_string();
                if endpoint.is_empty() || bucket.is_empty() {
                    anyhow::bail!("endpoint and bucket are required");
                }
                if access_key.is_empty() || secret_key.is_empty() {
                    anyhow::bail!("access key / secret key are required");
                }
                if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                    endpoint = format!("https://{endpoint}");
                }
                let S3Store { store, runtime } = build_s3_store(
                    &endpoint,
                    &region,
                    &bucket,
                    &prefix,
                    &access_key,
                    &secret_key,
                )?;
                let source = crate::data::dataset_source::DatasetSource::S3 {
                    endpoint,
                    region: if region.is_empty() {
                        "auto".to_string()
                    } else {
                        region
                    },
                    bucket,
                    prefix,
                };
                let dataset = OmeZarrDataset::open_with_store(source, store.clone())?;
                Ok(RootRemoteAction::OpenSingle {
                    dataset,
                    store,
                    runtime: Some(runtime),
                })
            }
        }
    }

    fn ui_remote_dialog(&mut self, ctx: &egui::Context) -> Option<RootRemoteAction> {
        if !self.remote_dialog_open {
            return None;
        }
        let mut open = self.remote_dialog_open;
        let mut s3_inputs_changed = false;
        let mut connect_s3 = false;
        let mut refresh_s3 = false;
        let mut browse_to: Option<String> = None;
        let mut open_single = false;
        let mut open_mosaic = false;
        let mut add_to_project = false;
        let mut action = None;
        egui::Window::new("Open Remote OME-Zarr")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.remote_mode, RemoteMode::Http, "HTTP(S)");
                    ui.selectable_value(&mut self.remote_mode, RemoteMode::S3, "S3 / R2");
                });
                ui.separator();
                match self.remote_mode {
                    RemoteMode::Http => {
                        ui.label("Dataset URL (points to the OME-Zarr directory):");
                        ui.text_edit_singleline(&mut self.remote_http_url);
                    }
                    RemoteMode::S3 => {
                        ui.label("Endpoint (R2): https://<accountid>.r2.cloudflarestorage.com");
                        s3_inputs_changed |= ui
                            .text_edit_singleline(&mut self.remote_s3_endpoint)
                            .changed();
                        ui.horizontal(|ui| {
                            ui.label("Region:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_region)
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Bucket:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_bucket)
                                .changed();
                        });
                        ui.label("Prefix (path to the OME-Zarr directory within the bucket):");
                        s3_inputs_changed |= ui
                            .text_edit_singleline(&mut self.remote_s3_prefix)
                            .changed();
                        ui.separator();
                        ui.label("Credentials (static):");
                        ui.horizontal(|ui| {
                            ui.label("Access key:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_access_key)
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Secret key:");
                            s3_inputs_changed |= ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.remote_s3_secret_key)
                                        .password(true),
                                )
                                .changed();
                        });
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            let connect_label = if self.remote_s3_browser.is_some() {
                                "Reconnect"
                            } else {
                                "Connect"
                            };
                            if ui.button(connect_label).clicked() {
                                connect_s3 = true;
                            }
                            if ui
                                .add_enabled(
                                    self.remote_s3_browser.is_some(),
                                    egui::Button::new("Refresh"),
                                )
                                .clicked()
                            {
                                refresh_s3 = true;
                            }
                        });
                        let browser_view = self.remote_s3_browser.as_ref().map(|state| {
                            (
                                state.current_prefix.clone(),
                                state.parent_prefix.clone(),
                                state.current_is_dataset,
                                state.entries.clone(),
                                state.selected_dataset_prefixes.clone(),
                            )
                        });
                        if let Some((
                            current_prefix,
                            parent_prefix,
                            current_is_dataset,
                            entries,
                            mut selected_prefixes,
                        )) = browser_view
                        {
                            ui.add_space(6.0);
                            ui.separator();
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.set_min_width(620.0);
                                ui.horizontal(|ui| {
                                    ui.label("Browser");
                                    ui.label(if current_prefix.is_empty() {
                                        "<bucket root>".to_string()
                                    } else {
                                        current_prefix.clone()
                                    });
                                    if ui
                                        .add_enabled(
                                            parent_prefix.is_some() || !current_prefix.is_empty(),
                                            egui::Button::new("Up"),
                                        )
                                        .clicked()
                                    {
                                        browse_to = Some(parent_prefix.unwrap_or_default());
                                    }
                                });
                                if current_is_dataset {
                                    ui.horizontal(|ui| {
                                        let mut selected =
                                            selected_prefixes.contains(&current_prefix);
                                        if ui.checkbox(&mut selected, "Select current").changed() {
                                            if selected {
                                                selected_prefixes.insert(current_prefix.clone());
                                            } else {
                                                selected_prefixes.remove(&current_prefix);
                                            }
                                        }
                                        ui.label("This prefix looks like an OME-Zarr dataset.");
                                        if ui.button("Use this prefix").clicked() {
                                            self.remote_s3_prefix = current_prefix.clone();
                                        }
                                    });
                                }
                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.add_sized([28.0, 18.0], egui::Label::new("Sel"));
                                    ui.small("Name");
                                });
                                egui::ScrollArea::vertical()
                                    .auto_shrink([false, false])
                                    .max_height(260.0)
                                    .show(ui, |ui| {
                                        ui.set_min_width(ui.available_width());
                                        for entry in &entries {
                                            ui.horizontal(|ui| {
                                                if entry.is_dataset {
                                                    let mut selected =
                                                        selected_prefixes.contains(&entry.prefix);
                                                    if ui
                                                        .add_sized(
                                                            [28.0, 20.0],
                                                            egui::Checkbox::without_text(
                                                                &mut selected,
                                                            ),
                                                        )
                                                        .on_hover_text("Select this OME-Zarr")
                                                        .changed()
                                                    {
                                                        if selected {
                                                            selected_prefixes
                                                                .insert(entry.prefix.clone());
                                                        } else {
                                                            selected_prefixes.remove(&entry.prefix);
                                                        }
                                                    }
                                                    if ui
                                                        .selectable_label(
                                                            self.remote_s3_prefix.trim()
                                                                == entry.prefix,
                                                            format!(
                                                                "[{}] {}",
                                                                if entry
                                                                    .prefix
                                                                    .ends_with(".ome.zarr")
                                                                {
                                                                    "OME-Zarr"
                                                                } else {
                                                                    "Zarr"
                                                                },
                                                                entry.name
                                                            ),
                                                        )
                                                        .clicked()
                                                    {
                                                        self.remote_s3_prefix =
                                                            entry.prefix.clone();
                                                    }
                                                    if ui.small_button("Browse").clicked() {
                                                        browse_to = Some(entry.prefix.clone());
                                                    }
                                                } else {
                                                    ui.add_space(28.0);
                                                    if ui
                                                        .button(format!("[dir] {}", entry.name))
                                                        .clicked()
                                                    {
                                                        browse_to = Some(entry.prefix.clone());
                                                    }
                                                }
                                            });
                                        }
                                    });
                            });
                            if let Some(state) = self.remote_s3_browser.as_mut() {
                                state.selected_dataset_prefixes = selected_prefixes;
                            }
                        }
                    }
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.remote_dialog_open = false;
                        self.remote_status.clear();
                    }
                    if ui.button("Open").clicked() {
                        open_single = true;
                    }
                    let selected_remote = self.selected_remote_s3_datasets();
                    if ui
                        .add_enabled(
                            self.remote_mode == RemoteMode::S3 && selected_remote.len() >= 2,
                            egui::Button::new(format!("Open Mosaic ({})", selected_remote.len())),
                        )
                        .clicked()
                    {
                        open_mosaic = true;
                    }
                    if ui
                        .add_enabled(
                            self.remote_mode == RemoteMode::S3 && !selected_remote.is_empty(),
                            egui::Button::new(format!(
                                "Add to Project ({})",
                                selected_remote.len()
                            )),
                        )
                        .clicked()
                    {
                        add_to_project = true;
                    }
                });
                if !self.remote_status.is_empty() {
                    ui.add_space(6.0);
                    ui.label(self.remote_status.clone());
                }
            });

        if s3_inputs_changed {
            self.clear_remote_s3_browser();
        }
        if connect_s3 {
            match self.connect_remote_s3_browser() {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if refresh_s3 {
            match self.refresh_remote_s3_browser() {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if let Some(prefix) = browse_to {
            match self.browse_remote_s3_prefix(prefix) {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if open_single {
            match self.open_remote_dataset_from_dialog() {
                Ok(req) => {
                    self.remote_dialog_open = false;
                    self.remote_status.clear();
                    action = Some(req);
                }
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if open_mosaic {
            let selected = self.selected_remote_s3_datasets();
            if selected.len() >= 2 {
                self.remote_dialog_open = false;
                self.remote_status.clear();
                action = Some(RootRemoteAction::OpenS3Mosaic(selected));
            } else {
                self.remote_status = "Select at least 2 S3 OME-Zarr datasets.".to_string();
            }
        } else if add_to_project {
            let sources = self
                .selected_remote_s3_datasets()
                .into_iter()
                .map(|dataset| DatasetSource::S3 {
                    endpoint: dataset.endpoint,
                    region: dataset.region,
                    bucket: dataset.bucket,
                    prefix: dataset.prefix,
                })
                .collect::<Vec<_>>();
            if sources.is_empty() {
                self.remote_status = "Select at least 1 S3 OME-Zarr dataset.".to_string();
            } else {
                self.remote_dialog_open = false;
                self.remote_status.clear();
                action = Some(RootRemoteAction::AddToProject(sources));
            }
        }

        self.remote_dialog_open = open && self.remote_dialog_open;
        if !open {
            self.remote_dialog_open = false;
        }
        action
    }

    pub fn new_project(
        cc: &eframe::CreationContext<'_>,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let (app_settings, mut settings_status) = Self::load_app_settings();
        let control_bridge = Self::spawn_control_bridge(&cc.egui_ctx, &mut settings_status);
        let mut ps = ProjectSpace::default();
        if let Some(path) = project_path.as_deref() {
            if let Err(err) = ps.load_from_file(path) {
                ps.set_status(format!("Load project failed: {err}"));
            }
        }
        let mut root = Self {
            mode: Mode::Project { project_space: ps },
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
            pending_control_open_root: None,
            pending_deep_link: None,
            deep_link_rx: None,
            object_preload_project: None,
            object_preload_rx: None,
            object_preload_cache: HashMap::new(),
            object_preload_settings: ObjectPreloadSettings::default(),
            object_preload_total: 0,
            object_preload_done: 0,
            object_preload_failed: 0,
            view_show_scale_bar: true,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            app_settings,
            settings_open: false,
            settings_status,
            active_help_topic: None,
            control_bridge,
            control_external_revision: 0,
            control_project_revision: 0,
            control_observed_state: None,
            control_mutated_this_frame: false,
            control_last_observed_at: Instant::now() - Duration::from_millis(34),
            #[cfg(target_os = "macos")]
            native_menu: None,
        };
        root.load_control_manifest_from_project();
        Ok(root)
    }

    pub fn new_single(
        cc: &eframe::CreationContext<'_>,
        dataset: OmeZarrDataset,
        store: std::sync::Arc<dyn zarrs::storage::ReadableStorageTraits>,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let (app_settings, mut settings_status) = Self::load_app_settings();
        let control_bridge = Self::spawn_control_bridge(&cc.egui_ctx, &mut settings_status);
        let mut app = OmeZarrViewerApp::new(cc, dataset, store, app_settings.auto_contrast);
        app.set_show_scale_bar(true);
        app.set_fast_object_rendering(app_settings.fast_object_rendering);
        if let Some(path) = project_path.as_deref() {
            let mut ps = ProjectSpace::default();
            if let Err(err) = ps.load_from_file(path) {
                ps.set_status(format!("Load project failed: {err}"));
            }
            app.set_project_space(ps);
        }
        let mut root = Self {
            mode: Mode::Single(app),
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
            pending_control_open_root: None,
            pending_deep_link: None,
            deep_link_rx: None,
            object_preload_project: None,
            object_preload_rx: None,
            object_preload_cache: HashMap::new(),
            object_preload_settings: ObjectPreloadSettings::default(),
            object_preload_total: 0,
            object_preload_done: 0,
            object_preload_failed: 0,
            view_show_scale_bar: true,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            app_settings,
            settings_open: false,
            settings_status,
            active_help_topic: None,
            control_bridge,
            control_external_revision: 0,
            control_project_revision: 0,
            control_observed_state: None,
            control_mutated_this_frame: false,
            control_last_observed_at: Instant::now() - Duration::from_millis(34),
            #[cfg(target_os = "macos")]
            native_menu: None,
        };
        root.load_control_manifest_from_project();
        Ok(root)
    }

    pub fn new_mosaic(
        cc: &eframe::CreationContext<'_>,
        mut mosaic: MosaicViewerApp,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let (app_settings, mut settings_status) = Self::load_app_settings();
        let control_bridge = Self::spawn_control_bridge(&cc.egui_ctx, &mut settings_status);
        let mut ps = ProjectSpace::default();
        if let Some(path) = project_path.as_deref() {
            if let Err(err) = ps.load_from_file(path) {
                ps.set_status(format!("Load project failed: {err}"));
            }
        }
        mosaic.set_fast_object_rendering(app_settings.fast_object_rendering);
        mosaic.set_layer_groups(ps.layer_groups().clone());
        let mut root = Self {
            mode: Mode::Mosaic {
                mosaic,
                ret: ReturnToSingleState { dataset_root: None },
            },
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
            pending_control_open_root: None,
            pending_deep_link: None,
            deep_link_rx: None,
            object_preload_project: None,
            object_preload_rx: None,
            object_preload_cache: HashMap::new(),
            object_preload_settings: ObjectPreloadSettings::default(),
            object_preload_total: 0,
            object_preload_done: 0,
            object_preload_failed: 0,
            view_show_scale_bar: true,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            app_settings,
            settings_open: false,
            settings_status,
            active_help_topic: None,
            control_bridge,
            control_external_revision: 0,
            control_project_revision: 0,
            control_observed_state: None,
            control_mutated_this_frame: false,
            control_last_observed_at: Instant::now() - Duration::from_millis(34),
            #[cfg(target_os = "macos")]
            native_menu: None,
        };
        root.load_control_manifest_from_project();
        Ok(root)
    }

    pub fn queue_open_root(&mut self, root: PathBuf) {
        self.pending_open_root = Some(root);
    }

    pub fn queue_deep_link(&mut self, request: DeepLinkRequest) {
        self.pending_deep_link = Some(request);
    }

    pub fn set_deep_link_receiver(&mut self, rx: Receiver<DeepLinkRequest>) {
        self.deep_link_rx = Some(rx);
    }

    fn poll_project_object_preload(&mut self) {
        let Some(rx) = self.object_preload_rx.take() else {
            return;
        };
        let mut keep_rx = true;
        while let Ok(event) = rx.try_recv() {
            self.object_preload_done = self.object_preload_done.saturating_add(1);
            match event.result {
                Ok(preloaded) => {
                    log_warn!(
                        "project preload: cached {} ({}) object segmentation {}",
                        event.settings.mode.label(),
                        event.settings.property_label(),
                        event.path.display()
                    );
                    let preloaded = Arc::new(preloaded);
                    self.object_preload_cache
                        .insert((event.path.clone(), event.settings), preloaded.clone());
                    if event.settings == self.object_preload_settings
                        && let Mode::Mosaic { mosaic, .. } = &mut self.mode
                    {
                        let installed = mosaic.install_preloaded_project_segmentations(&[(
                            event.path.clone(),
                            preloaded,
                        )]);
                        if installed > 0 {
                            log_warn!(
                                "project preload: installed cached object segmentation for {installed} visible mosaic ROI(s)"
                            );
                        }
                    }
                }
                Err(err) => {
                    self.object_preload_failed = self.object_preload_failed.saturating_add(1);
                    log_warn!(
                        "project preload: failed object segmentation {}: {err}",
                        event.path.display()
                    );
                }
            }
            if event.finished {
                keep_rx = false;
                log_warn!(
                    "project preload: finished ({} cached object segmentation(s))",
                    self.object_preload_cache.len()
                );
            }
        }
        if keep_rx {
            self.object_preload_rx = Some(rx);
        }
    }

    fn sync_project_object_preload_scope(&mut self, project_path: Option<PathBuf>) {
        if self.object_preload_project == project_path {
            return;
        }
        self.object_preload_project = project_path;
        self.object_preload_rx = None;
        self.object_preload_cache.clear();
        self.object_preload_settings = ObjectPreloadSettings::default();
        self.object_preload_total = 0;
        self.object_preload_done = 0;
        self.object_preload_failed = 0;
    }

    fn start_project_object_preload(
        &mut self,
        project_space: &ProjectSpace,
        settings: ObjectPreloadSettings,
    ) {
        let Some(project_path) = project_space.saved_project_path() else {
            return;
        };

        let paths = project_object_segmentation_paths(project_space);
        self.object_preload_project = Some(project_path);
        self.object_preload_rx = None;
        self.object_preload_cache.clear();
        self.object_preload_settings = settings;
        self.object_preload_total = paths.len();
        self.object_preload_done = 0;
        self.object_preload_failed = 0;
        if paths.is_empty() {
            return;
        }

        let total = paths.len();
        let (tx, rx) = std::sync::mpsc::channel::<ProjectObjectPreloadEvent>();
        self.object_preload_rx = Some(rx);
        log_warn!(
            "project preload: starting {total} {} ({}) object segmentation(s)",
            settings.mode.label(),
            settings.property_label()
        );
        if let Err(err) = std::thread::Builder::new()
            .name("odon-project-object-preload".to_string())
            .spawn(move || {
                for (idx, path) in paths.into_iter().enumerate() {
                    let result =
                        crate::objects::preload_objects_from_path(path.clone(), 1.0, settings)
                            .map_err(|err| err.to_string());
                    if tx
                        .send(ProjectObjectPreloadEvent {
                            path,
                            settings,
                            result,
                            finished: idx + 1 == total,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
        {
            log_warn!("project preload: failed to start background thread: {err}");
            self.object_preload_rx = None;
            self.object_preload_total = 0;
        }
    }

    fn clear_project_object_preload(&mut self) {
        self.object_preload_rx = None;
        self.object_preload_cache.clear();
        self.object_preload_total = 0;
        self.object_preload_done = 0;
        self.object_preload_failed = 0;
    }

    fn cached_project_object_layer(
        &self,
        project_space: &ProjectSpace,
        roi: &ProjectRoi,
    ) -> Option<Arc<PreloadedObjectLayer>> {
        let path = project_roi_segmentation_path(project_space, roi)?;
        self.object_preload_cache
            .get(&(path, self.object_preload_settings))
            .cloned()
    }

    fn cached_project_object_layers_for_rois(
        &self,
        project_space: &ProjectSpace,
        rois: &[ProjectRoi],
    ) -> Vec<(PathBuf, Arc<PreloadedObjectLayer>)> {
        let mut seen = HashSet::new();
        rois.iter()
            .filter_map(|roi| {
                let path = project_roi_segmentation_path(project_space, roi)?;
                if !seen.insert(path.clone()) {
                    return None;
                }
                let preloaded = self
                    .object_preload_cache
                    .get(&(path.clone(), self.object_preload_settings))
                    .cloned()?;
                Some((path, preloaded))
            })
            .collect()
    }

    pub fn add_paths_to_project(&mut self, paths: Vec<PathBuf>) {
        match &mut self.mode {
            Mode::Project { project_space } => project_space.handle_dropped_paths(paths),
            Mode::Single(app) => {
                let mut ps = app.take_project_space();
                ps.handle_dropped_paths(paths);
                app.set_project_space(ps);
            }
            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().handle_dropped_paths(paths),
            Mode::Transition => {}
        }
    }

    fn open_single(&mut self, ctx: &egui::Context, root: &PathBuf, project_space: ProjectSpace) {
        let root = normalize_local_dataset_path(root).unwrap_or_else(|| root.clone());
        log_info!("open_single: {}", root.to_string_lossy());
        if matches!(
            classify_local_dataset_path(&root),
            Some(LocalDatasetKind::Tiff)
        ) {
            match OmeZarrViewerApp::new_tiff_runtime(
                ctx,
                self.gpu_available,
                root.clone(),
                self.app_settings.auto_contrast,
            ) {
                Ok(mut app) => {
                    log_debug!("open_single: detected TIFF");
                    self.configure_single_app(&mut app);
                    app.set_project_space(project_space);
                    self.mode = Mode::Single(app);
                }
                Err(err) => {
                    let mut ps = project_space;
                    log_warn!("open_single: open_tiff failed: {err:?}");
                    ps.set_status(format!("Open TIFF failed: {err}"));
                    self.mode = Mode::Project { project_space: ps };
                }
            }
            return;
        }
        match OmeZarrDataset::open_local(&root) {
            Ok((dataset, store)) => {
                log_debug!("open_single: detected OME-Zarr");
                let mut app = OmeZarrViewerApp::new_runtime(
                    ctx,
                    self.gpu_available,
                    dataset,
                    store,
                    self.app_settings.auto_contrast,
                );
                self.configure_single_app(&mut app);
                app.set_project_space(project_space);
                self.mode = Mode::Single(app);
            }
            Err(err) => {
                // If the root looks like SpatialData, show a chooser for which image group to open
                // (and optional points/shapes overlays).
                match discover_spatialdata(&root) {
                    Ok(discovery) if !discovery.images.is_empty() => {
                        log_debug!("open_single: detected SpatialData");
                        let mut dlg = SpatialOpenDialog {
                            discovery,
                            selected_image: 0,
                            selected_labels: None,
                            selected_shapes: Vec::new(),
                            selected_points: None,
                            points_max: 200_000,
                            status: String::new(),
                        };
                        if let Some(i) = dlg
                            .discovery
                            .labels
                            .iter()
                            .position(|s| s.name == "cells" || s.name == "point8_labels")
                        {
                            dlg.selected_labels = Some(i);
                        }
                        // Default: turn on cell boundaries if present; keep points off by default.
                        if let Some(i) = dlg
                            .discovery
                            .shapes
                            .iter()
                            .position(|s| s.name == "cell_boundaries")
                        {
                            dlg.selected_shapes.push(i);
                        }
                        // Restore the project state so the UI stays intact while the dialog is open.
                        self.mode = Mode::Project { project_space };
                        self.spatial_open = Some(dlg);
                    }
                    _ => {
                        // Xenium Explorer bundle (experiment.xenium + morphology OME-TIFF + zarr.zip overlays).
                        if let Ok(x) = discover_xenium_explorer(&root) {
                            log_debug!("open_single: detected Xenium Explorer");
                            if let Some(img_root) = x.morphology_mip_omezarr.clone() {
                                match OmeZarrDataset::open_local(&img_root) {
                                    Ok((dataset, store)) => {
                                        let mut app = OmeZarrViewerApp::new_runtime(
                                            ctx,
                                            self.gpu_available,
                                            dataset,
                                            store,
                                            self.app_settings.auto_contrast,
                                        );
                                        self.configure_single_app(&mut app);
                                        app.attach_xenium_layers(
                                            x.root.clone(),
                                            x.cells_zarr_zip.clone(),
                                            x.transcripts_zarr_zip.clone(),
                                            x.pixel_size_um,
                                        );
                                        app.set_project_space(project_space);
                                        self.mode = Mode::Single(app);
                                        return;
                                    }
                                    Err(e) => {
                                        let mut ps = project_space;
                                        ps.set_status(format!(
                                            "Open Xenium failed: could not open morphology OME-Zarr: {e}"
                                        ));
                                        self.mode = Mode::Project { project_space: ps };
                                        return;
                                    }
                                }
                            } else {
                                if let Some(morph_tiff) = x.morphology_mip_tiff.clone() {
                                    match OmeZarrViewerApp::new_xenium_runtime(
                                        ctx,
                                        self.gpu_available,
                                        x.root.clone(),
                                        morph_tiff,
                                        x.cells_zarr_zip.clone(),
                                        x.transcripts_zarr_zip.clone(),
                                        x.pixel_size_um,
                                        self.app_settings.auto_contrast,
                                    ) {
                                        Ok(mut app) => {
                                            self.configure_single_app(&mut app);
                                            app.set_project_space(project_space);
                                            self.mode = Mode::Single(app);
                                            return;
                                        }
                                        Err(e) => {
                                            let mut ps = project_space;
                                            ps.set_status(format!(
                                                "Open Xenium failed: could not open morphology OME-TIFF: {e}"
                                            ));
                                            self.mode = Mode::Project { project_space: ps };
                                            return;
                                        }
                                    }
                                }
                                let mut ps = project_space;
                                ps.set_status(
                                    "Open Xenium failed: morphology base image was not found as OME-Zarr or OME-TIFF."
                                        .to_string(),
                                );
                                self.mode = Mode::Project { project_space: ps };
                                return;
                            }
                        }

                        let mut ps = project_space;
                        log_warn!("open_single: open_local failed: {err:?}");
                        ps.set_status(format!("Open failed: {err}"));
                        self.mode = Mode::Project { project_space: ps };
                    }
                }
            }
        }
    }

    fn open_dataset_source(
        &mut self,
        ctx: &egui::Context,
        source: DatasetSource,
        project_space: ProjectSpace,
    ) {
        match source {
            DatasetSource::Local(path) => self.open_single(ctx, &path, project_space),
            DatasetSource::Http { base_url } => {
                match build_http_store(&base_url).and_then(|store| {
                    OmeZarrDataset::open_with_store(
                        DatasetSource::Http {
                            base_url: base_url.clone(),
                        },
                        store.clone(),
                    )
                    .map(|dataset| (dataset, store))
                }) {
                    Ok((dataset, store)) => {
                        let mut app = OmeZarrViewerApp::new_runtime(
                            ctx,
                            self.gpu_available,
                            dataset,
                            store,
                            self.app_settings.auto_contrast,
                        );
                        self.configure_single_app(&mut app);
                        app.set_project_space(project_space);
                        self.mode = Mode::Single(app);
                    }
                    Err(err) => {
                        let mut ps = project_space;
                        ps.set_status(format!("Open remote dataset failed: {err}"));
                        self.mode = Mode::Project { project_space: ps };
                    }
                }
            }
            DatasetSource::S3 {
                endpoint,
                region,
                bucket,
                prefix,
            } => {
                if self.remote_s3_access_key.trim().is_empty()
                    || self.remote_s3_secret_key.trim().is_empty()
                {
                    let mut ps = project_space;
                    ps.set_status(
                        "S3 credentials are not available in this session. Use Open Remote... and reconnect first."
                            .to_string(),
                    );
                    self.mode = Mode::Project { project_space: ps };
                    return;
                }
                match build_s3_store(
                    &endpoint,
                    &region,
                    &bucket,
                    &prefix,
                    &self.remote_s3_access_key,
                    &self.remote_s3_secret_key,
                )
                .and_then(|S3Store { store, runtime }| {
                    OmeZarrDataset::open_with_store(
                        DatasetSource::S3 {
                            endpoint: endpoint.clone(),
                            region: region.clone(),
                            bucket: bucket.clone(),
                            prefix: prefix.clone(),
                        },
                        store.clone(),
                    )
                    .map(|dataset| (dataset, store, runtime))
                }) {
                    Ok((dataset, store, runtime)) => {
                        let mut app = OmeZarrViewerApp::new_runtime(
                            ctx,
                            self.gpu_available,
                            dataset,
                            store,
                            self.app_settings.auto_contrast,
                        );
                        app.set_remote_runtime(Some(runtime));
                        self.configure_single_app(&mut app);
                        app.set_project_space(project_space);
                        self.mode = Mode::Single(app);
                    }
                    Err(err) => {
                        let mut ps = project_space;
                        ps.set_status(format!("Open remote dataset failed: {err}"));
                        self.mode = Mode::Project { project_space: ps };
                    }
                }
            }
        }
    }

    fn open_project_roi(
        &mut self,
        ctx: &egui::Context,
        roi: ProjectRoi,
        project_space: ProjectSpace,
    ) {
        let Some(source) = roi.dataset_source() else {
            let mut ps = project_space;
            ps.set_status("Project ROI has no dataset source configured.".to_string());
            self.mode = Mode::Project { project_space: ps };
            return;
        };
        let cached_objects = self.cached_project_object_layer(&project_space, &roi);
        self.open_dataset_source(ctx, source, project_space);
        if let (Some(preloaded), Mode::Single(app)) = (cached_objects.as_ref(), &mut self.mode) {
            log_warn!(
                "project preload: installing cached object segmentation for {}",
                roi.source_display()
            );
            app.install_preloaded_project_segmentation(preloaded);
        }
    }

    fn open_mosaic_from_project(
        &mut self,
        ctx: &egui::Context,
        rois: Vec<ProjectRoi>,
        project_space: ProjectSpace,
    ) {
        let ret = ReturnToSingleState { dataset_root: None };
        let project_dir = project_space.project_dir();
        if rois.len() < 2 {
            let mut ps = project_space;
            ps.set_status("Need at least 2 ROIs to open mosaic.".to_string());
            self.mode = Mode::Project { project_space: ps };
            return;
        }
        let cached_objects = self.cached_project_object_layers_for_rois(&project_space, &rois);
        let mosaic_result =
            MosaicViewerApp::from_project_rois(ctx, self.gpu_available, rois, project_dir, None);
        match mosaic_result {
            Ok(mut mosaic) => {
                self.configure_mosaic_app(&mut mosaic);
                if !cached_objects.is_empty() {
                    let installed = mosaic.install_preloaded_project_segmentations(&cached_objects);
                    log_warn!(
                        "project preload: installed cached object segmentations for {installed} mosaic ROI(s)"
                    );
                }
                mosaic.set_project_space(project_space);
                self.mode = Mode::Mosaic { mosaic, ret };
            }
            Err(err) => {
                let mut ps = project_space;
                ps.set_status(format!("Open mosaic failed: {err}"));
                self.mode = Mode::Project { project_space: ps };
            }
        }
    }

    fn switch_single_to_mosaic(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        let prev = std::mem::replace(&mut self.mode, Mode::Transition);
        let Mode::Single(mut single) = prev else {
            self.mode = prev;
            return;
        };

        let ret = ReturnToSingleState {
            dataset_root: single.current_local_dataset_root(),
        };
        let project_space = single.take_project_space();
        let project_rois = project_space.rois_for_local_paths(&paths);
        let project_dir = project_space.project_dir();
        let cached_objects =
            self.cached_project_object_layers_for_rois(&project_space, &project_rois);
        let mosaic_result = if project_rois.len() >= 2 {
            MosaicViewerApp::from_project_rois(
                ctx,
                self.gpu_available,
                project_rois,
                project_dir,
                None,
            )
        } else {
            MosaicViewerApp::from_local_paths(ctx, self.gpu_available, paths, None)
        };

        match mosaic_result {
            Ok(mut mosaic) => {
                self.configure_mosaic_app(&mut mosaic);
                if !cached_objects.is_empty() {
                    let installed = mosaic.install_preloaded_project_segmentations(&cached_objects);
                    log_warn!(
                        "project preload: installed cached object segmentations for {installed} mosaic ROI(s)"
                    );
                }
                mosaic.set_project_space(project_space);
                self.mode = Mode::Mosaic { mosaic, ret };
            }
            Err(err) => {
                let mut single = single;
                let mut ps = project_space;
                ps.set_status(format!("Open mosaic failed: {err}"));
                single.set_project_space(ps);
                self.mode = Mode::Single(single);
            }
        }
    }

    fn switch_mosaic_to_single(&mut self, ctx: &egui::Context) {
        let prev = std::mem::replace(&mut self.mode, Mode::Transition);
        let Mode::Mosaic { mosaic, ret } = prev else {
            self.mode = prev;
            return;
        };

        let mut mosaic = mosaic;
        let project_space = mosaic.take_project_space();

        let Some(root) = ret.dataset_root.clone() else {
            // No known return target; return to project landing.
            self.mode = Mode::Project { project_space };
            return;
        };

        match OmeZarrDataset::open_local(&root) {
            Ok((dataset, store)) => {
                let mut app = OmeZarrViewerApp::new_runtime(
                    ctx,
                    self.gpu_available,
                    dataset,
                    store,
                    self.app_settings.auto_contrast,
                );
                self.configure_single_app(&mut app);
                app.set_project_space(project_space);
                self.mode = Mode::Single(app);
            }
            Err(err) => {
                // If reopen fails, fall back to staying in mosaic.
                eprintln!("Back failed: {err}");
                mosaic.set_project_space(project_space);
                self.mode = Mode::Mosaic { mosaic, ret };
            }
        }
    }

    fn ui_spatial_open_dialog(&mut self, ctx: &egui::Context) {
        let mut open_clicked = false;
        let mut cancel_clicked = false;

        {
            let Some(dlg) = self.spatial_open.as_mut() else {
                return;
            };

            egui::Window::new("Open SpatialData")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(dlg.discovery.root.to_string_lossy().to_string());
                    ui.add_space(6.0);

                    ui.label("Image");
                    egui::ComboBox::from_id_salt("spatial_image")
                        .selected_text(
                            dlg.discovery
                                .images
                                .get(dlg.selected_image)
                                .map(|e| e.name.clone())
                                .unwrap_or_else(|| "(none)".to_string()),
                        )
                        .show_ui(ui, |ui| {
                            for (i, e) in dlg.discovery.images.iter().enumerate() {
                                ui.selectable_value(&mut dlg.selected_image, i, e.name.clone());
                            }
                        });

                    ui.separator();
                    ui.label("Overlays");

                    ui.horizontal(|ui| {
                        ui.label("Labels");
                        let selected_text = dlg
                            .selected_labels
                            .and_then(|i| dlg.discovery.labels.get(i))
                            .map(|e| e.name.clone())
                            .unwrap_or_else(|| "None".to_string());
                        egui::ComboBox::from_id_salt("spatial_labels")
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut dlg.selected_labels, None, "None");
                                for (i, e) in dlg.discovery.labels.iter().enumerate() {
                                    ui.selectable_value(
                                        &mut dlg.selected_labels,
                                        Some(i),
                                        e.name.clone(),
                                    );
                                }
                            });
                    });

                    ui.horizontal(|ui| {
                        ui.label("Shapes");
                        if dlg.selected_shapes.is_empty() {
                            ui.label("None");
                        } else {
                            ui.label(format!("{} selected", dlg.selected_shapes.len()));
                        }
                    });
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_min_width(240.0);
                        for (i, e) in dlg.discovery.shapes.iter().enumerate() {
                            let mut selected = dlg.selected_shapes.contains(&i);
                            if ui.checkbox(&mut selected, e.name.as_str()).changed() {
                                if selected {
                                    if !dlg.selected_shapes.contains(&i) {
                                        dlg.selected_shapes.push(i);
                                        dlg.selected_shapes.sort_unstable();
                                    }
                                } else {
                                    dlg.selected_shapes.retain(|&idx| idx != i);
                                }
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Points");
                        let selected_text = dlg
                            .selected_points
                            .and_then(|i| dlg.discovery.points.get(i))
                            .map(|e| e.name.clone())
                            .unwrap_or_else(|| "None".to_string());
                        egui::ComboBox::from_id_salt("spatial_points")
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut dlg.selected_points, None, "None");
                                for (i, e) in dlg.discovery.points.iter().enumerate() {
                                    ui.selectable_value(
                                        &mut dlg.selected_points,
                                        Some(i),
                                        e.name.clone(),
                                    );
                                }
                            });
                        let mut all = dlg.points_max == 0;
                        if ui
                            .checkbox(&mut all, "All")
                            .on_hover_text("Load all points (may be slow / memory-heavy).")
                            .changed()
                        {
                            dlg.points_max = if all { 0 } else { 200_000 };
                        }
                        ui.add(
                            egui::DragValue::new(&mut dlg.points_max)
                                .speed(1)
                                .range(0..=200_000_000)
                                .prefix("Max "),
                        )
                        .on_hover_text("0 means no cap (load all).");
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        open_clicked = ui.button("Open").clicked();
                        cancel_clicked = ui.button("Cancel").clicked();
                    });

                    if !dlg.status.is_empty() {
                        ui.add_space(6.0);
                        ui.label(dlg.status.clone());
                    }
                });
        }

        if cancel_clicked {
            self.spatial_open = None;
            return;
        }
        if !open_clicked {
            return;
        }

        let (root, img, labels, tables, shapes, points, points_max) = {
            let Some(dlg) = self.spatial_open.as_ref() else {
                return;
            };
            let root = dlg.discovery.root.clone();
            let img = dlg.discovery.images.get(dlg.selected_image).cloned();
            let labels = dlg
                .selected_labels
                .and_then(|i| dlg.discovery.labels.get(i))
                .cloned();
            let tables = dlg.discovery.tables.clone();
            let shapes = dlg
                .selected_shapes
                .iter()
                .filter_map(|&i| dlg.discovery.shapes.get(i).cloned())
                .collect::<Vec<_>>();
            let points = dlg
                .selected_points
                .and_then(|i| dlg.discovery.points.get(i))
                .cloned();
            (root, img, labels, tables, shapes, points, dlg.points_max)
        };

        let Some(img) = img else {
            if let Some(dlg) = self.spatial_open.as_mut() {
                dlg.status = "No image selected.".to_string();
            }
            return;
        };

        // Take the project space from the current mode (the dialog always runs in Project mode).
        let project_space = match std::mem::replace(&mut self.mode, Mode::Transition) {
            Mode::Project { project_space } => project_space,
            other => {
                self.mode = other;
                if let Some(dlg) = self.spatial_open.as_mut() {
                    dlg.status = "Internal error: not in Project mode.".to_string();
                }
                return;
            }
        };

        let img_root = root.join(&img.rel_group);
        match OmeZarrDataset::open_local(&img_root) {
            Ok((dataset, store)) => {
                let mut app = OmeZarrViewerApp::new_runtime(
                    ctx,
                    self.gpu_available,
                    dataset,
                    store,
                    self.app_settings.auto_contrast,
                );
                self.configure_single_app(&mut app);
                app.set_project_space(project_space);
                app.attach_spatialdata_layers(
                    root,
                    img.transform,
                    Vec::new(),
                    labels,
                    tables,
                    shapes,
                    points.map(|e| (e, points_max)),
                );
                if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
                    app.fit_to_viewport(viewport);
                }
                self.mode = Mode::Single(app);
                self.spatial_open = None;
            }
            Err(err) => {
                self.mode = Mode::Project { project_space };
                if let Some(dlg) = self.spatial_open.as_mut() {
                    dlg.status = format!("Open image failed: {err}");
                }
            }
        }
    }
}

impl eframe::App for RootApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.handle_viewport_screenshot_events(ctx);
        self.control_mutated_this_frame = false;
        self.process_control_requests(ctx);
        if let Some(rx) = self.deep_link_rx.as_ref() {
            ctx.request_repaint_after(Duration::from_millis(100));
            let mut received_deep_link = false;
            while let Ok(request) = rx.try_recv() {
                log_warn!("deep_link: received {:?}", request);
                self.pending_deep_link = Some(request);
                received_deep_link = true;
            }
            if received_deep_link {
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
                    egui::UserAttentionType::Informational,
                ));
            }
        }
        self.poll_project_object_preload();
        if self.object_preload_rx.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        let current_project_path = match &self.mode {
            Mode::Project { project_space } => project_space.saved_project_path(),
            Mode::Single(app) => app.project_space().saved_project_path(),
            Mode::Mosaic { mosaic, .. } => mosaic.project_space().saved_project_path(),
            Mode::Transition => None,
        };
        self.sync_project_object_preload_scope(current_project_path);

        let open_mosaic: Option<Vec<PathBuf>> = None;
        let mut open_remote_single: Option<(
            OmeZarrDataset,
            Arc<dyn zarrs::storage::ReadableStorageTraits>,
            Option<Arc<tokio::runtime::Runtime>>,
            ProjectSpace,
        )> = None;
        let mut open_remote_s3_mosaic: Option<(Vec<crate::app::S3DatasetSelection>, ProjectSpace)> =
            None;
        let mut back_to_single = false;
        let mut open_single: Option<(PathBuf, ProjectSpace)> = None;
        let mut open_project_roi: Option<(ProjectRoi, ProjectSpace, Option<DeepLinkRequest>)> =
            None;
        let mut open_project_path: Option<PathBuf> = None;
        let mut forget_recent_project_path: Option<PathBuf> = None;
        let mut clear_recent_projects = false;
        let mut open_mosaic_from_project: Option<(Vec<ProjectRoi>, ProjectSpace)> = None;

        if let Some(req) = self.pending_deep_link.take() {
            let mut req = req;
            let deep_link_started = Instant::now();
            if let Some(example) = req.example.clone() {
                apply_example_defaults(&mut req, &example);
                if req.project_path.is_none() {
                    req.project_path = resolve_example_project_path(&example);
                }
                if req.project_path.is_none() {
                    log_warn!("deep_link: unknown or unavailable example '{example}'");
                }
            }
            log_warn!("deep_link: handling {:?}", req);
            let previous_mode = std::mem::replace(&mut self.mode, Mode::Transition);
            let (mut project_space, single_restore, mosaic_restore) = match previous_mode {
                Mode::Project { project_space } => (project_space, None, None),
                Mode::Single(mut app) => (app.take_project_space(), Some(app), None),
                Mode::Mosaic { mut mosaic, ret } => {
                    (mosaic.take_project_space(), None, Some((mosaic, ret)))
                }
                Mode::Transition => (ProjectSpace::default(), None, None),
            };

            let mut status = None;
            if let Some(path) = req.project_path.as_deref() {
                if project_space.saved_project_path().as_deref() == Some(path) {
                    log_warn!(
                        "deep_link: project already loaded: {} ({} ROIs)",
                        path.display(),
                        project_space.config().rois.len()
                    );
                    self.record_recent_project(path);
                } else {
                    log_warn!("deep_link: loading project {}", path.display());
                    match project_space.load_from_file(path) {
                        Ok(()) => {
                            log_warn!(
                                "deep_link: loaded project {} ({} ROIs) after {:.3}s",
                                path.display(),
                                project_space.config().rois.len(),
                                deep_link_started.elapsed().as_secs_f32()
                            );
                            self.record_recent_project(path);
                        }
                        Err(err) => {
                            log_warn!("deep_link: project load failed: {err:?}");
                            status = Some(format!("Deep link project load failed: {err}"));
                        }
                    }
                }
            }

            if let Some(status) = status {
                log_warn!("deep_link: aborting: {status}");
                project_space.set_status(status);
                if let Some(mut app) = single_restore {
                    app.set_project_space(project_space);
                    self.mode = Mode::Single(app);
                } else if let Some((mut mosaic, ret)) = mosaic_restore {
                    mosaic.set_project_space(project_space);
                    self.mode = Mode::Mosaic { mosaic, ret };
                } else {
                    self.mode = Mode::Project { project_space };
                }
            } else {
                match project_space.roi_for_link_target(req.roi.as_deref(), req.sample.as_deref()) {
                    Ok(roi) => {
                        log_warn!(
                            "deep_link: resolved roi={:?} sample={:?} to {}",
                            req.roi,
                            req.sample,
                            roi.source_display()
                        );
                        if let Some(mut app) = single_restore {
                            if app.is_viewing_project_roi(&roi) {
                                log_warn!(
                                    "deep_link: reusing already open ROI {}",
                                    roi.source_display()
                                );
                                app.set_project_space(project_space);
                                log_warn!("deep_link: applying view request {:?}", req);
                                let apply_started = Instant::now();
                                app.apply_deep_link_request(&req);
                                log_warn!(
                                    "deep_link: applied view request to existing ROI after {:.3}s (total {:.3}s)",
                                    apply_started.elapsed().as_secs_f32(),
                                    deep_link_started.elapsed().as_secs_f32()
                                );
                                self.mode = Mode::Single(app);
                            } else {
                                open_project_roi = Some((roi, project_space, Some(req)));
                            }
                        } else {
                            open_project_roi = Some((roi, project_space, Some(req)));
                        }
                    }
                    Err(err) => {
                        log_warn!("deep_link: ROI resolution failed: {err}");
                        project_space.set_status(err);
                        if let Some(mut app) = single_restore {
                            app.set_project_space(project_space);
                            self.mode = Mode::Single(app);
                        } else if let Some((mut mosaic, ret)) = mosaic_restore {
                            mosaic.set_project_space(project_space);
                            self.mode = Mode::Mosaic { mosaic, ret };
                        } else {
                            self.mode = Mode::Project { project_space };
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if self.native_menu.is_none() {
                if let Ok(m) = NativeMenu::init("odon", self.view_show_scale_bar) {
                    self.native_menu = Some(m);
                }
            }
            if let Some(menu) = self.native_menu.as_ref() {
                for action in menu.drain_actions() {
                    match action {
                        NativeMenuAction::Settings => {
                            self.settings_open = true;
                        }
                        NativeMenuAction::OpenOmeZarr => {
                            if let Some(root) =
                                FileDialog::new().set_title("Open OME-Zarr").pick_folder()
                            {
                                let ps = match &mut self.mode {
                                    Mode::Project { project_space } => {
                                        let mut ps = std::mem::take(project_space);
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Single(app) => {
                                        let mut ps = app.take_project_space();
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Mosaic { mosaic, .. } => {
                                        let mut ps = mosaic.take_project_space();
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Transition => ProjectSpace::default(),
                                };
                                open_single = Some((root, ps));
                            }
                        }
                        NativeMenuAction::OpenTiff => {
                            if let Some(root) = FileDialog::new()
                                .add_filter("TIFF / OME-TIFF", &["tif", "tiff"])
                                .set_title("Open TIFF / OME-TIFF")
                                .pick_file()
                            {
                                let ps = match &mut self.mode {
                                    Mode::Project { project_space } => {
                                        let mut ps = std::mem::take(project_space);
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Single(app) => {
                                        let mut ps = app.take_project_space();
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Mosaic { mosaic, .. } => {
                                        let mut ps = mosaic.take_project_space();
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Transition => ProjectSpace::default(),
                                };
                                open_single = Some((root, ps));
                            }
                        }
                        NativeMenuAction::OpenProject => {
                            if let Some(path) = FileDialog::new()
                                .add_filter("Project JSON", &["json"])
                                .set_title("Load Project")
                                .pick_file()
                            {
                                self.load_project_into_current_mode(&path);
                            }
                        }
                        NativeMenuAction::SaveProject => {
                            let save_target = match &self.mode {
                                Mode::Project { project_space } => {
                                    project_space.saved_project_path()
                                }
                                Mode::Single(app) => app.project_space().saved_project_path(),
                                Mode::Mosaic { mosaic, .. } => {
                                    mosaic.project_space().saved_project_path()
                                }
                                Mode::Transition => None,
                            };
                            if let Some(path) = save_target {
                                match &mut self.mode {
                                    Mode::Project { project_space } => {
                                        if let Err(err) = project_space.save_to_file(&path) {
                                            project_space
                                                .set_status(format!("Save project failed: {err}"));
                                        }
                                    }
                                    Mode::Single(app) => {
                                        let mut ps = app.take_project_space();
                                        if let Err(err) = ps.save_to_file(&path) {
                                            ps.set_status(format!("Save project failed: {err}"));
                                        }
                                        app.set_project_space(ps);
                                    }
                                    Mode::Mosaic { mosaic, .. } => {
                                        let mut ps = mosaic.take_project_space();
                                        if let Err(err) = ps.save_to_file(&path) {
                                            ps.set_status(format!("Save project failed: {err}"));
                                        }
                                        mosaic.set_project_space(ps);
                                    }
                                    Mode::Transition => {}
                                }
                            } else {
                                match &mut self.mode {
                                    Mode::Project { project_space } => {
                                        project_space.save_as_project()
                                    }
                                    Mode::Single(app) => {
                                        let mut ps = app.take_project_space();
                                        ps.save_as_project();
                                        app.set_project_space(ps);
                                    }
                                    Mode::Mosaic { mosaic, .. } => {
                                        let mut ps = mosaic.take_project_space();
                                        ps.save_as_project();
                                        mosaic.set_project_space(ps);
                                    }
                                    Mode::Transition => {}
                                }
                            }
                        }
                        NativeMenuAction::SaveNewProject => match &mut self.mode {
                            Mode::Project { project_space } => project_space.save_new_project(),
                            Mode::Single(app) => {
                                let mut ps = app.take_project_space();
                                ps.save_new_project();
                                app.set_project_space(ps);
                            }
                            Mode::Mosaic { mosaic, .. } => {
                                let mut ps = mosaic.take_project_space();
                                ps.save_new_project();
                                mosaic.set_project_space(ps);
                            }
                            Mode::Transition => {}
                        },
                        NativeMenuAction::SaveScreenshot => {
                            self.save_screenshot_via_dialog();
                        }
                        NativeMenuAction::QuickScreenshot => {
                            self.quick_screenshot();
                        }
                        NativeMenuAction::ScreenshotSettings => match &mut self.mode {
                            Mode::Single(app) => app.open_screenshot_settings(),
                            Mode::Project { project_space } => project_space.set_status(
                                "Screenshot Settings: open a dataset first.".to_string(),
                            ),
                            Mode::Mosaic { mosaic, .. } => mosaic.open_screenshot_settings(),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::RoiInfo => match &mut self.mode {
                            Mode::Single(app) => app.open_roi_info_window(),
                            Mode::Project { project_space } => project_space
                                .set_status("ROI Info: open a dataset first.".to_string()),
                            Mode::Mosaic { mosaic, .. } => mosaic
                                .project_space_mut()
                                .set_status("ROI Info: open a single ROI first.".to_string()),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::AddAnnotations => match &mut self.mode {
                            Mode::Single(app) => app.add_annotation_layer_from_menu(),
                            Mode::Project { project_space } => project_space
                                .set_status("Add annotations: open a dataset first.".to_string()),
                            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().set_status(
                                "Add annotations: open a single ROI first.".to_string(),
                            ),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::LoadSegGeoJson => match &mut self.mode {
                            Mode::Single(app) => app.open_seg_geojson_dialog(),
                            Mode::Project { project_space } => project_space
                                .set_status("Load Seg GeoJSON: open a dataset first.".to_string()),
                            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().set_status(
                                "Load Seg GeoJSON: open a single ROI first.".to_string(),
                            ),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::LoadSegObjects => match &mut self.mode {
                            Mode::Single(app) => app.open_seg_objects_dialog(),
                            Mode::Project { project_space } => project_space
                                .set_status("Load Seg Objects: open a dataset first.".to_string()),
                            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().set_status(
                                "Load Seg Objects: open a single ROI first.".to_string(),
                            ),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::ExportMasksGeoJson => {
                            if let Some(path) = FileDialog::new()
                                .add_filter("GeoJSON", &["geojson", "json"])
                                .set_file_name("masks.geojson")
                                .set_title("Export Masks GeoJSON")
                                .save_file()
                            {
                                match &mut self.mode {
                                    Mode::Single(app) => match app.export_masks_geojson(&path) {
                                        Ok(()) => app.set_status(format!(
                                            "Exported masks -> {}",
                                            path.to_string_lossy()
                                        )),
                                        Err(err) => {
                                            app.set_status(format!("Export masks failed: {err}"))
                                        }
                                    },
                                    Mode::Project { project_space } => project_space.set_status(
                                        "Export masks failed: open a dataset first.".to_string(),
                                    ),
                                    Mode::Mosaic { mosaic, .. } => {
                                        mosaic.project_space_mut().set_status(
                                            "Export masks failed: open a single ROI first."
                                                .to_string(),
                                        )
                                    }
                                    Mode::Transition => {}
                                }
                            }
                        }
                        NativeMenuAction::SetScaleBarVisible(visible) => {
                            self.view_show_scale_bar = visible;
                            if let Mode::Single(app) = &mut self.mode {
                                app.set_show_scale_bar(visible);
                            }
                        }
                        NativeMenuAction::CloseWindow | NativeMenuAction::Quit => {
                            let should_close = match &mut self.mode {
                                Mode::Project { .. } => {
                                    if self.close_dialog_open {
                                        self.close_dialog_open = false;
                                        true
                                    } else {
                                        self.close_dialog_open = true;
                                        false
                                    }
                                }
                                Mode::Single(app) => app.confirm_or_request_close_dialog(),
                                Mode::Mosaic { mosaic, .. } => {
                                    mosaic.confirm_or_request_close_dialog()
                                }
                                Mode::Transition => false,
                            };
                            if should_close {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        }
                    }
                }
            }
        }

        if !ctx.wants_keyboard_input()
            && ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Comma))
        {
            self.settings_open = true;
        }

        let object_preload_cached = self.object_preload_cache.len();
        let object_preload_total = self.object_preload_total;
        let object_preload_done = self.object_preload_done;
        let object_preload_failed = self.object_preload_failed;
        let object_preload_loading = self.object_preload_rx.is_some();
        let object_preload_settings = self.object_preload_settings;
        let mut object_preload_start = None;
        let mut object_preload_clear = false;
        let external_layers = self
            .control_bridge
            .as_ref()
            .map(OdonControlBridge::external_layers)
            .filter(|(revision, _, _)| *revision != self.control_external_revision);
        if self.control_bridge.is_some() {
            let observed = self.control_observed_snapshot();
            if let Some(bridge) = self.control_bridge.as_ref() {
                bridge.render_extension_ui(ctx, &observed);
            }
        }
        self.sync_control_manifest_to_project();

        match &mut self.mode {
            Mode::Project { project_space } => {
                project_space.set_recent_projects(&self.app_settings.recent_projects);
                let dropped = ctx.input(|i| i.raw.dropped_files.clone());
                if !dropped.is_empty() {
                    project_space.handle_dropped_paths(
                        dropped
                            .into_iter()
                            .filter_map(|f| f.path)
                            .collect::<Vec<_>>(),
                    );
                }

                // Napari-like "close window" prompt:
                // - Cmd/Ctrl+W opens confirmation
                // - Cmd/Ctrl+W again confirms close
                if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open)
                    || top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open)
                {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }

                // Minimal "landing" UI: show the project workspace and let users open datasets.
                egui::TopBottomPanel::top("top").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("odon");
                        ui.add_space(8.0);
                        ui.label("Project");
                    });
                });
                egui::CentralPanel::default().show(ctx, |ui| {
                    project_space.set_object_cache_ui_state(project_object_cache_ui_state(
                        project_space,
                        object_preload_cached,
                        object_preload_total,
                        object_preload_done,
                        object_preload_failed,
                        object_preload_loading,
                        object_preload_settings,
                    ));
                    let action = project_space.ui(ui, None);
                    if let Some(action) = action {
                        match action {
                            ProjectSpaceAction::Open(roi) => {
                                let ps = std::mem::take(project_space);
                                open_project_roi = Some((roi, ps, None));
                            }
                            ProjectSpaceAction::OpenView(roi, spec) => {
                                let req = spec.to_deep_link_request(None);
                                let ps = std::mem::take(project_space);
                                open_project_roi = Some((roi, ps, Some(req)));
                            }
                            ProjectSpaceAction::OpenProject(path) => {
                                open_project_path = Some(path);
                            }
                            ProjectSpaceAction::OpenLocalPath(path) => {
                                let mut ps = std::mem::take(project_space);
                                ps.handle_dropped_paths([path.clone()]);
                                open_single = Some((path, ps));
                            }
                            ProjectSpaceAction::ForgetRecentProject(path) => {
                                forget_recent_project_path = Some(path);
                            }
                            ProjectSpaceAction::ClearRecentProjects => {
                                clear_recent_projects = true;
                            }
                            ProjectSpaceAction::CaptureCurrentView => {}
                            ProjectSpaceAction::OpenMosaic(rois) => {
                                let ps = std::mem::take(project_space);
                                open_mosaic_from_project = Some((rois, ps));
                            }
                            ProjectSpaceAction::OpenRemoteDialog => {
                                self.remote_dialog_open = true;
                                self.remote_status.clear();
                            }
                            ProjectSpaceAction::PreloadObjectSegmentations(mode) => {
                                object_preload_start = Some((project_space.clone(), mode));
                            }
                            ProjectSpaceAction::ClearObjectCache => {
                                object_preload_clear = true;
                            }
                            ProjectSpaceAction::ShowHelp(topic) => {
                                self.active_help_topic = Some(topic);
                            }
                        }
                    }
                });
                if let Some(action) = project_space.ui_floating_windows(ctx, false) {
                    match action {
                        ProjectSpaceAction::Open(roi) => {
                            let ps = std::mem::take(project_space);
                            open_project_roi = Some((roi, ps, None));
                        }
                        ProjectSpaceAction::OpenView(roi, spec) => {
                            let req = spec.to_deep_link_request(None);
                            let ps = std::mem::take(project_space);
                            open_project_roi = Some((roi, ps, Some(req)));
                        }
                        ProjectSpaceAction::OpenProject(path) => {
                            open_project_path = Some(path);
                        }
                        ProjectSpaceAction::OpenLocalPath(path) => {
                            let mut ps = std::mem::take(project_space);
                            ps.handle_dropped_paths([path.clone()]);
                            open_single = Some((path, ps));
                        }
                        ProjectSpaceAction::ForgetRecentProject(path) => {
                            forget_recent_project_path = Some(path);
                        }
                        ProjectSpaceAction::ClearRecentProjects => {
                            clear_recent_projects = true;
                        }
                        ProjectSpaceAction::CaptureCurrentView => {}
                        ProjectSpaceAction::OpenMosaic(rois) => {
                            let ps = std::mem::take(project_space);
                            open_mosaic_from_project = Some((rois, ps));
                        }
                        ProjectSpaceAction::OpenRemoteDialog => {
                            self.remote_dialog_open = true;
                            self.remote_status.clear();
                        }
                        ProjectSpaceAction::PreloadObjectSegmentations(mode) => {
                            object_preload_start = Some((project_space.clone(), mode));
                        }
                        ProjectSpaceAction::ClearObjectCache => {
                            object_preload_clear = true;
                        }
                        ProjectSpaceAction::ShowHelp(topic) => {
                            self.active_help_topic = Some(topic);
                        }
                    }
                }

                // Startup open (e.g. when launched with a dataset path that isn't a direct OME image root).
                if open_single.is_none() && self.spatial_open.is_none() {
                    if let Some(root) = self.pending_open_root.take() {
                        let ps = std::mem::take(project_space);
                        open_single = Some((root, ps));
                    }
                }
            }
            Mode::Single(app) => {
                if let Some((revision, layers, resources)) = external_layers.as_ref() {
                    app.sync_control_external_layers(layers, resources);
                    self.control_external_revision = *revision;
                }
                app.project_space_mut()
                    .set_recent_projects(&self.app_settings.recent_projects);
                app.set_project_object_cache_ui_state(project_object_cache_ui_state(
                    app.project_space(),
                    object_preload_cached,
                    object_preload_total,
                    object_preload_done,
                    object_preload_failed,
                    object_preload_loading,
                    object_preload_settings,
                ));
                app.update(ctx, frame);
                self.label_prompt_preference = app.label_prompt_preference();
                if let Some(req) = app.take_request() {
                    match req {
                        ViewerRequest::OpenProjectRoi(roi) => {
                            let ps = app.take_project_space();
                            open_project_roi = Some((roi, ps, None));
                        }
                        ViewerRequest::OpenProjectRoiView(roi, spec) => {
                            let req = spec.to_deep_link_request(None);
                            if app.is_viewing_project_roi(&roi) {
                                app.apply_deep_link_request(&req);
                            } else {
                                let ps = app.take_project_space();
                                open_project_roi = Some((roi, ps, Some(req)));
                            }
                        }
                        ViewerRequest::OpenProject(path) => {
                            open_project_path = Some(path);
                        }
                        ViewerRequest::OpenLocalPath(path) => {
                            let mut ps = app.take_project_space();
                            ps.handle_dropped_paths([path.clone()]);
                            open_single = Some((path, ps));
                        }
                        ViewerRequest::ForgetRecentProject(path) => {
                            forget_recent_project_path = Some(path);
                        }
                        ViewerRequest::ClearRecentProjects => {
                            clear_recent_projects = true;
                        }
                        ViewerRequest::OpenProjectMosaic(rois) => {
                            let ps = app.take_project_space();
                            open_mosaic_from_project = Some((rois, ps));
                        }
                        ViewerRequest::OpenRemoteS3Mosaic(datasets) => {
                            let ps = app.take_project_space();
                            open_remote_s3_mosaic = Some((datasets, ps));
                        }
                        ViewerRequest::PreloadObjectSegmentations(project_space, mode) => {
                            object_preload_start = Some((project_space, mode));
                        }
                        ViewerRequest::ClearObjectCache => {
                            object_preload_clear = true;
                        }
                    }
                }
            }
            Mode::Mosaic { mosaic, .. } => {
                mosaic
                    .project_space_mut()
                    .set_recent_projects(&self.app_settings.recent_projects);
                mosaic.set_project_object_cache_ui_state(project_object_cache_ui_state(
                    mosaic.project_space(),
                    object_preload_cached,
                    object_preload_total,
                    object_preload_done,
                    object_preload_failed,
                    object_preload_loading,
                    object_preload_settings,
                ));
                let dropped = ctx.input(|i| i.raw.dropped_files.clone());
                if !dropped.is_empty() {
                    mosaic.project_space_mut().handle_dropped_paths(
                        dropped
                            .into_iter()
                            .filter_map(|f| f.path)
                            .collect::<Vec<_>>(),
                    );
                }
                mosaic.update(ctx, frame);
                if let Some(req) = mosaic.take_request() {
                    match req {
                        MosaicRequest::BackToSingle => {
                            back_to_single = true;
                        }
                        MosaicRequest::OpenProjectRoi(roi) => {
                            let ps = mosaic.take_project_space();
                            open_project_roi = Some((roi, ps, None));
                        }
                        MosaicRequest::OpenProjectRoiView(roi, spec) => {
                            let req = spec.to_deep_link_request(None);
                            let ps = mosaic.take_project_space();
                            open_project_roi = Some((roi, ps, Some(req)));
                        }
                        MosaicRequest::OpenProject(path) => {
                            open_project_path = Some(path);
                        }
                        MosaicRequest::OpenLocalPath(path) => {
                            let mut ps = mosaic.take_project_space();
                            ps.handle_dropped_paths([path.clone()]);
                            open_single = Some((path, ps));
                        }
                        MosaicRequest::ForgetRecentProject(path) => {
                            forget_recent_project_path = Some(path);
                        }
                        MosaicRequest::ClearRecentProjects => {
                            clear_recent_projects = true;
                        }
                        MosaicRequest::OpenProjectMosaic(rois) => {
                            let ps = mosaic.take_project_space();
                            open_mosaic_from_project = Some((rois, ps));
                        }
                        MosaicRequest::OpenRemoteDialog => {
                            self.remote_dialog_open = true;
                            self.remote_status.clear();
                        }
                        MosaicRequest::PreloadObjectSegmentations(project_space, mode) => {
                            object_preload_start = Some((project_space, mode));
                        }
                        MosaicRequest::ClearObjectCache => {
                            object_preload_clear = true;
                        }
                    }
                }
            }
            Mode::Transition => {}
        }

        if matches!(self.mode, Mode::Project { .. }) {
            self.ui_spatial_open_dialog(ctx);
        }

        if let Some((project_space, mode)) = object_preload_start {
            self.start_project_object_preload(&project_space, mode);
        }
        if object_preload_clear {
            self.clear_project_object_preload();
        }

        self.ui_settings_dialog(ctx);

        if let Some(action) = self.ui_remote_dialog(ctx) {
            let previous_mode = std::mem::replace(&mut self.mode, Mode::Transition);
            let (project_space, single_restore, mosaic_restore) = match previous_mode {
                Mode::Project { project_space } => (project_space, None, None),
                Mode::Single(mut app) => (app.take_project_space(), Some(app), None),
                Mode::Mosaic { mut mosaic, ret } => {
                    (mosaic.take_project_space(), None, Some((mosaic, ret)))
                }
                Mode::Transition => (ProjectSpace::default(), None, None),
            };
            match action {
                RootRemoteAction::OpenSingle {
                    dataset,
                    store,
                    runtime,
                } => {
                    open_remote_single = Some((dataset, store, runtime, project_space));
                }
                RootRemoteAction::OpenS3Mosaic(datasets) => {
                    open_remote_s3_mosaic = Some((datasets, project_space));
                }
                RootRemoteAction::AddToProject(sources) => {
                    let mut project_space = project_space;
                    let count = sources.len();
                    for source in sources {
                        project_space.add_roi_source(source);
                    }
                    project_space
                        .set_status(format!("Added {count} remote ROI(s) to the project."));
                    if let Some(mut app) = single_restore {
                        app.set_project_space(project_space);
                        self.mode = Mode::Single(app);
                    } else if let Some((mut mosaic, ret)) = mosaic_restore {
                        mosaic.set_project_space(project_space);
                        self.mode = Mode::Mosaic { mosaic, ret };
                    } else {
                        self.mode = Mode::Project { project_space };
                    }
                }
            }
        }

        if let Some(path) = forget_recent_project_path {
            self.forget_recent_project(&path);
        }
        if clear_recent_projects {
            self.clear_recent_projects();
        }
        if let Some(root) = self.pending_control_open_root.take() {
            let project_space = match &mut self.mode {
                Mode::Project { project_space } => {
                    let mut ps = std::mem::take(project_space);
                    ps.handle_dropped_paths([root.clone()]);
                    ps
                }
                Mode::Single(app) => {
                    let mut ps = app.take_project_space();
                    ps.handle_dropped_paths([root.clone()]);
                    ps
                }
                Mode::Mosaic { mosaic, .. } => {
                    let mut ps = mosaic.take_project_space();
                    ps.handle_dropped_paths([root.clone()]);
                    ps
                }
                Mode::Transition => ProjectSpace::default(),
            };
            open_single = Some((root, project_space));
        }
        if let Some(path) = open_project_path {
            self.load_project_into_current_mode(&path);
        }

        if let Some((root, ps)) = open_single {
            self.open_single(ctx, &root, ps);
        }
        if let Some((roi, ps, deep_link)) = open_project_roi {
            if deep_link.is_some() {
                log_warn!("deep_link: opening ROI {}", roi.source_display());
            }
            let open_started = Instant::now();
            self.open_project_roi(ctx, roi, ps);
            if deep_link.is_some() {
                log_warn!(
                    "deep_link: ROI open returned after {:.3}s",
                    open_started.elapsed().as_secs_f32()
                );
            }
            if let (Some(req), Mode::Single(app)) = (deep_link.as_ref(), &mut self.mode) {
                log_warn!("deep_link: applying view request {:?}", req);
                let apply_started = Instant::now();
                app.apply_deep_link_request(req);
                log_warn!(
                    "deep_link: applied view request after {:.3}s",
                    apply_started.elapsed().as_secs_f32()
                );
            }
        }
        if let Some((dataset, store, runtime, project_space)) = open_remote_single {
            let mut app = OmeZarrViewerApp::new_runtime(
                ctx,
                self.gpu_available,
                dataset,
                store,
                self.app_settings.auto_contrast,
            );
            app.set_remote_runtime(runtime);
            self.configure_single_app(&mut app);
            app.set_project_space(project_space);
            self.mode = Mode::Single(app);
        }
        if let Some((paths, ps)) = open_mosaic_from_project {
            self.open_mosaic_from_project(ctx, paths, ps);
        }
        if let Some((datasets, project_space)) = open_remote_s3_mosaic {
            let ret = ReturnToSingleState { dataset_root: None };
            match MosaicViewerApp::from_remote_s3_sources(ctx, self.gpu_available, datasets, None) {
                Ok(mut mosaic) => {
                    self.configure_mosaic_app(&mut mosaic);
                    mosaic.set_project_space(project_space);
                    self.mode = Mode::Mosaic { mosaic, ret };
                }
                Err(err) => {
                    let mut ps = project_space;
                    ps.set_status(format!("Open remote mosaic failed: {err}"));
                    self.mode = Mode::Project { project_space: ps };
                }
            }
        }
        if let Some(paths) = open_mosaic {
            self.switch_single_to_mosaic(ctx, paths);
        }
        if back_to_single {
            self.switch_mosaic_to_single(ctx);
        }
        self.publish_observed_control_changes();
        crate::ui::help::show_help_window(ctx, &mut self.active_help_topic);
    }
}

#[cfg(test)]
mod control_boundary_tests {
    use super::*;

    #[test]
    fn application_errors_become_structured_control_errors() {
        let missing = control_application_error(
            "open_project",
            &serde_json::json!({"error": "Project file does not exist."}),
        )
        .expect("structured error");
        assert_eq!(missing.kind, ControlErrorKind::ResourceNotFound);

        let wrong_mode = control_application_error(
            "get_camera",
            &serde_json::json!({"error": "No dataset viewer is currently open."}),
        )
        .expect("structured error");
        assert_eq!(wrong_mode.kind, ControlErrorKind::WrongMode);

        let nested = control_application_error(
            "viewer.channels.set_contrast",
            &serde_json::json!({
                "mode": "single",
                "contrast": {"error": "channel index 99 is out of range"}
            }),
        )
        .expect("nested application error");
        assert_eq!(nested.kind, ControlErrorKind::ResourceNotFound);

        let conflict = control_application_error(
            "viewer.viewports.camera.set",
            &serde_json::json!({
                "error": "viewport navigation revision conflict: expected 1, current 2"
            }),
        )
        .expect("revision conflict");
        assert_eq!(conflict.kind, ControlErrorKind::Conflict);

        assert!(control_application_error("get_camera", &serde_json::json!({})).is_none());
    }

    #[test]
    fn explicit_viewport_mutations_keep_active_view_legacy_event_compatibility() {
        assert_eq!(
            active_viewport_compatibility_event("viewer.viewports.camera.set"),
            Some("viewer.camera.changed")
        );
        assert_eq!(
            active_viewport_compatibility_event("viewer.viewports.channels.set_color"),
            Some("viewer.channels.changed")
        );
        assert_eq!(
            active_viewport_compatibility_event("viewer.viewports.objects.style.set"),
            Some("viewer.layers.changed")
        );
        assert_eq!(
            active_viewport_compatibility_event("viewer.viewports.get"),
            None
        );
    }

    #[test]
    fn workspace_screenshot_crop_scales_clips_and_rejects_empty_rectangles() {
        assert_eq!(
            screenshot_crop_bounds(
                [800, 600],
                Some(egui::Rect::from_min_max(
                    egui::pos2(10.25, 20.5),
                    egui::pos2(110.75, 70.25),
                )),
                2.0,
            ),
            Some((20, 41, 222, 141))
        );
        assert_eq!(
            screenshot_crop_bounds(
                [100, 80],
                Some(egui::Rect::from_min_max(
                    egui::pos2(-20.0, -10.0),
                    egui::pos2(200.0, 100.0),
                )),
                1.0,
            ),
            Some((0, 0, 100, 80))
        );
        assert_eq!(
            screenshot_crop_bounds(
                [100, 80],
                Some(egui::Rect::from_min_max(
                    egui::pos2(120.0, 5.0),
                    egui::pos2(130.0, 10.0),
                )),
                1.0,
            ),
            None
        );
    }
}
