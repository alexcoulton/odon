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
use crate::app_support::settings::{AppSettings, settings_file_path};
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
use crate::deep_link::DeepLinkRequest;
use crate::mosaic::{MosaicRequest, MosaicViewerApp};
use crate::objects::{ObjectPreloadSettings, PreloadedObjectLayer};
use crate::project::{ProjectObjectCacheUiState, ProjectSpace, ProjectSpaceAction};
use crate::spatialdata::{SpatialDataDiscovery, discover_spatialdata};
use crate::ui::top_bar;
use crate::xenium::discover_xenium_explorer;
use crate::{log_debug, log_info, log_warn};
use odon::mcp::{OdonControlBridge, OdonControlRequest};
use rfd::FileDialog;

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

struct ProjectObjectPreloadEvent {
    path: PathBuf,
    settings: ObjectPreloadSettings,
    result: Result<PreloadedObjectLayer, String>,
    finished: bool,
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

pub struct RootApp {
    mode: Mode,
    gpu_available: bool,
    close_dialog_open: bool,
    spatial_open: Option<SpatialOpenDialog>,
    pending_open_root: Option<PathBuf>,
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
                let msg = format!("MCP control bridge unavailable: {err}");
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
    }

    fn apply_app_settings_to_mode(&mut self) {
        if let Mode::Single(app) = &mut self.mode {
            app.set_auto_contrast_settings(self.app_settings.auto_contrast);
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

    fn process_control_requests(&mut self) {
        let mut requests = Vec::new();
        if let Some(bridge) = self.control_bridge.as_ref() {
            while let Ok(request) = bridge.try_recv() {
                requests.push(request);
            }
        }
        for request in requests {
            self.reply_to_control_request(request);
        }
    }

    fn reply_to_control_request(&mut self, request: OdonControlRequest) {
        let response = match request.method.as_str() {
            "get_current_view" => self.control_current_view(),
            "list_project_rois" => self.control_project_rois(),
            "list_channels" => self.control_channels(),
            "list_visible_channels" => self.control_visible_channels(),
            "get_active_channel" => self.control_active_channel(),
            "set_active_channel" => self.control_set_active_channel(&request.params),
            "set_visible_channels" => self.control_set_visible_channels(&request.params),
            "open_roi" => self.control_open_roi(&request.params),
            "save_project" => self.control_save_project(),
            "get_channel_contrast" => self.control_get_channel_contrast(&request.params),
            "set_channel_contrast" => self.control_set_channel_contrast(&request.params),
            "get_object_overlay_visibility" => {
                self.control_get_object_overlay_visibility(&request.params)
            }
            "set_object_overlay_visibility" => {
                self.control_set_object_overlay_visibility(&request.params)
            }
            "get_channel_intensity_stats" => {
                self.control_get_channel_intensity_stats(&request.params)
            }
            "set_channel_order" => self.control_set_channel_order(&request.params),
            "list_channel_groups" => self.control_list_channel_groups(),
            "set_channel_group" => self.control_set_channel_group(&request.params),
            "get_camera" => self.control_get_camera(),
            "set_camera" => self.control_set_camera(&request.params),
            "zoom_in" => self.control_zoom(&request.params, true),
            "zoom_out" => self.control_zoom(&request.params, false),
            "fit_to_view" => self.control_fit_to_view(),
            "capture_screenshot" => self.control_capture_screenshot(&request.params),
            method => serde_json::json!({
                "error": format!("unknown Odon control method '{method}'"),
            }),
        };
        let _ = request.reply.send(response);
    }

    fn current_project_space(&self) -> Option<&ProjectSpace> {
        match &self.mode {
            Mode::Project { project_space } => Some(project_space),
            Mode::Single(app) => Some(app.project_space()),
            Mode::Mosaic { mosaic, .. } => Some(mosaic.project_space()),
            Mode::Transition => None,
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

    fn control_capture_screenshot(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match &mut self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "screenshot": app.control_capture_screenshot(params),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "screenshot": mosaic.control_capture_screenshot(params),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
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
                ui.checkbox(
                    &mut self.app_settings.auto_contrast.enabled_on_open,
                    "Apply auto contrast when opening a dataset",
                );

                egui::ComboBox::from_label("Method")
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
                ui.label(self.app_settings.auto_contrast.method.description());

                let settings = &mut self.app_settings.auto_contrast;
                match settings.method {
                    crate::app_support::settings::AutoContrastMethod::ZeroToP97 => {
                        ui.horizontal(|ui| {
                            ui.label("Upper percentile");
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
                            ui.add(
                                egui::DragValue::new(&mut settings.lower_percentile)
                                    .range(0..=99)
                                    .speed(0.2)
                                    .suffix("%"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Upper percentile");
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

                ui.add_space(8.0);
                let can_apply_now = matches!(self.mode, Mode::Single(_));
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
        Ok(Self {
            mode: Mode::Project { project_space: ps },
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
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
            #[cfg(target_os = "macos")]
            native_menu: None,
        })
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
        if let Some(path) = project_path.as_deref() {
            let mut ps = ProjectSpace::default();
            if let Err(err) = ps.load_from_file(path) {
                ps.set_status(format!("Load project failed: {err}"));
            }
            app.set_project_space(ps);
        }
        Ok(Self {
            mode: Mode::Single(app),
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
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
            #[cfg(target_os = "macos")]
            native_menu: None,
        })
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
        mosaic.set_layer_groups(ps.layer_groups().clone());
        Ok(Self {
            mode: Mode::Mosaic {
                mosaic,
                ret: ReturnToSingleState { dataset_root: None },
            },
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
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
            #[cfg(target_os = "macos")]
            native_menu: None,
        })
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
        self.process_control_requests();
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
            let deep_link_started = Instant::now();
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
        crate::ui::help::show_help_window(ctx, &mut self.active_help_topic);
    }
}
