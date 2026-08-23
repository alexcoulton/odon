use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::app::{
    LabelPromptSessionPreference, NativeControlIntent, OmeZarrViewerApp, S3DatasetSelection,
    ViewerRequest,
};
use crate::app_support::menu::{NativeMenu, NativeMenuAction};
use crate::app_support::settings::{AppSettings, settings_file_path};
use crate::data::dataset_kind::{
    LocalDatasetKind, classify_local_dataset_path, normalize_local_dataset_path,
};
use crate::data::dataset_source::DatasetSource;
use crate::data::ome::OmeZarrDataset;
use crate::data::project_config::ProjectRoi;
use crate::data::remote_store::{S3BrowseEntry, S3BrowseListing};
use crate::log_warn;
use crate::mosaic::{MosaicRequest, MosaicViewerApp};
use crate::objects::{ObjectPreloadMode, ObjectPreloadSettings, PreloadedObjectLayer};
use crate::project::{ProjectObjectCacheUiState, ProjectSpace, ProjectSpaceAction};
use crate::spatialdata::{SpatialDataDiscovery, discover_spatialdata};
use crate::ui::top_bar;
use odon::control::ControlError;
use odon::control::actor::{
    PresentationCaptureCompletion, PresentationCaptureScope, PresentationPixels, RenderProjection,
};
use odon::deep_link::DeepLinkRequest;
use odon::mcp::OdonControlRuntime;
use odon::model::{ModelMode, ProjectObjectPreloadMode, ProjectObjectPreloadProjection};
use rfd::FileDialog;

mod actor_projection;
mod remote;
#[cfg(test)]
mod tests;

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
    current_prefix: String,
    parent_prefix: Option<String>,
    entries: Vec<S3BrowseEntry>,
    current_is_dataset: bool,
    selected_dataset_prefixes: HashSet<String>,
    listing_cache: HashMap<String, S3BrowseListing>,
}

enum RootRemoteControlPending {
    Configure {
        reply: crossbeam_channel::Receiver<Result<serde_json::Value, ControlError>>,
        browse_prefix: String,
    },
    List {
        reply: crossbeam_channel::Receiver<Result<serde_json::Value, ControlError>>,
    },
    Open {
        reply: crossbeam_channel::Receiver<Result<serde_json::Value, ControlError>>,
    },
}

enum RootRemoteAction {
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

#[derive(Debug)]
struct ViewportScreenshotRequest {
    destination: ViewportScreenshotDestination,
    crop_rect_points: Option<egui::Rect>,
}

#[derive(Debug)]
enum ViewportScreenshotDestination {
    Presentation {
        capture_id: u64,
        tx: crossbeam_channel::Sender<PresentationCaptureCompletion>,
    },
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

fn project_object_cache_ui_state(
    available_count: usize,
    on_disk_bytes: u64,
    cached: usize,
    total: usize,
    done: usize,
    failed: usize,
    loading: bool,
    cached_settings: ObjectPreloadSettings,
) -> ProjectObjectCacheUiState {
    ProjectObjectCacheUiState {
        available_count,
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
    deep_link_rx: Option<Receiver<DeepLinkRequest>>,
    object_preload_cache: HashMap<(PathBuf, ObjectPreloadSettings), Arc<PreloadedObjectLayer>>,
    object_preload_settings: ObjectPreloadSettings,
    object_preload_available_count: usize,
    object_preload_on_disk_bytes: u64,
    object_preload_total: usize,
    object_preload_done: usize,
    object_preload_failed: usize,
    object_preload_loading: bool,
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
    remote_control_pending: Option<RootRemoteControlPending>,
    label_prompt_preference: LabelPromptSessionPreference,
    app_settings: AppSettings,
    settings_open: bool,
    settings_status: String,
    active_help_topic: Option<crate::ui::help::HelpTopic>,
    control_runtime: OdonControlRuntime,
    control_external_revision: u64,
    control_project_revision: u64,
    control_last_observed_at: Instant,
    control_projection_applied_this_frame: Option<u64>,
    control_projection_revision_applied: u64,
    control_document_generation_applied: u64,
    control_actor_mode_signature: Option<String>,
    control_projection_gap: bool,
    deferred_control_projection: Option<RenderProjection>,
    pending_native_control_intents: VecDeque<NativeControlIntent>,
    #[cfg(target_os = "macos")]
    native_menu: Option<NativeMenu>,
}

impl RootApp {
    fn control_actor_mode_signature(&self) -> String {
        match &self.mode {
            Mode::Single(app) => format!("single:{}", app.control_actor_source_key()),
            Mode::Project { .. } => "project".to_string(),
            Mode::Mosaic { mosaic, .. } => mosaic.control_actor_signature(),
            Mode::Transition => "transition".to_string(),
        }
    }

    fn bootstrap_control_actor(&mut self) {
        let signature = self.control_actor_mode_signature();
        self.deferred_control_projection = None;
        let project_snapshot = self
            .current_project_space()
            .map(ProjectSpace::control_actor_project_snapshot);
        let runtime = &self.control_runtime;
        runtime.bootstrap_settings(self.app_settings.clone(), settings_file_path().ok());
        runtime.report_renderer_capabilities(self.gpu_available);
        match &mut self.mode {
            Mode::Single(app) => runtime.bootstrap_dataset_model(
                app.control_actor_dataset(),
                app.control_viewport_workspace_snapshot(),
                app.control_actor_store(),
                app.control_actor_dataset()
                    .source
                    .local_path()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from(app.control_actor_source_key())),
            ),
            Mode::Project { .. } => runtime.bootstrap_model_mode(ModelMode::Project),
            Mode::Mosaic { mosaic, .. } => runtime.bootstrap_mosaic_model(
                mosaic.control_actor_resource(),
                mosaic.control_actor_semantic_snapshot(),
            ),
            Mode::Transition => runtime.bootstrap_model_mode(ModelMode::Transition),
        }
        if let Some(snapshot) = project_snapshot {
            runtime.bootstrap_project_model(snapshot);
        }
        self.control_actor_mode_signature = Some(signature);
    }

    fn sync_control_actor_mode_from_native(&mut self) {
        let signature = self.control_actor_mode_signature();
        if self.control_actor_mode_signature.as_deref() != Some(signature.as_str()) {
            self.bootstrap_control_actor();
        }
    }

    fn report_control_viewport_geometry(&self) {
        let Mode::Single(app) = &self.mode else {
            return;
        };
        let runtime = &self.control_runtime;
        for (viewport_id, x, y, width, height) in app.control_actor_viewport_geometry() {
            let _ = runtime.report_viewport_geometry(viewport_id, x, y, width, height);
        }
    }

    fn report_control_presentation(&mut self) {
        let Some(revision) = self.control_projection_applied_this_frame.take() else {
            return;
        };
        if !self.control_runtime.report_presentation_applied(revision) {
            self.control_projection_applied_this_frame = Some(revision);
        }
    }

    fn spawn_control_runtime(
        ctx: &egui::Context,
        settings_status: &mut String,
    ) -> anyhow::Result<OdonControlRuntime> {
        let object_loader: Arc<dyn odon::model::ObjectResourceLoader> =
            Arc::new(crate::objects::NativeObjectControlService);
        let dataset_inspector: Arc<dyn odon::data::document::DatasetInspector> =
            Arc::new(crate::app_support::datasets::NativeDatasetInspector);
        let alternate_backend: Arc<dyn odon::data::document::AlternateDatasetBackend> =
            Arc::new(crate::app_support::datasets::NativeAlternateDatasetBackend);
        match OdonControlRuntime::spawn_default_with_services(
            ctx.clone(),
            object_loader,
            dataset_inspector,
            alternate_backend,
        ) {
            Ok(bridge) => {
                if let Some(error) = bridge.server_error() {
                    let msg = format!(
                        "Python/MCP control server unavailable ({error}); local actor controls remain active"
                    );
                    if settings_status.trim().is_empty() {
                        *settings_status = msg;
                    } else {
                        settings_status.push_str("; ");
                        settings_status.push_str(&msg);
                    }
                }
                Ok(bridge)
            }
            Err(err) => Err(err.context("could not start Odon's canonical control actor")),
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

    fn process_control_presentations(&mut self, ctx: &egui::Context) {
        let mut updates = Vec::new();
        // The actor channel is latest-value and capacity one. Keep the drain loop so a
        // concurrently published replacement can be consumed in the same frame.
        for _ in 0..2 {
            match self.control_runtime.try_recv_presentation() {
                Ok(update) => updates.push(update),
                Err(_) => break,
            }
        }
        if self.control_runtime.pending_presentation_len() > 0 {
            ctx.request_repaint();
        }
        let latest = self
            .deferred_control_projection
            .take()
            .into_iter()
            .chain(updates)
            .max_by_key(|projection| projection.revision);
        let Some(update) = latest else {
            return;
        };
        let gesture_active = matches!(
            &self.mode,
            Mode::Single(app) if app.control_projection_gesture_active()
        );
        if gesture_active && update.mode == ModelMode::Single {
            self.deferred_control_projection = Some(update);
            ctx.request_repaint_after(Duration::from_millis(5));
            return;
        }

        let revision = update.revision;
        if self.apply_control_presentation(ctx, update) {
            self.control_projection_gap = false;
            self.control_projection_revision_applied =
                self.control_projection_revision_applied.max(revision);
            self.control_projection_applied_this_frame = Some(
                self.control_projection_applied_this_frame
                    .map_or(revision, |current| current.max(revision)),
            );
        } else {
            self.control_projection_gap = true;
        }
    }

    fn process_control_presentation_captures(&mut self, ctx: &egui::Context) {
        let runtime = &self.control_runtime;
        let completion_tx = runtime.presentation_completion_sender();
        let mut requests = Vec::new();
        while let Ok(request) = runtime.try_recv_presentation_capture() {
            requests.push(request);
        }
        for request in requests {
            let capture_id = request.capture_id;
            let result = match request.scope {
                PresentationCaptureScope::Viewer { viewport_id } => match &mut self.mode {
                    Mode::Single(app) if request.mode == ModelMode::Single => app
                        .request_actor_screenshot(
                            capture_id,
                            viewport_id.as_deref(),
                            &request.screenshot_preferences,
                            completion_tx.clone(),
                        )
                        .map_err(|error| error.to_string()),
                    Mode::Mosaic { mosaic, .. } if request.mode == ModelMode::Mosaic => mosaic
                        .request_actor_screenshot(
                            capture_id,
                            &request.screenshot_preferences,
                            completion_tx.clone(),
                        )
                        .map_err(|error| error.to_string()),
                    _ => Err(format!(
                        "renderer mode changed before viewer capture {} could run",
                        request.desired_projection_revision
                    )),
                },
                PresentationCaptureScope::Workspace => match &mut self.mode {
                    Mode::Single(app) if request.mode == ModelMode::Single => {
                        match app.workspace_canvas_rect() {
                            Some(crop_rect_points) => {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                                    egui::UserData::new(ViewportScreenshotRequest {
                                        destination: ViewportScreenshotDestination::Presentation {
                                            capture_id,
                                            tx: completion_tx.clone(),
                                        },
                                        crop_rect_points: Some(crop_rect_points),
                                    }),
                                ));
                                Ok(())
                            }
                            None => {
                                Err("workspace canvases have not been laid out yet".to_string())
                            }
                        }
                    }
                    _ => Err("workspace capture requires the single-image renderer".to_string()),
                },
                PresentationCaptureScope::Window => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                        ViewportScreenshotRequest {
                            destination: ViewportScreenshotDestination::Presentation {
                                capture_id,
                                tx: completion_tx.clone(),
                            },
                            crop_rect_points: None,
                        },
                    )));
                    Ok(())
                }
                PresentationCaptureScope::Project => {
                    if matches!(self.mode, Mode::Project { .. }) {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                            egui::UserData::new(ViewportScreenshotRequest {
                                destination: ViewportScreenshotDestination::Presentation {
                                    capture_id,
                                    tx: completion_tx.clone(),
                                },
                                crop_rect_points: None,
                            }),
                        ));
                        Ok(())
                    } else {
                        Err(
                            "project capture projection was not realized by the renderer"
                                .to_string(),
                        )
                    }
                }
            };
            if let Err(message) = result {
                let _ = completion_tx.send(PresentationCaptureCompletion {
                    capture_id,
                    result: Err(message),
                });
            } else {
                ctx.request_repaint();
            }
        }
    }

    fn process_control_platform_effects(&self, ctx: &egui::Context) {
        let runtime = &self.control_runtime;
        while let Ok(effect) = runtime.try_recv_platform_effect() {
            match effect {
                odon::control::actor::PlatformEffect::CloseWindow { .. } => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
    }

    fn apply_control_presentation(
        &mut self,
        ctx: &egui::Context,
        projection: RenderProjection,
    ) -> bool {
        if self.app_settings != projection.settings {
            self.app_settings = projection.settings.clone();
            self.apply_app_settings_to_mode();
        }
        if let Mode::Single(app) = &mut self.mode {
            app.apply_control_actor_screenshot_preferences(&projection.screenshot_preferences);
            if let Err(error) =
                app.apply_control_actor_tile_loading_policy(&projection.tile_loading_policy)
            {
                log_warn!("could not realize actor tile-loading policy: {error}");
            }
            app.apply_control_actor_pinned_levels(&projection.pinned_levels);
            if let Err(error) = app.apply_control_actor_threshold_preview(
                ctx,
                projection.threshold_preview_generation,
                projection.threshold_preview_pending,
                projection.threshold_preview.as_ref(),
                &projection.threshold_preview_state,
            ) {
                log_warn!("could not realize actor threshold preview: {error}");
            }
            if let Err(error) = app.apply_control_actor_analysis_state(
                projection.analysis_generation,
                &projection.analysis_state,
            ) {
                log_warn!("could not realize actor analysis state: {error}");
            }
            if let Err(error) = app.apply_control_actor_measurement_state(
                projection.measurement_generation,
                &projection.measurement_state,
            ) {
                log_warn!("could not realize actor measurement state: {error}");
            }
            app.apply_control_actor_object_export_state(
                projection.object_export_generation,
                &projection.object_export_state,
            );
        }
        if let Mode::Mosaic { mosaic, .. } = &mut self.mode {
            mosaic.apply_control_actor_screenshot_preferences(&projection.screenshot_preferences);
        }
        self.apply_project_object_preload_projection(&projection.project_object_preload);
        if projection.mode == ModelMode::Project {
            let mut project_space = match &mut self.mode {
                Mode::Project { project_space } => std::mem::take(project_space),
                Mode::Single(app) => app.take_project_space(),
                Mode::Mosaic { mosaic, .. } => mosaic.take_project_space(),
                Mode::Transition => ProjectSpace::default(),
            };
            project_space.apply_control_actor_project_projection(&projection.project);
            self.mode = Mode::Project { project_space };
            self.control_document_generation_applied = projection.document_generation;
            self.control_actor_mode_signature = Some("project".to_string());
            return true;
        }
        if projection.mode == ModelMode::Mosaic {
            let needs_renderer = !matches!(
                &self.mode,
                Mode::Mosaic { mosaic, .. }
                    if mosaic.control_actor_generation() == projection.mosaic_resource_generation
            );
            if needs_renderer {
                let Some(resource) = projection.mosaic_resource.as_ref() else {
                    log_warn!(
                        "actor mosaic projection generation {} has no resource",
                        projection.mosaic_resource_generation
                    );
                    return false;
                };
                if resource.generation != projection.mosaic_resource_generation {
                    log_warn!("actor mosaic projection and resource generations disagree");
                    return false;
                }
                let previous = std::mem::replace(&mut self.mode, Mode::Transition);
                let (mut project_space, ret) = match previous {
                    Mode::Project { project_space } => {
                        (project_space, ReturnToSingleState { dataset_root: None })
                    }
                    Mode::Single(mut app) => {
                        let dataset_root = app.current_local_dataset_root();
                        (
                            app.take_project_space(),
                            ReturnToSingleState { dataset_root },
                        )
                    }
                    Mode::Mosaic { mut mosaic, ret } => (mosaic.take_project_space(), ret),
                    Mode::Transition => (
                        ProjectSpace::default(),
                        ReturnToSingleState { dataset_root: None },
                    ),
                };
                project_space.apply_control_actor_project_projection(&projection.project);
                let mut mosaic =
                    match MosaicViewerApp::from_control_resource(ctx, self.gpu_available, resource)
                    {
                        Ok(mosaic) => mosaic,
                        Err(error) => {
                            log_warn!("could not realize actor mosaic renderer: {error}");
                            self.mode = Mode::Project { project_space };
                            return false;
                        }
                    };
                self.configure_mosaic_app(&mut mosaic);
                mosaic.set_project_space(project_space);
                if let Err(error) = mosaic.apply_control_actor_state(
                    &projection.mosaic_state,
                    &projection.mosaic_object_resources,
                    &projection.mosaic_pinned_levels,
                ) {
                    log_warn!("could not apply actor mosaic state: {error}");
                    self.mode = Mode::Project {
                        project_space: mosaic.take_project_space(),
                    };
                    return false;
                }
                self.mode = Mode::Mosaic { mosaic, ret };
            } else if let Mode::Mosaic { mosaic, .. } = &mut self.mode {
                mosaic
                    .project_space_mut()
                    .apply_control_actor_project_projection(&projection.project);
                if let Err(error) = mosaic.apply_control_actor_state(
                    &projection.mosaic_state,
                    &projection.mosaic_object_resources,
                    &projection.mosaic_pinned_levels,
                ) {
                    log_warn!("could not apply actor mosaic state: {error}");
                    return false;
                }
            }
            self.control_document_generation_applied = projection.document_generation;
            self.control_actor_mode_signature = Some(self.control_actor_mode_signature());
            return true;
        }
        if projection.mode != ModelMode::Single {
            log_warn!("unsupported transition actor render projection");
            return false;
        }
        if projection.document_generation != self.control_document_generation_applied {
            let Some(document) = projection.document.as_ref() else {
                let actor_source = projection
                    .workspace
                    .as_ref()
                    .and_then(|workspace| workspace.get("shared_resources"))
                    .and_then(|resources| resources.get("dataset_source"))
                    .and_then(serde_json::Value::as_str);
                let renderer_source = match &self.mode {
                    Mode::Single(app) => Some(app.control_actor_source_key()),
                    _ => None,
                };
                if renderer_source.as_deref() == actor_source {
                    self.control_document_generation_applied = projection.document_generation;
                    self.control_actor_mode_signature =
                        actor_source.map(|source| format!("single:{source}"));
                    if let Mode::Single(app) = &mut self.mode {
                        app.project_space_mut()
                            .apply_control_actor_project_projection(&projection.project);
                    }
                    return self.apply_control_projection_workspace(
                        projection.workspace.as_ref(),
                        projection.object_resource.as_ref(),
                        projection.label_resource.as_ref(),
                    );
                }
                log_warn!(
                    "actor projection generation {} has no matching render document",
                    projection.document_generation
                );
                return false;
            };
            if document.generation != projection.document_generation {
                log_warn!("actor projection and render document generations disagree");
                return false;
            }
            let mut project_space = match &mut self.mode {
                Mode::Project { project_space } => std::mem::take(project_space),
                Mode::Single(app) => app.take_project_space(),
                Mode::Mosaic { mosaic, .. } => mosaic.take_project_space(),
                Mode::Transition => ProjectSpace::default(),
            };
            if let Some(path) = document.path() {
                project_space.handle_dropped_paths([path.to_path_buf()]);
            }
            let app = match &document.opened.resource {
                odon::data::document::DocumentResource::OmeZarr(_) => {
                    Ok(OmeZarrViewerApp::new_runtime(
                        ctx,
                        self.gpu_available,
                        document.dataset().clone(),
                        Arc::clone(document.store()),
                        self.app_settings.auto_contrast,
                    ))
                }
                odon::data::document::DocumentResource::Alternate(resource)
                    if document.opened.descriptor.kind
                        == odon::data::document::DocumentKind::Tiff =>
                {
                    OmeZarrViewerApp::new_tiff_runtime_from_resource(
                        ctx,
                        self.gpu_available,
                        resource,
                        self.app_settings.auto_contrast,
                    )
                }
                odon::data::document::DocumentResource::Alternate(resource)
                    if document.opened.descriptor.kind
                        == odon::data::document::DocumentKind::SpatialData =>
                {
                    let payload = resource
                        .payload::<crate::app_support::datasets::PreparedSpatialDataDocument>()
                        .ok_or_else(|| {
                            anyhow::anyhow!("SpatialData document has an incompatible resource")
                        });
                    payload.map(|payload| {
                        let mut app = OmeZarrViewerApp::new_runtime(
                            ctx,
                            self.gpu_available,
                            resource.dataset.clone(),
                            Arc::clone(&resource.store),
                            self.app_settings.auto_contrast,
                        );
                        app.attach_prepared_spatialdata_layers(
                            payload.root.clone(),
                            payload.image_transform,
                            payload.extra_images.clone(),
                            payload.labels.clone(),
                            payload.tables.clone(),
                            payload.shapes.clone(),
                            payload.points.clone(),
                        );
                        app
                    })
                }
                odon::data::document::DocumentResource::Alternate(resource)
                    if document.opened.descriptor.kind
                        == odon::data::document::DocumentKind::Xenium =>
                {
                    let payload = resource
                        .payload::<crate::app_support::datasets::PreparedXeniumDocument>()
                        .ok_or_else(|| {
                            anyhow::anyhow!("Xenium document has an incompatible resource")
                        });
                    payload.and_then(|payload| {
                        let mut app = match &payload.imagery {
                            crate::app_support::datasets::PreparedXeniumImagery::OmeZarr => {
                                OmeZarrViewerApp::new_runtime(
                                    ctx,
                                    self.gpu_available,
                                    resource.dataset.clone(),
                                    Arc::clone(&resource.store),
                                    self.app_settings.auto_contrast,
                                )
                            }
                            crate::app_support::datasets::PreparedXeniumImagery::Tiff(pyramid) => {
                                OmeZarrViewerApp::new_tiff_runtime_from_prepared_resource(
                                    ctx,
                                    self.gpu_available,
                                    resource,
                                    Arc::clone(pyramid),
                                    self.app_settings.auto_contrast,
                                )?
                            }
                        };
                        app.attach_prepared_xenium_layers(
                            payload.root.clone(),
                            payload.cells.clone(),
                            payload.transcripts.clone(),
                            payload.pixel_size_um,
                        );
                        Ok(app)
                    })
                }
                odon::data::document::DocumentResource::Alternate(_) => Err(anyhow::anyhow!(
                    "no renderer adapter for {:?}",
                    document.opened.descriptor.kind
                )),
            };
            let mut app = match app {
                Ok(app) => app,
                Err(error) => {
                    log_warn!("could not realize actor document renderer: {error}");
                    return false;
                }
            };
            app.set_remote_runtime(document.opened.resource.runtime_guard());
            self.configure_single_app(&mut app);
            app.set_project_space_from_actor(project_space);
            self.mode = Mode::Single(app);
            self.control_document_generation_applied = projection.document_generation;
            self.control_actor_mode_signature = Some(format!(
                "single:{}",
                document.opened.descriptor.source.source_key()
            ));
        }
        if let Mode::Single(app) = &mut self.mode {
            app.project_space_mut()
                .apply_control_actor_project_projection(&projection.project);
            app.apply_control_actor_screenshot_preferences(&projection.screenshot_preferences);
            if let Err(error) =
                app.apply_control_actor_tile_loading_policy(&projection.tile_loading_policy)
            {
                log_warn!("could not realize actor tile-loading policy: {error}");
            }
            app.apply_control_actor_pinned_levels(&projection.pinned_levels);
            if let Err(error) = app.apply_control_actor_threshold_preview(
                ctx,
                projection.threshold_preview_generation,
                projection.threshold_preview_pending,
                projection.threshold_preview.as_ref(),
                &projection.threshold_preview_state,
            ) {
                log_warn!("could not realize actor threshold preview: {error}");
            }
            if let Err(error) = app.apply_control_actor_analysis_state(
                projection.analysis_generation,
                &projection.analysis_state,
            ) {
                log_warn!("could not realize actor analysis state: {error}");
            }
            if let Err(error) = app.apply_control_actor_measurement_state(
                projection.measurement_generation,
                &projection.measurement_state,
            ) {
                log_warn!("could not realize actor measurement state: {error}");
            }
            app.apply_control_actor_object_export_state(
                projection.object_export_generation,
                &projection.object_export_state,
            );
            if let Err(error) = app.install_control_actor_secondary_object_resources(
                &projection.secondary_object_layers,
            ) {
                log_warn!("could not realize actor secondary object resources: {error}");
                return false;
            }
        }
        if let Mode::Mosaic { mosaic, .. } = &mut self.mode {
            mosaic.apply_control_actor_screenshot_preferences(&projection.screenshot_preferences);
        }
        self.apply_control_projection_workspace(
            projection.workspace.as_ref(),
            projection.object_resource.as_ref(),
            projection.label_resource.as_ref(),
        )
    }

    fn apply_project_object_preload_projection(
        &mut self,
        projection: &ProjectObjectPreloadProjection,
    ) {
        let mode = match projection.settings.mode {
            ProjectObjectPreloadMode::FullGeometry => ObjectPreloadMode::FullGeometry,
            ProjectObjectPreloadMode::CentroidPoints => ObjectPreloadMode::CentroidPoints,
        };
        let settings = ObjectPreloadSettings {
            mode,
            lazy_properties: projection.settings.lazy_properties,
        };
        self.object_preload_settings = settings;
        self.object_preload_available_count =
            projection.state["available_count"].as_u64().unwrap_or(0) as usize;
        self.object_preload_on_disk_bytes = projection.state["on_disk_bytes"].as_u64().unwrap_or(0);
        self.object_preload_total = projection.state["total"].as_u64().unwrap_or(0) as usize;
        self.object_preload_done = projection.state["done"].as_u64().unwrap_or(0) as usize;
        self.object_preload_failed = projection.state["failed"].as_u64().unwrap_or(0) as usize;
        self.object_preload_loading = projection.state["loading"].as_bool().unwrap_or(false);
        self.object_preload_cache.clear();
        for (path, resource) in projection.resources.iter() {
            let Some(preloaded) = resource.renderer_payload::<PreloadedObjectLayer>() else {
                continue;
            };
            self.object_preload_cache
                .insert((path.clone(), settings), Arc::new(preloaded.clone()));
        }

        match &mut self.mode {
            Mode::Mosaic { mosaic, .. } => {
                let cached = self
                    .object_preload_cache
                    .iter()
                    .filter_map(|((path, cached_settings), preloaded)| {
                        (*cached_settings == settings)
                            .then_some((path.clone(), Arc::clone(preloaded)))
                    })
                    .collect::<Vec<_>>();
                mosaic.install_preloaded_project_segmentations(&cached);
            }
            Mode::Single(app) => {
                let matching = app
                    .project_space()
                    .rois()
                    .iter()
                    .find(|roi| app.is_viewing_project_roi(roi))
                    .and_then(|roi| {
                        project_roi_segmentation_path(app.project_space(), roi)
                            .and_then(|path| self.object_preload_cache.get(&(path, settings)))
                    })
                    .cloned();
                if let Some(preloaded) = matching {
                    app.install_preloaded_project_segmentation(&preloaded);
                }
            }
            Mode::Project { .. } | Mode::Transition => {}
        }
    }

    fn apply_control_projection_workspace(
        &mut self,
        workspace: Option<&serde_json::Value>,
        object_resource: Option<&Arc<odon::model::ControlObjectResource>>,
        label_resource: Option<&Arc<odon::model::ControlLabelResource>>,
    ) -> bool {
        let Some(workspace) = workspace else {
            log_warn!("single-viewer actor projection has no workspace");
            return false;
        };
        let Mode::Single(app) = &mut self.mode else {
            log_warn!("actor projection could not establish a single-image viewer");
            return false;
        };
        if let Some(resource) = object_resource {
            let generation = workspace
                .get("object_resource")
                .and_then(|descriptor| descriptor.get("generation"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            app.install_control_actor_object_resource(generation, resource);
        }
        if workspace
            .get("labels")
            .and_then(|labels| labels.get("actor_owned"))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            let generation = workspace["labels"]["generation"].as_u64().unwrap_or(0);
            let result = if let Some(resource) = label_resource {
                app.install_control_actor_label_resource(generation, resource)
                    .map(|_| ())
            } else {
                app.unload_control_actor_label_resource(generation);
                Ok(())
            };
            if let Err(error) = result {
                log_warn!("actor label resource could not be installed: {error}");
                return false;
            }
        }
        if let Err(error) = app.apply_control_actor_workspace_projection(workspace) {
            log_warn!("actor render projection could not be applied: {error}");
            return false;
        }
        true
    }

    fn report_control_renderer_observation(&mut self) {
        if self.control_last_observed_at.elapsed() < Duration::from_millis(33) {
            return;
        }
        self.control_last_observed_at = Instant::now();
        let Mode::Single(app) = &mut self.mode else {
            return;
        };
        let _ = self.control_runtime.observe_renderer_workspace(
            app.control_viewport_workspace_snapshot(),
            self.control_projection_revision_applied,
        );
    }

    fn current_project_space(&self) -> Option<&ProjectSpace> {
        match &self.mode {
            Mode::Project { project_space } => Some(project_space),
            Mode::Single(app) => Some(app.project_space()),
            Mode::Mosaic { mosaic, .. } => Some(mosaic.project_space()),
            Mode::Transition => None,
        }
    }

    fn current_project_space_mut(&mut self) -> Option<&mut ProjectSpace> {
        match &mut self.mode {
            Mode::Project { project_space } => Some(project_space),
            Mode::Single(app) => Some(app.project_space_mut()),
            Mode::Mosaic { mosaic, .. } => Some(mosaic.project_space_mut()),
            Mode::Transition => None,
        }
    }

    fn sync_control_manifest_to_project(&mut self) {
        let (resources, layers) = self.control_runtime.project_control_manifest();
        let revision = self.control_runtime.revision();
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
            .control_runtime
            .replace_project_control_manifest(&resources, &layers);
        if let Err(error) = result
            && let Some(project_space) = self.current_project_space_mut()
        {
            project_space.set_status(format!("Project external layer restore failed: {error}"));
        }
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
                let ViewportScreenshotDestination::Presentation { capture_id, tx } =
                    &request.destination;
                let _ = tx.send(PresentationCaptureCompletion {
                    capture_id: *capture_id,
                    result: Err("screenshot crop was empty".to_string()),
                });
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
            let ViewportScreenshotDestination::Presentation { capture_id, tx } =
                &request.destination;
            let _ = tx.send(PresentationCaptureCompletion {
                capture_id: *capture_id,
                result: Ok(PresentationPixels {
                    width: capture_width,
                    height: capture_height,
                    rgba,
                    bottom_up: false,
                }),
            });
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
            let submitted = self.control_runtime.submit_native_command(
                ctx,
                "app.settings.set",
                serde_json::json!({
                    "auto_contrast":self.app_settings.auto_contrast,
                    "fast_object_rendering":self.app_settings.fast_object_rendering,
                }),
            );
            if !submitted {
                self.persist_app_settings();
            }
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

    pub fn new_project(
        cc: &eframe::CreationContext<'_>,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let (app_settings, mut settings_status) = Self::load_app_settings();
        let control_runtime = Self::spawn_control_runtime(&cc.egui_ctx, &mut settings_status)?;
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
            deep_link_rx: None,
            object_preload_cache: HashMap::new(),
            object_preload_settings: ObjectPreloadSettings::default(),
            object_preload_available_count: 0,
            object_preload_on_disk_bytes: 0,
            object_preload_total: 0,
            object_preload_done: 0,
            object_preload_failed: 0,
            object_preload_loading: false,
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
            remote_control_pending: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            app_settings,
            settings_open: false,
            settings_status,
            active_help_topic: None,
            control_runtime,
            control_external_revision: 0,
            control_project_revision: 0,
            control_last_observed_at: Instant::now() - Duration::from_millis(34),
            control_projection_applied_this_frame: None,
            control_projection_revision_applied: 0,
            control_document_generation_applied: 0,
            control_actor_mode_signature: None,
            control_projection_gap: false,
            deferred_control_projection: None,
            pending_native_control_intents: VecDeque::new(),
            #[cfg(target_os = "macos")]
            native_menu: None,
        };
        root.bootstrap_control_actor();
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
        let control_runtime = Self::spawn_control_runtime(&cc.egui_ctx, &mut settings_status)?;
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
            deep_link_rx: None,
            object_preload_cache: HashMap::new(),
            object_preload_settings: ObjectPreloadSettings::default(),
            object_preload_available_count: 0,
            object_preload_on_disk_bytes: 0,
            object_preload_total: 0,
            object_preload_done: 0,
            object_preload_failed: 0,
            object_preload_loading: false,
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
            remote_control_pending: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            app_settings,
            settings_open: false,
            settings_status,
            active_help_topic: None,
            control_runtime,
            control_external_revision: 0,
            control_project_revision: 0,
            control_last_observed_at: Instant::now() - Duration::from_millis(34),
            control_projection_applied_this_frame: None,
            control_projection_revision_applied: 0,
            control_document_generation_applied: 0,
            control_actor_mode_signature: None,
            control_projection_gap: false,
            deferred_control_projection: None,
            pending_native_control_intents: VecDeque::new(),
            #[cfg(target_os = "macos")]
            native_menu: None,
        };
        root.bootstrap_control_actor();
        root.load_control_manifest_from_project();
        Ok(root)
    }

    pub fn new_mosaic(
        cc: &eframe::CreationContext<'_>,
        mut mosaic: MosaicViewerApp,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let (app_settings, mut settings_status) = Self::load_app_settings();
        let control_runtime = Self::spawn_control_runtime(&cc.egui_ctx, &mut settings_status)?;
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
            deep_link_rx: None,
            object_preload_cache: HashMap::new(),
            object_preload_settings: ObjectPreloadSettings::default(),
            object_preload_available_count: 0,
            object_preload_on_disk_bytes: 0,
            object_preload_total: 0,
            object_preload_done: 0,
            object_preload_failed: 0,
            object_preload_loading: false,
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
            remote_control_pending: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            app_settings,
            settings_open: false,
            settings_status,
            active_help_topic: None,
            control_runtime,
            control_external_revision: 0,
            control_project_revision: 0,
            control_last_observed_at: Instant::now() - Duration::from_millis(34),
            control_projection_applied_this_frame: None,
            control_projection_revision_applied: 0,
            control_document_generation_applied: 0,
            control_actor_mode_signature: None,
            control_projection_gap: false,
            deferred_control_projection: None,
            pending_native_control_intents: VecDeque::new(),
            #[cfg(target_os = "macos")]
            native_menu: None,
        };
        root.bootstrap_control_actor();
        root.load_control_manifest_from_project();
        Ok(root)
    }

    pub fn queue_open_root(&mut self, root: PathBuf) {
        let root = normalize_local_dataset_path(&root).unwrap_or(root);
        let method =
            match classify_local_dataset_path(&root) {
                Some(LocalDatasetKind::OmeZarr) => "datasets.open_ome_zarr",
                Some(LocalDatasetKind::Tiff) => "datasets.open_tiff",
                Some(LocalDatasetKind::Xenium) => "datasets.open_xenium",
                None => match discover_spatialdata(&root) {
                    Ok(discovery) if !discovery.images.is_empty() => {
                        let mut dialog = SpatialOpenDialog {
                            discovery,
                            selected_image: 0,
                            selected_labels: None,
                            selected_shapes: Vec::new(),
                            selected_points: None,
                            points_max: 200_000,
                            status: String::new(),
                        };
                        if let Some(index) = dialog.discovery.labels.iter().position(|layer| {
                            layer.name == "cells" || layer.name == "point8_labels"
                        }) {
                            dialog.selected_labels = Some(index);
                        }
                        if let Some(index) = dialog
                            .discovery
                            .shapes
                            .iter()
                            .position(|layer| layer.name == "cell_boundaries")
                        {
                            dialog.selected_shapes.push(index);
                        }
                        self.spatial_open = Some(dialog);
                        return;
                    }
                    _ => "datasets.open_ome_zarr",
                },
            };
        self.pending_native_control_intents
            .push_back(NativeControlIntent {
                method,
                params: serde_json::json!({"path":root}),
            });
    }

    pub fn queue_deep_link(&mut self, request: DeepLinkRequest) {
        self.pending_native_control_intents
            .push_back(NativeControlIntent {
                method: "deep_links.apply",
                params: serde_json::json!({"request":request}),
            });
    }

    pub fn set_deep_link_receiver(&mut self, rx: Receiver<DeepLinkRequest>) {
        self.deep_link_rx = Some(rx);
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

        let (root, img, labels, shapes, points, points_max) = {
            let Some(dlg) = self.spatial_open.as_ref() else {
                return;
            };
            let root = dlg.discovery.root.clone();
            let img = dlg.discovery.images.get(dlg.selected_image).cloned();
            let labels = dlg
                .selected_labels
                .and_then(|i| dlg.discovery.labels.get(i))
                .cloned();
            let shapes = dlg
                .selected_shapes
                .iter()
                .filter_map(|&i| dlg.discovery.shapes.get(i).cloned())
                .collect::<Vec<_>>();
            let points = dlg
                .selected_points
                .and_then(|i| dlg.discovery.points.get(i))
                .cloned();
            (root, img, labels, shapes, points, dlg.points_max)
        };

        let Some(img) = img else {
            if let Some(dlg) = self.spatial_open.as_mut() {
                dlg.status = "No image selected.".to_string();
            }
            return;
        };

        let params = serde_json::json!({
            "path":root,
            "image":img.name,
            "extra_images":[],
            "labels":labels.as_ref().map(|element| element.name.clone()),
            "shapes":shapes.iter().map(|element| element.name.clone()).collect::<Vec<_>>(),
            "points":points.as_ref().map(|element| element.name.clone()),
            "points_max":points_max,
        });
        if self
            .control_runtime
            .submit_native_command(ctx, "datasets.open_spatialdata", params)
        {
            self.spatial_open = None;
        } else if let Some(dlg) = self.spatial_open.as_mut() {
            dlg.status = "Odon's control actor is busy; retry Open.".to_string();
        }
    }
}

impl eframe::App for RootApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.handle_viewport_screenshot_events(ctx);
        self.process_control_presentations(ctx);
        self.process_control_presentation_captures(ctx);
        self.process_control_platform_effects(ctx);
        if let Some(rx) = self.deep_link_rx.as_ref() {
            ctx.request_repaint_after(Duration::from_millis(100));
            let mut received_deep_link = false;
            while let Ok(request) = rx.try_recv() {
                log_warn!("deep_link: received {:?}", request);
                self.pending_native_control_intents
                    .push_back(NativeControlIntent {
                        method: "deep_links.apply",
                        params: serde_json::json!({"request":request}),
                    });
                received_deep_link = true;
            }
            if received_deep_link {
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
                    egui::UserAttentionType::Informational,
                ));
            }
        }
        let open_mosaic: Option<Vec<PathBuf>> = None;
        let mut open_remote_s3_mosaic: Option<(Vec<crate::app::S3DatasetSelection>, ProjectSpace)> =
            None;
        let mut back_to_single = false;
        let mut native_menu_control_intents = Vec::new();

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
                                native_menu_control_intents.push(NativeControlIntent {
                                    method: "datasets.open_ome_zarr",
                                    params: serde_json::json!({"path":root}),
                                });
                            }
                        }
                        NativeMenuAction::OpenTiff => {
                            if let Some(root) = FileDialog::new()
                                .add_filter("TIFF / OME-TIFF", &["tif", "tiff"])
                                .set_title("Open TIFF / OME-TIFF")
                                .pick_file()
                            {
                                native_menu_control_intents.push(NativeControlIntent {
                                    method: "datasets.open_tiff",
                                    params: serde_json::json!({"path":root}),
                                });
                            }
                        }
                        NativeMenuAction::OpenProject => {
                            if let Some(path) = FileDialog::new()
                                .add_filter("Project JSON", &["json"])
                                .set_title("Load Project")
                                .pick_file()
                            {
                                native_menu_control_intents.push(NativeControlIntent {
                                    method: "project.open",
                                    params: serde_json::json!({"path": path}),
                                });
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
                            let save_target = save_target.or_else(|| {
                                FileDialog::new()
                                    .add_filter("Project JSON", &["json"])
                                    .set_file_name("odon.project.json")
                                    .set_title("Save Project")
                                    .save_file()
                            });
                            if let Some(path) = save_target {
                                native_menu_control_intents.push(NativeControlIntent {
                                    method: "project.save_as",
                                    params: serde_json::json!({"path": path}),
                                });
                            }
                        }
                        NativeMenuAction::SaveNewProject => {
                            if let Some(path) = FileDialog::new()
                                .add_filter("Project JSON", &["json"])
                                .set_file_name("odon.project.json")
                                .set_title("Save Project As")
                                .save_file()
                            {
                                native_menu_control_intents.push(NativeControlIntent {
                                    method: "project.save_as",
                                    params: serde_json::json!({"path": path}),
                                });
                            }
                        }
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
                        action @ (NativeMenuAction::CloseWindow | NativeMenuAction::Quit) => {
                            let quit = matches!(action, NativeMenuAction::Quit);
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
                                native_menu_control_intents.push(NativeControlIntent {
                                    method: if quit {
                                        "app.lifecycle.request_quit"
                                    } else {
                                        "app.lifecycle.request_close"
                                    },
                                    params: serde_json::json!({"save":"discard"}),
                                });
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
        let object_preload_available_count = self.object_preload_available_count;
        let object_preload_on_disk_bytes = self.object_preload_on_disk_bytes;
        let object_preload_total = self.object_preload_total;
        let object_preload_done = self.object_preload_done;
        let object_preload_failed = self.object_preload_failed;
        let object_preload_loading = self.object_preload_loading;
        let object_preload_settings = self.object_preload_settings;
        let external_layers = Some(self.control_runtime.external_layers())
            .filter(|(revision, _, _)| *revision != self.control_external_revision);
        let observed = self.actor_renderer_observation();
        self.control_runtime.render_extension_ui(ctx, &observed);
        self.sync_control_manifest_to_project();

        if let Some(project_space) = self.current_project_space_mut() {
            project_space.set_control_actor_owned(true);
        }
        if let Mode::Mosaic { mosaic, .. } = &mut self.mode {
            mosaic.set_control_actor_owned(true);
        }

        let mut native_control_intents = Vec::new();
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
                    native_control_intents.push(NativeControlIntent {
                        method: "app.lifecycle.request_close",
                        params: serde_json::json!({"save":"discard"}),
                    });
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
                        object_preload_available_count,
                        object_preload_on_disk_bytes,
                        object_preload_cached,
                        object_preload_total,
                        object_preload_done,
                        object_preload_failed,
                        object_preload_loading,
                        object_preload_settings,
                    ));
                    let action = project_space.ui(ui, None);
                    if let Some(action) = action {
                        if !project_space.submit_action_control_intent(&action) {
                            match action {
                                ProjectSpaceAction::CaptureCurrentView => {}
                                ProjectSpaceAction::OpenRemoteDialog => {
                                    self.remote_dialog_open = true;
                                    self.remote_status.clear();
                                }
                                ProjectSpaceAction::ShowHelp(topic) => {
                                    self.active_help_topic = Some(topic);
                                }
                                _ => unreachable!(
                                    "actor-owned project action was not accepted by its command outbox"
                                ),
                            }
                        }
                    }
                });
                if let Some(action) = project_space.ui_floating_windows(ctx, false) {
                    if !project_space.submit_action_control_intent(&action) {
                        match action {
                            ProjectSpaceAction::CaptureCurrentView => {}
                            ProjectSpaceAction::OpenRemoteDialog => {
                                self.remote_dialog_open = true;
                                self.remote_status.clear();
                            }
                            ProjectSpaceAction::ShowHelp(topic) => {
                                self.active_help_topic = Some(topic);
                            }
                            _ => unreachable!(
                                "actor-owned project action was not accepted by its command outbox"
                            ),
                        }
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
                    object_preload_available_count,
                    object_preload_on_disk_bytes,
                    object_preload_cached,
                    object_preload_total,
                    object_preload_done,
                    object_preload_failed,
                    object_preload_loading,
                    object_preload_settings,
                ));
                app.update(ctx, frame);
                native_control_intents.extend(app.take_native_control_intents());
                self.label_prompt_preference = app.label_prompt_preference();
                if let Some(req) = app.take_request() {
                    match req {
                        ViewerRequest::OpenRemoteS3Mosaic(datasets) => {
                            let ps = app.take_project_space();
                            open_remote_s3_mosaic = Some((datasets, ps));
                        }
                    }
                }
            }
            Mode::Mosaic { mosaic, .. } => {
                mosaic
                    .project_space_mut()
                    .set_recent_projects(&self.app_settings.recent_projects);
                mosaic.set_project_object_cache_ui_state(project_object_cache_ui_state(
                    object_preload_available_count,
                    object_preload_on_disk_bytes,
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
                native_control_intents.extend(mosaic.take_native_control_intents());
                if let Some(req) = mosaic.take_request() {
                    match req {
                        MosaicRequest::CloseWindow => {
                            native_control_intents.push(NativeControlIntent {
                                method: "app.lifecycle.request_close",
                                params: serde_json::json!({"save":"discard"}),
                            });
                        }
                        MosaicRequest::BackToSingle => {
                            back_to_single = true;
                        }
                        MosaicRequest::OpenRemoteDialog => {
                            self.remote_dialog_open = true;
                            self.remote_status.clear();
                        }
                    }
                }
            }
            Mode::Transition => {}
        }

        if let Some(project_space) = self.current_project_space_mut() {
            native_control_intents.extend(project_space.take_control_intents().into_iter().map(
                |intent| NativeControlIntent {
                    method: intent.method,
                    params: intent.params,
                },
            ));
        }

        if !native_control_intents.is_empty() || !native_menu_control_intents.is_empty() {
            let mut lifecycle_intents = Vec::new();
            native_control_intents.retain(|intent| {
                if intent.method.starts_with("app.lifecycle.request_") {
                    lifecycle_intents.push(intent.clone());
                    false
                } else {
                    true
                }
            });
            native_menu_control_intents.retain(|intent| {
                if intent.method.starts_with("app.lifecycle.request_") {
                    lifecycle_intents.push(intent.clone());
                    false
                } else {
                    true
                }
            });
            // The actor publishes the canonical revisions and events for these optimistic native
            // commits.
            self.pending_native_control_intents
                .extend(native_control_intents);
            // Persistence commands must follow any semantic edits collected in this frame so the
            // actor's single mailbox snapshots the updated project.
            self.pending_native_control_intents
                .extend(native_menu_control_intents);
            // Close/quit validation observes every semantic mutation and persistence command
            // collected earlier in this frame.
            self.pending_native_control_intents
                .extend(lifecycle_intents);
        }
        while let Some(intent) = self.pending_native_control_intents.front() {
            if self
                .control_runtime
                .submit_native_command(ctx, intent.method, intent.params.clone())
            {
                self.pending_native_control_intents.pop_front();
            } else {
                ctx.request_repaint_after(Duration::from_millis(5));
                break;
            }
        }

        if matches!(self.mode, Mode::Project { .. }) {
            self.ui_spatial_open_dialog(ctx);
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
        self.sync_control_actor_mode_from_native();
        self.report_control_viewport_geometry();
        self.report_control_presentation();
        self.report_control_renderer_observation();
        crate::ui::help::show_help_window(ctx, &mut self.active_help_topic);
    }
}
