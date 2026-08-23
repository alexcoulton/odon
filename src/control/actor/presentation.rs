use super::*;
use std::collections::BTreeMap;
use std::path::Path;

use super::completion::reject_cancelled_request;

const PRESENTATION_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentationCaptureScope {
    Viewer { viewport_id: Option<String> },
    Workspace,
    Window,
    Project,
}

impl PresentationCaptureScope {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Viewer { .. } => "viewer",
            Self::Workspace => "workspace",
            Self::Window => "window",
            Self::Project => "project",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PresentationCaptureRequest {
    pub capture_id: u64,
    pub desired_projection_revision: u64,
    pub mode: ModelMode,
    pub scope: PresentationCaptureScope,
    pub screenshot_preferences: ScreenshotPreferences,
}

#[derive(Debug)]
pub struct PresentationPixels {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    pub bottom_up: bool,
}

#[derive(Debug)]
pub struct PresentationCaptureCompletion {
    pub capture_id: u64,
    pub result: Result<PresentationPixels, String>,
}

pub(super) struct ScreenshotWriteSpec {
    pub(super) capture_id: u64,
    pub(super) desired_projection_revision: u64,
    pub(super) mode: ModelMode,
    pub(super) scope: PresentationCaptureScope,
    pub(super) screenshot_preferences: ScreenshotPreferences,
    pub(super) path: PathBuf,
    pub(super) overwrite: bool,
    pub(super) project_transition: Option<Value>,
}

struct PendingCapture {
    request: OdonControlRequest,
    spec: ScreenshotWriteSpec,
    queued_at: Instant,
    sent_to_renderer: bool,
}

#[derive(Default)]
pub(super) struct PresentationCaptureManager {
    next_id: u64,
    pending: BTreeMap<u64, PendingCapture>,
}

impl PresentationCaptureManager {
    pub(super) fn barrier_active(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(super) fn begin(
        &mut self,
        model: &mut AppModel,
        request: OdonControlRequest,
        render_document: &Option<Arc<RenderDocument>>,
        projection_tx: &Sender<RenderProjection>,
        projection_coalesce_rx: &Receiver<RenderProjection>,
        capture_tx: &Sender<PresentationCaptureRequest>,
        wake_ui: &UiWake,
        diagnostics: &ActorDiagnostics,
    ) {
        // A renderer currently owns one GPU/viewport readback slot. Keeping this transaction
        // singular also prevents a later projection from overtaking the revision being captured.
        if !self.pending.is_empty() || capture_tx.is_full() {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(
                    ControlErrorKind::NotReady,
                    "the screenshot presentation queue is full; retry later",
                ),
            );
            return;
        }
        let method = request.command.method();
        let params = request.command.params();
        let original_mode = model.mode();
        let path = match capture_path(model, method, params) {
            Ok(path) => path,
            Err(error) => {
                reject_actor_request(request, diagnostics, error);
                return;
            }
        };
        let overwrite = params
            .get("overwrite")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if path.exists() && !overwrite {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(
                    ControlErrorKind::Conflict,
                    "destination exists; pass overwrite=true to replace it",
                )
                .with_data(json!({"path":path.to_string_lossy()})),
            );
            return;
        }
        let (scope, project_transition) = match method {
            "viewer.screenshot.capture" => {
                match model.capture_viewport_id(params.get("viewport_id").and_then(Value::as_str)) {
                    Ok(viewport_id) => (PresentationCaptureScope::Viewer { viewport_id }, None),
                    Err(error) => {
                        reject_actor_request(request, diagnostics, error);
                        return;
                    }
                }
            }
            "viewer.workspace.screenshot.capture" => {
                if original_mode != ModelMode::Single {
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(
                            ControlErrorKind::WrongMode,
                            "workspace screenshots require single-image mode",
                        ),
                    );
                    return;
                }
                (PresentationCaptureScope::Workspace, None)
            }
            "app.screenshot.capture" => (PresentationCaptureScope::Window, None),
            "project.screenshot.capture" => match model.prepare_project_capture() {
                Ok(transition) => (PresentationCaptureScope::Project, Some(transition)),
                Err(error) => {
                    reject_actor_request(request, diagnostics, error);
                    return;
                }
            },
            _ => {
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::MethodNotFound,
                        format!("unsupported screenshot method '{method}'"),
                    ),
                );
                return;
            }
        };
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let capture_id = self.next_id;
        model.begin_presentation_capture(capture_id, "Waiting for screenshot presentation");
        publish_projection(
            model,
            render_document.clone(),
            projection_tx,
            projection_coalesce_rx,
            wake_ui,
            diagnostics,
        );
        let desired_projection_revision = model.projection_revision();
        if let Some(task_id) = request.task_id.as_deref() {
            let _ = request.task_registry.progress(
                task_id,
                None,
                format!("waiting_for_presentation:projection:{desired_projection_revision}"),
            );
        }
        self.pending.insert(
            capture_id,
            PendingCapture {
                request,
                spec: ScreenshotWriteSpec {
                    capture_id,
                    desired_projection_revision,
                    mode: if matches!(scope, PresentationCaptureScope::Project) {
                        ModelMode::Project
                    } else {
                        original_mode
                    },
                    scope,
                    screenshot_preferences: model.screenshot_preferences().clone(),
                    path,
                    overwrite,
                    project_transition,
                },
                queued_at: Instant::now(),
                sent_to_renderer: false,
            },
        );
        self.release_presentable(model.presented_projection_revision(), capture_tx, wake_ui);
    }

    pub(super) fn release_presentable(
        &mut self,
        presented_revision: u64,
        capture_tx: &Sender<PresentationCaptureRequest>,
        wake_ui: &UiWake,
    ) {
        for pending in self.pending.values_mut() {
            if pending.sent_to_renderer
                || pending.spec.desired_projection_revision > presented_revision
            {
                continue;
            }
            let request = PresentationCaptureRequest {
                capture_id: pending.spec.capture_id,
                desired_projection_revision: pending.spec.desired_projection_revision,
                mode: pending.spec.mode,
                scope: pending.spec.scope.clone(),
                screenshot_preferences: pending.spec.screenshot_preferences.clone(),
            };
            match capture_tx.try_send(request) {
                Ok(()) => {
                    pending.sent_to_renderer = true;
                    wake_ui();
                }
                Err(crossbeam_channel::TrySendError::Full(_)) => break,
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => break,
            }
        }
    }

    pub(super) fn receive_pixels(
        &mut self,
        model: &mut AppModel,
        completion: PresentationCaptureCompletion,
        load_job_tx: &Sender<LoadJob>,
        diagnostics: &ActorDiagnostics,
    ) {
        let Some(pending) = self.pending.remove(&completion.capture_id) else {
            diagnostics
                .stale_worker_completions
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        if request_cancelled(&pending.request) {
            model.cancel_presentation_capture(
                pending.spec.capture_id,
                "Screenshot capture was cancelled",
            );
            reject_cancelled_request(pending.request, diagnostics, "screenshot capture");
            return;
        }
        let pixels = match completion.result {
            Ok(pixels) => pixels,
            Err(message) => {
                model.fail_presentation_capture(pending.spec.capture_id, &message);
                reject_actor_request(
                    pending.request,
                    diagnostics,
                    ControlError::new(ControlErrorKind::Application, message),
                );
                return;
            }
        };
        if let Some(task_id) = pending.request.task_id.as_deref() {
            let _ = pending
                .request
                .task_registry
                .progress(task_id, None, "writing screenshot PNG");
        }
        match load_job_tx.try_send(LoadJob::ScreenshotWrite {
            request: pending.request,
            spec: pending.spec,
            pixels,
        }) {
            Ok(()) => {
                diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                let LoadJob::ScreenshotWrite { request, spec, .. } = error.into_inner() else {
                    unreachable!("screenshot submission returns its own job")
                };
                model.fail_presentation_capture(
                    spec.capture_id,
                    "Screenshot output worker queue is unavailable",
                );
                reject_worker_submission(request, diagnostics);
            }
        }
    }

    pub(super) fn sweep(&mut self, model: &mut AppModel, diagnostics: &ActorDiagnostics) {
        let expired = self
            .pending
            .iter()
            .filter_map(|(id, pending)| {
                (request_cancelled(&pending.request)
                    || pending.queued_at.elapsed() >= PRESENTATION_TIMEOUT)
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        for id in expired {
            let Some(pending) = self.pending.remove(&id) else {
                continue;
            };
            if request_cancelled(&pending.request) {
                model.cancel_presentation_capture(id, "Screenshot capture was cancelled");
                reject_cancelled_request(pending.request, diagnostics, "screenshot capture");
            } else {
                let message = format!(
                    "timed out after {} seconds waiting for projection {} to be presented",
                    PRESENTATION_TIMEOUT.as_secs(),
                    pending.spec.desired_projection_revision
                );
                model.fail_presentation_capture(id, &message);
                reject_actor_request(
                    pending.request,
                    diagnostics,
                    ControlError::new(ControlErrorKind::Timeout, message).with_data(json!({
                        "capture_id":id,
                        "waiting_for_projection":pending.spec.desired_projection_revision,
                        "phase":"waiting_for_presentation",
                    })),
                );
            }
        }
    }
}

fn request_cancelled(request: &OdonControlRequest) -> bool {
    request
        .task_id
        .as_deref()
        .and_then(|task_id| request.task_registry.get(task_id).ok())
        .is_some_and(|task| task.state == TaskState::Cancelled)
}

fn capture_path(model: &AppModel, method: &str, params: &Value) -> Result<PathBuf, ControlError> {
    if let Some(path) = params
        .get("path")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        return Ok(expand_path(path));
    }
    if method != "viewer.screenshot.capture" {
        return Err(ControlError::invalid_params(method, "path is required"));
    }
    let directory = model.screenshot_preferences().output_dir().ok_or_else(|| {
        ControlError::new(
            ControlErrorKind::NotReady,
            "No screenshot folder is configured; provide path or set output_dir",
        )
    })?;
    if !directory.is_dir() {
        return Err(ControlError::new(
            ControlErrorKind::ResourceNotFound,
            format!("screenshot folder does not exist: {}", directory.display()),
        ));
    }
    let filename = model.capture_default_filename()?;
    let filename = Path::new(&filename);
    let stem = filename
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("odon.screenshot");
    let extension = filename
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("png");
    (1..=999_999)
        .map(|index| directory.join(format!("{stem}.{index:04}.{extension}")))
        .find(|candidate| !candidate.exists())
        .ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "no free screenshot filename exists in {}",
                    directory.display()
                ),
            )
        })
}

pub(super) fn screenshot_result(spec: &ScreenshotWriteSpec, bytes: u64) -> Value {
    let mut screenshot = json!({
        "queued":true,
        "completed":true,
        "path":spec.path.to_string_lossy(),
        "bytes":bytes,
        "capture_id":spec.capture_id,
        "presented_projection_revision":spec.desired_projection_revision,
    });
    match &spec.scope {
        PresentationCaptureScope::Viewer { viewport_id } => {
            screenshot["viewport_id"] = json!(viewport_id);
            screenshot["scope"] = json!(spec.scope.as_str());
            json!({
                "mode":spec.mode.as_str(),
                "screenshot":screenshot,
            })
        }
        PresentationCaptureScope::Workspace => {
            let mut value = screenshot;
            value["scope"] = json!("workspace");
            value
        }
        PresentationCaptureScope::Window => screenshot,
        PresentationCaptureScope::Project => json!({
            "project":spec.project_transition,
            "screenshot":screenshot,
        }),
    }
}
