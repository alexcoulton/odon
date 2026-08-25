//! Required local control runtime shared by native UI and remote transports.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use serde_json::{Value, json};

use crate::control::{
    ControlCommand, ControlError, EventHub, OdonControlRequest, ResourceRegistry, TaskRegistry,
    UiRegistry,
};

use super::bridge::{ControlServerServices, spawn_control_server};

pub const DEFAULT_ADDR: &str = "127.0.0.1:0";
const NATIVE_COMMAND_QUEUE_CAPACITY: usize = 512;

#[derive(Debug, Clone)]
pub struct NativeCommand {
    pub method: String,
    pub params: Value,
}

/// Bounded native-UI ingress that forwards typed commands to the control actor without waiting
/// for another `RootApp::update` call.
#[derive(Debug, Clone)]
pub struct NativeCommandIngress {
    tx: Sender<NativeCommand>,
    recorded_rx: Option<Receiver<NativeCommand>>,
    pending_methods: Arc<Mutex<HashMap<String, usize>>>,
}

impl NativeCommandIngress {
    fn actor(
        actor_tx: Sender<OdonControlRequest>,
        event_hub: Arc<EventHub>,
        task_registry: Arc<TaskRegistry>,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = crossbeam_channel::bounded::<NativeCommand>(NATIVE_COMMAND_QUEUE_CAPACITY);
        let pending_methods = Arc::new(Mutex::new(HashMap::<String, usize>::new()));
        let worker_pending_methods = Arc::clone(&pending_methods);
        std::thread::Builder::new()
            .name("odon-native-command-ingress".to_string())
            .spawn(move || {
                while let Ok(native) = rx.recv() {
                    let method = native.method.clone();
                    let Ok(command) = ControlCommand::decode(&native.method, native.params) else {
                        clear_pending_method(&worker_pending_methods, &method);
                        continue;
                    };
                    let (reply, result) = crossbeam_channel::bounded(1);
                    if actor_tx
                        .send(OdonControlRequest {
                            command,
                            reply,
                            session_id: "native-ui".to_string(),
                            request_id: None,
                            event_hub: Arc::clone(&event_hub),
                            task_registry: Arc::clone(&task_registry),
                            task_id: None,
                        })
                        .is_err()
                    {
                        clear_pending_method(&worker_pending_methods, &method);
                        break;
                    }
                    let _ = result.recv();
                    clear_pending_method(&worker_pending_methods, &method);
                }
            })?;
        Ok(Self {
            tx,
            recorded_rx: None,
            pending_methods,
        })
    }

    /// A bounded, non-consuming ingress used by renderer-only tests and before a renderer is
    /// attached to its required application runtime.
    pub fn detached() -> Self {
        let (tx, rx) = crossbeam_channel::bounded::<NativeCommand>(NATIVE_COMMAND_QUEUE_CAPACITY);
        Self {
            tx,
            recorded_rx: Some(rx),
            pending_methods: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Applies bounded backpressure to the native interaction that committed the command.
    pub fn submit(&self, method: impl Into<String>, params: Value) -> bool {
        let method = method.into();
        *self
            .pending_methods
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(method.clone())
            .or_default() += 1;
        if self
            .tx
            .send(NativeCommand {
                method: method.clone(),
                params,
            })
            .is_ok()
        {
            true
        } else {
            clear_pending_method(&self.pending_methods, &method);
            false
        }
    }

    pub fn contains_pending(&self, method: &str) -> bool {
        self.pending_methods
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(method)
            .is_some_and(|count| *count > 0)
    }

    pub fn take_recorded(&self) -> Vec<NativeCommand> {
        let commands: Vec<NativeCommand> = self
            .recorded_rx
            .as_ref()
            .map(|rx| rx.try_iter().collect())
            .unwrap_or_default();
        for command in &commands {
            clear_pending_method(&self.pending_methods, &command.method);
        }
        commands
    }
}

fn clear_pending_method(pending_methods: &Mutex<HashMap<String, usize>>, method: &str) {
    let mut pending = pending_methods
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(count) = pending.get_mut(method) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            pending.remove(method);
        }
    }
}

#[derive(Debug)]
pub struct OdonControlRuntime {
    tx: Sender<OdonControlRequest>,
    presentation_rx: Receiver<crate::control::actor::RenderProjection>,
    presentation_capture_rx: Receiver<crate::control::actor::PresentationCaptureRequest>,
    presentation_completion_tx: Sender<crate::control::actor::PresentationCaptureCompletion>,
    platform_effect_rx: Receiver<crate::control::actor::PlatformEffect>,
    actor_model_tx: Sender<crate::control::actor::ActorModelUpdate>,
    local_addr: Option<SocketAddr>,
    manifest: Option<crate::control::discovery::InstanceManifestGuard>,
    server_error: Option<String>,
    event_hub: Arc<EventHub>,
    ui_registry: Arc<UiRegistry>,
    task_registry: Arc<TaskRegistry>,
    resource_registry: Arc<ResourceRegistry>,
    _task_service: crate::control::TaskServiceHandle,
    actor_diagnostics: Arc<crate::control::actor::ActorDiagnostics>,
    native_command_ingress: NativeCommandIngress,
}

/// Backward-compatible name for callers that explicitly create a TCP-exposed runtime.
pub type OdonControlBridge = OdonControlRuntime;

impl OdonControlRuntime {
    pub fn spawn_default(ctx: egui::Context) -> anyhow::Result<Self> {
        Self::spawn_inner(DEFAULT_ADDR, ctx, true, false, None, None, None)
    }

    pub fn spawn_default_with_object_loader(
        ctx: egui::Context,
        object_loader: Arc<dyn crate::model::ObjectResourceLoader>,
    ) -> anyhow::Result<Self> {
        Self::spawn_inner(
            DEFAULT_ADDR,
            ctx,
            true,
            false,
            Some(object_loader),
            None,
            None,
        )
    }

    pub fn spawn_default_with_services(
        ctx: egui::Context,
        object_loader: Arc<dyn crate::model::ObjectResourceLoader>,
        dataset_inspector: Arc<dyn crate::data::document::DatasetInspector>,
        alternate_backend: Arc<dyn crate::data::document::AlternateDatasetBackend>,
    ) -> anyhow::Result<Self> {
        Self::spawn_inner(
            DEFAULT_ADDR,
            ctx,
            true,
            false,
            Some(object_loader),
            Some(dataset_inspector),
            Some(alternate_backend),
        )
    }

    pub fn spawn(addr: &str, ctx: egui::Context) -> anyhow::Result<Self> {
        Self::spawn_inner(addr, ctx, false, true, None, None, None)
    }

    pub(super) fn spawn_inner(
        addr: &str,
        ctx: egui::Context,
        publish: bool,
        server_required: bool,
        object_loader: Option<Arc<dyn crate::model::ObjectResourceLoader>>,
        dataset_inspector: Option<Arc<dyn crate::data::document::DatasetInspector>>,
        alternate_backend: Option<Arc<dyn crate::data::document::AlternateDatasetBackend>>,
    ) -> anyhow::Result<Self> {
        let event_hub = EventHub::shared();
        let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));
        let resource_registry = ResourceRegistry::shared(Arc::clone(&event_hub));
        let ui_registry = UiRegistry::shared(Arc::clone(&event_hub));
        let actor = crate::control::actor::spawn_control_actor_with_services_and_ui(
            Arc::new({
                let ctx = ctx.clone();
                move || ctx.request_repaint()
            }),
            Arc::clone(&resource_registry),
            object_loader,
            dataset_inspector,
            Some(Arc::clone(&task_registry)),
            None,
            alternate_backend,
            Some(Arc::clone(&ui_registry)),
        )?;
        let tx = actor.request_tx;
        let presentation_rx = actor.presentation_rx;
        let presentation_capture_rx = actor.presentation_capture_rx;
        let presentation_completion_tx = actor.presentation_completion_tx;
        let platform_effect_rx = actor.platform_effect_rx;
        let actor_model_tx = actor.model_tx;
        let task_service = actor.task_service;
        let actor_diagnostics = actor.diagnostics;
        let native_command_ingress = NativeCommandIngress::actor(
            tx.clone(),
            Arc::clone(&event_hub),
            Arc::clone(&task_registry),
        )?;

        let server = spawn_control_server(
            addr,
            publish,
            ctx.clone(),
            tx.clone(),
            ControlServerServices {
                event_hub: Arc::clone(&event_hub),
                task_registry: Arc::clone(&task_registry),
                task_service: task_service.clone(),
                resource_registry: Arc::clone(&resource_registry),
                ui_registry: Arc::clone(&ui_registry),
                actor_diagnostics: Arc::clone(&actor_diagnostics),
            },
        );
        let (local_addr, manifest, server_error) = match server {
            Ok(server) => (Some(server.local_addr), server.manifest, None),
            Err(error) if server_required => return Err(error),
            Err(error) => (None, None, Some(error.to_string())),
        };
        Ok(Self {
            tx,
            presentation_rx,
            presentation_capture_rx,
            presentation_completion_tx,
            platform_effect_rx,
            actor_model_tx,
            local_addr,
            manifest,
            server_error,
            event_hub,
            ui_registry,
            task_registry,
            resource_registry,
            _task_service: task_service,
            actor_diagnostics,
            native_command_ingress,
        })
    }

    pub fn native_command_ingress(&self) -> NativeCommandIngress {
        self.native_command_ingress.clone()
    }

    pub fn try_recv_presentation(
        &self,
    ) -> Result<crate::control::actor::RenderProjection, crossbeam_channel::TryRecvError> {
        self.presentation_rx.try_recv()
    }

    pub fn pending_presentation_len(&self) -> usize {
        self.presentation_rx.len()
    }

    pub fn try_recv_presentation_capture(
        &self,
    ) -> Result<crate::control::actor::PresentationCaptureRequest, crossbeam_channel::TryRecvError>
    {
        self.presentation_capture_rx.try_recv()
    }

    pub fn presentation_completion_sender(
        &self,
    ) -> Sender<crate::control::actor::PresentationCaptureCompletion> {
        self.presentation_completion_tx.clone()
    }

    pub fn try_recv_platform_effect(
        &self,
    ) -> Result<crate::control::actor::PlatformEffect, crossbeam_channel::TryRecvError> {
        self.platform_effect_rx.try_recv()
    }

    pub fn bootstrap_dataset_model(
        &self,
        dataset: crate::data::ome::OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
    ) {
        let _ = self
            .actor_model_tx
            .send(crate::control::actor::ActorModelUpdate::BootstrapDataset { dataset, store });
    }

    pub fn report_renderer_capabilities(&self, gpu_available: bool) {
        let _ = self
            .actor_model_tx
            .send(crate::control::actor::ActorModelUpdate::RendererCapabilities { gpu_available });
    }

    pub fn bootstrap_model_mode(&self, mode: crate::model::ModelMode) {
        let _ = self
            .actor_model_tx
            .send(crate::control::actor::ActorModelUpdate::BootstrapMode(mode));
    }

    pub fn bootstrap_mosaic_model(&self, resource: crate::model::ControlMosaicResource) {
        let _ = self
            .actor_model_tx
            .send(crate::control::actor::ActorModelUpdate::BootstrapMosaic { resource });
    }

    pub fn bootstrap_project_model(&self, snapshot: crate::model::ProjectModelSnapshot) {
        let _ =
            self.actor_model_tx
                .send(crate::control::actor::ActorModelUpdate::BootstrapProject(
                    snapshot,
                ));
    }

    pub fn bootstrap_settings(
        &self,
        settings: crate::settings::AppSettings,
        path: Option<PathBuf>,
    ) {
        let recent_project_exists = settings
            .recent_projects
            .iter()
            .map(|project| (project.path.clone(), project.path.exists()))
            .collect();
        let _ =
            self.actor_model_tx
                .send(crate::control::actor::ActorModelUpdate::BootstrapSettings {
                    settings,
                    path,
                    recent_project_exists,
                });
    }

    pub fn report_presentation_applied(&self, revision: u64) -> bool {
        self.actor_model_tx
            .try_send(crate::control::actor::ActorModelUpdate::PresentationApplied(revision))
            .is_ok()
    }

    pub fn report_viewport_geometry(
        &self,
        viewport_id: String,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> bool {
        self.actor_model_tx
            .try_send(crate::control::actor::ActorModelUpdate::ViewportGeometry {
                viewport_id,
                x,
                y,
                width,
                height,
            })
            .is_ok()
    }

    pub fn report_renderer_observation(
        &self,
        observation: Value,
        based_on_projection_revision: u64,
    ) -> bool {
        self.actor_model_tx
            .try_send(
                crate::control::actor::ActorModelUpdate::RendererObservation {
                    observation,
                    based_on_projection_revision,
                },
            )
            .is_ok()
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
            .expect("control bridge was created without a TCP server")
    }

    pub fn instance_manifest(&self) -> Option<&crate::control::discovery::InstanceManifest> {
        self.manifest.as_ref().map(|guard| guard.manifest())
    }

    pub fn server_error(&self) -> Option<&str> {
        self.server_error.as_deref()
    }

    pub fn actor_is_alive(&self) -> bool {
        self.actor_diagnostics.snapshot()["alive"]
            .as_bool()
            .unwrap_or(false)
    }

    pub fn revision(&self) -> u64 {
        self.event_hub.revision()
    }

    pub fn ui_registry(&self) -> Arc<UiRegistry> {
        Arc::clone(&self.ui_registry)
    }

    pub fn render_extension_ui(&self, ctx: &egui::Context, native_state: &Value) {
        self.prepare_extension_ui(native_state);
        self.render_extension_hosts(ctx, native_state.get("shell"), false);
        self.finish_extension_ui(ctx);
    }

    pub fn prepare_extension_ui(&self, native_state: &Value) {
        self.ui_registry
            .sync_native_bindings(native_state, &self.resource_registry.list_layers());
    }

    pub fn render_extension_hosts(
        &self,
        ctx: &egui::Context,
        shell: Option<&Value>,
        show_extension_manager: bool,
    ) {
        self.ui_registry.render(ctx, shell, show_extension_manager);
    }

    pub fn finish_extension_ui(&self, ctx: &egui::Context) {
        for action in self.ui_registry.drain_actions() {
            match action.action.get("type").and_then(Value::as_str) {
                Some("command") => {
                    let Some(method) = action.action.get("method").and_then(Value::as_str) else {
                        continue;
                    };
                    let params = action
                        .action
                        .get("params")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    self.queue_native_command(ctx, action.owner_session_id, method, params);
                }
                Some("bind")
                    if action.action.get("target").and_then(Value::as_str)
                        == Some("viewer.layers") =>
                {
                    let Some(layer_id) = action.action.get("layer_id").and_then(Value::as_str)
                    else {
                        continue;
                    };
                    let Some(property) = action.action.get("property").and_then(Value::as_str)
                    else {
                        continue;
                    };
                    if matches!(property, "opacity" | "visible") {
                        let mut patch = serde_json::Map::new();
                        patch.insert("layer_id".to_string(), Value::String(layer_id.to_string()));
                        patch.insert(property.to_string(), action.value);
                        self.queue_native_command(
                            ctx,
                            action.owner_session_id,
                            "viewer.layers.update",
                            Value::Object(patch),
                        );
                    }
                }
                Some("bind") => {
                    let target = action.action.get("target").and_then(Value::as_str);
                    let property = action.action.get("property").and_then(Value::as_str);
                    let command = match (target, property) {
                        (Some("viewer.channels"), Some("active")) => Some((
                            "viewer.channels.set_active",
                            if let Some(index) = action.value.as_u64() {
                                json!({"index": index})
                            } else {
                                json!({"name": action.value})
                            },
                        )),
                        (Some("viewer.channels"), Some("visible")) => Some((
                            "viewer.channels.set_visible",
                            json!({"channels": action.value, "mode": "only"}),
                        )),
                        (Some("viewer.camera"), Some("zoom")) => {
                            Some(("viewer.camera.set", json!({"zoom": action.value})))
                        }
                        (Some("viewer"), Some("smooth_pixels")) => {
                            Some(("set_smooth_pixels", json!({"smooth": action.value})))
                        }
                        _ => None,
                    };
                    if let Some((method, params)) = command {
                        self.queue_native_command(ctx, action.owner_session_id, method, params);
                    }
                }
                _ => {}
            }
        }
    }

    fn queue_native_command(
        &self,
        ctx: &egui::Context,
        session_id: String,
        method: &str,
        params: Value,
    ) -> bool {
        let Ok(command) = ControlCommand::decode(method, params) else {
            return false;
        };
        let (reply, _result) = crossbeam_channel::bounded(1);
        let sent = self
            .tx
            .try_send(OdonControlRequest {
                command,
                reply,
                session_id,
                request_id: None,
                event_hub: Arc::clone(&self.event_hub),
                task_registry: Arc::clone(&self.task_registry),
                task_id: None,
            })
            .is_ok();
        if sent {
            ctx.request_repaint();
        }
        sent
    }

    pub fn submit_native_command(&self, ctx: &egui::Context, method: &str, params: Value) -> bool {
        self.queue_native_command(ctx, "native-ui".to_string(), method, params)
    }

    pub fn submit_native_command_with_reply(
        &self,
        ctx: &egui::Context,
        method: &str,
        params: Value,
    ) -> Option<Receiver<Result<Value, ControlError>>> {
        let command = ControlCommand::decode(method, params).ok()?;
        let (reply, result) = crossbeam_channel::bounded(1);
        self.tx
            .try_send(OdonControlRequest {
                command,
                reply,
                session_id: "native-ui".to_string(),
                request_id: None,
                event_hub: Arc::clone(&self.event_hub),
                task_registry: Arc::clone(&self.task_registry),
                task_id: None,
            })
            .ok()?;
        ctx.request_repaint();
        Some(result)
    }

    pub fn external_layers(
        &self,
    ) -> (
        u64,
        Vec<crate::control::LayerSnapshot>,
        Vec<crate::control::DataResourceSnapshot>,
    ) {
        (
            self.event_hub.revision(),
            self.resource_registry.list_layers(),
            self.resource_registry.list_resources(),
        )
    }

    pub fn project_control_manifest(&self) -> (Vec<Value>, Vec<Value>) {
        self.resource_registry.project_manifest()
    }

    pub fn replace_project_control_manifest(
        &self,
        resources: &[Value],
        layers: &[Value],
    ) -> Result<(), ControlError> {
        self.resource_registry
            .replace_project_manifest(resources, layers)
    }
}
