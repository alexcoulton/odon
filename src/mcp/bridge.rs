use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use serde_json::{Value, json};

use crate::control::registry;
use crate::control::{
    ControlCommand, ControlError, ControlErrorKind, EventHub, HelloResponse, HelloServerInfo,
    JsonRpcRequest, ResourceRegistry, TaskRegistry, TaskState, UiRegistry, json_rpc_error,
    json_rpc_result,
};

pub const DEFAULT_ADDR: &str = "127.0.0.1:0";
const MAX_INLINE_PAYLOAD_BYTES: u64 = 1_048_576;

#[derive(Debug)]
pub struct OdonControlBridge {
    rx: Receiver<OdonControlRequest>,
    tx: Sender<OdonControlRequest>,
    local_addr: SocketAddr,
    manifest: Option<crate::control::discovery::InstanceManifestGuard>,
    event_hub: Arc<EventHub>,
    ui_registry: Arc<UiRegistry>,
    task_registry: Arc<TaskRegistry>,
    resource_registry: Arc<ResourceRegistry>,
}

#[derive(Debug)]
struct ControlServerIdentity {
    instance_id: String,
    expected_token: Option<String>,
    allow_legacy: bool,
    event_hub: Arc<EventHub>,
    task_registry: Arc<TaskRegistry>,
    resource_registry: Arc<ResourceRegistry>,
    ui_registry: Arc<UiRegistry>,
}

#[derive(Debug)]
pub struct OdonControlRequest {
    pub command: ControlCommand,
    pub reply: Sender<Result<Value, ControlError>>,
    pub session_id: String,
    pub request_id: Option<Value>,
    pub event_hub: Arc<EventHub>,
    pub task_registry: Arc<TaskRegistry>,
    pub task_id: Option<String>,
}

impl OdonControlBridge {
    pub fn spawn_default(ctx: egui::Context) -> anyhow::Result<Self> {
        Self::spawn_inner(DEFAULT_ADDR, ctx, true)
    }

    pub fn spawn(addr: &str, ctx: egui::Context) -> anyhow::Result<Self> {
        Self::spawn_inner(addr, ctx, false)
    }

    fn spawn_inner(addr: &str, ctx: egui::Context, publish: bool) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let local_addr = listener.local_addr()?;
        listener.set_nonblocking(false)?;
        let manifest = if publish {
            Some(crate::control::discovery::InstanceManifestGuard::publish(
                crate::control::discovery::InstanceManifest::new(local_addr)?,
            )?)
        } else {
            None
        };
        let event_hub = EventHub::shared();
        let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));
        let resource_registry = ResourceRegistry::shared(Arc::clone(&event_hub));
        let ui_registry = UiRegistry::shared(Arc::clone(&event_hub));
        let identity = Arc::new(ControlServerIdentity {
            instance_id: manifest
                .as_ref()
                .map(|guard| guard.manifest().instance_id.clone())
                .unwrap_or(crate::control::discovery::random_uuid_like()?),
            expected_token: manifest
                .as_ref()
                .map(|guard| guard.manifest().token.clone()),
            allow_legacy: !publish,
            event_hub: Arc::clone(&event_hub),
            task_registry: Arc::clone(&task_registry),
            resource_registry: Arc::clone(&resource_registry),
            ui_registry: Arc::clone(&ui_registry),
        });
        let (tx, rx) = crossbeam_channel::bounded::<OdonControlRequest>(256);
        thread::Builder::new()
            .name("odon-control-bridge".to_string())
            .spawn({
                let tx = tx.clone();
                move || serve_control_bridge(listener, tx, local_addr, identity, ctx)
            })
            .map_err(anyhow::Error::from)?;
        Ok(Self {
            rx,
            tx,
            local_addr,
            manifest,
            event_hub,
            ui_registry,
            task_registry,
            resource_registry,
        })
    }

    pub fn try_recv(&self) -> Result<OdonControlRequest, crossbeam_channel::TryRecvError> {
        self.rx.try_recv()
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn pending_len(&self) -> usize {
        self.rx.len()
    }

    pub fn instance_manifest(&self) -> Option<&crate::control::discovery::InstanceManifest> {
        self.manifest.as_ref().map(|guard| guard.manifest())
    }

    pub fn revision(&self) -> u64 {
        self.event_hub.revision()
    }

    pub fn publish_native_event(&self, event: &str, source: &str, data: Value) {
        let revision = self.event_hub.next_revision();
        self.event_hub
            .publish(event, source, revision, data, None, None);
    }

    pub fn render_extension_ui(&self, ctx: &egui::Context, native_state: &Value) {
        self.ui_registry
            .sync_native_bindings(native_state, &self.resource_registry.list_layers());
        self.ui_registry.render(ctx);
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
                        patch.insert(property.to_string(), action.value);
                        let _ = self.resource_registry.update_layer(
                            layer_id,
                            &Value::Object(patch),
                            &action.owner_session_id,
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
    ) {
        let Ok(command) = ControlCommand::decode(method, params) else {
            return;
        };
        let (reply, _result) = crossbeam_channel::bounded(1);
        let _ = self.tx.try_send(OdonControlRequest {
            command,
            reply,
            session_id,
            request_id: None,
            event_hub: Arc::clone(&self.event_hub),
            task_registry: Arc::clone(&self.task_registry),
            task_id: None,
        });
        ctx.request_repaint();
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

fn serve_control_bridge(
    listener: TcpListener,
    tx: Sender<OdonControlRequest>,
    address: SocketAddr,
    identity: Arc<ControlServerIdentity>,
    ctx: egui::Context,
) {
    eprintln!("odon control server listening on {address}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let tx = tx.clone();
                let ctx = ctx.clone();
                let identity = Arc::clone(&identity);
                let _ = thread::Builder::new()
                    .name("odon-control-client".to_string())
                    .spawn(move || handle_control_client(stream, tx, identity, ctx));
            }
            Err(err) => eprintln!("odon control bridge accept failed: {err}"),
        }
    }
}

fn handle_control_client(
    stream: TcpStream,
    tx: Sender<OdonControlRequest>,
    identity: Arc<ControlServerIdentity>,
    ctx: egui::Context,
) {
    let Ok(mut write_stream) = stream.try_clone() else {
        return;
    };
    let (outbound_tx, outbound_rx) = crossbeam_channel::bounded::<Value>(256);
    let writer_thread = thread::Builder::new()
        .name("odon-control-writer".to_string())
        .spawn(move || {
            while let Ok(message) = outbound_rx.recv() {
                if writeln!(write_stream, "{}", message).is_err() || write_stream.flush().is_err() {
                    break;
                }
            }
        });
    if writer_thread.is_err() {
        return;
    }
    let reader = BufReader::new(stream);
    let mut state = match ConnectionState::new(&identity, outbound_tx.clone()) {
        Ok(state) => state,
        Err(_) => return,
    };
    identity
        .event_hub
        .register(state.hello_server.session_id.clone(), outbound_tx.clone());
    let (work_tx, work_rx) = crossbeam_channel::bounded::<ControlWork>(64);
    let mut workers = Vec::new();
    for index in 0..4 {
        let work_rx = work_rx.clone();
        let app_tx = tx.clone();
        let app_ctx = ctx.clone();
        let outbound = outbound_tx.clone();
        if let Ok(worker) = thread::Builder::new()
            .name(format!("odon-control-worker-{index}"))
            .spawn(move || {
                while let Ok(mut work) = work_rx.recv() {
                    if let Some(response) =
                        handle_control_line(&work.line, &app_tx, &app_ctx, &mut work.state)
                    {
                        let _ = outbound.send_timeout(response, Duration::from_secs(5));
                    }
                }
            })
        {
            workers.push(worker);
        }
    }
    for line in reader.lines() {
        let line = match line {
            Ok(line) if line.len() as u64 > MAX_INLINE_PAYLOAD_BYTES => {
                state.close_after_response = true;
                let _ = outbound_tx.send_timeout(
                    json_rpc_error(
                        Value::Null,
                        &ControlError::new(
                            ControlErrorKind::ResourceLimit,
                            "control message exceeds the negotiated inline payload limit",
                        ),
                    ),
                    Duration::from_secs(5),
                );
                break;
            }
            Ok(line) => line,
            Err(err) => {
                let _ = outbound_tx.send_timeout(
                    json_rpc_error(
                        Value::Null,
                        &ControlError::new(
                            ControlErrorKind::Internal,
                            format!("read failed: {err}"),
                        ),
                    ),
                    Duration::from_secs(5),
                );
                break;
            }
        };
        if state.hello_complete {
            match work_tx.try_send(ControlWork {
                line,
                state: state.clone(),
            }) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(_)) => {
                    let _ = outbound_tx.send_timeout(
                        json_rpc_error(
                            Value::Null,
                            &ControlError::new(
                                ControlErrorKind::NotReady,
                                "this control connection's request queue is full",
                            ),
                        ),
                        Duration::from_secs(5),
                    );
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => break,
            }
        } else {
            if let Some(response) = handle_control_line(&line, &tx, &ctx, &mut state)
                && outbound_tx
                    .send_timeout(response, Duration::from_secs(5))
                    .is_err()
            {
                break;
            }
            if state.close_after_response {
                break;
            }
        }
    }
    drop(work_tx);
    for worker in workers {
        let _ = worker.join();
    }
    identity
        .resource_registry
        .cleanup_session(&state.hello_server.session_id);
    identity
        .ui_registry
        .cleanup_session(&state.hello_server.session_id);
    identity.event_hub.remove(&state.hello_server.session_id);
}

struct ControlWork {
    line: String,
    state: ConnectionState,
}

#[derive(Clone)]
struct ConnectionState {
    hello_complete: bool,
    close_after_response: bool,
    allow_legacy: bool,
    hello_server: HelloServerInfo,
    event_hub: Arc<EventHub>,
    task_registry: Arc<TaskRegistry>,
    resource_registry: Arc<ResourceRegistry>,
    ui_registry: Arc<UiRegistry>,
}

impl ConnectionState {
    fn new(identity: &ControlServerIdentity, _outbound: Sender<Value>) -> anyhow::Result<Self> {
        Ok(Self {
            hello_complete: false,
            close_after_response: false,
            allow_legacy: identity.allow_legacy,
            hello_server: HelloServerInfo {
                instance_id: identity.instance_id.clone(),
                session_id: crate::control::discovery::random_uuid_like()?,
                expected_token: identity.expected_token.clone(),
                max_inline_payload_bytes: MAX_INLINE_PAYLOAD_BYTES,
            },
            event_hub: Arc::clone(&identity.event_hub),
            task_registry: Arc::clone(&identity.task_registry),
            resource_registry: Arc::clone(&identity.resource_registry),
            ui_registry: Arc::clone(&identity.ui_registry),
        })
    }

    fn unauthenticated_test() -> Self {
        let (outbound, _rx) = crossbeam_channel::bounded(8);
        let event_hub = EventHub::shared();
        let identity = ControlServerIdentity {
            instance_id: "test-instance".to_string(),
            expected_token: None,
            allow_legacy: true,
            event_hub: Arc::clone(&event_hub),
            task_registry: TaskRegistry::shared(Arc::clone(&event_hub)),
            resource_registry: ResourceRegistry::shared(Arc::clone(&event_hub)),
            ui_registry: UiRegistry::shared(Arc::clone(&event_hub)),
        };
        let state = Self::new(&identity, outbound.clone()).expect("create test connection state");
        identity
            .event_hub
            .register(state.hello_server.session_id.clone(), outbound);
        state
    }
}

fn handle_control_line(
    line: &str,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &mut ConnectionState,
) -> Option<Value> {
    let value = match serde_json::from_str::<Value>(line) {
        Ok(value) => value,
        Err(err) => {
            return Some(json_rpc_error(
                Value::Null,
                &ControlError::new(ControlErrorKind::ParseError, format!("invalid JSON: {err}")),
            ));
        }
    };
    if value.get("jsonrpc").is_some() {
        handle_json_rpc_request(value, tx, ctx, state)
    } else if !state.allow_legacy {
        Some(json!({
            "ok": false,
            "error": "legacy control requests are disabled on authenticated Odon instances"
        }))
    } else {
        Some(handle_legacy_request(value, tx, ctx))
    }
}

fn handle_json_rpc_request(
    value: Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &mut ConnectionState,
) -> Option<Value> {
    let id = value.get("id").cloned().unwrap_or(Value::Null);
    let request: JsonRpcRequest = match serde_json::from_value(value) {
        Ok(request) => request,
        Err(error) => {
            return Some(json_rpc_error(
                id,
                &ControlError::new(
                    ControlErrorKind::InvalidRequest,
                    format!("invalid JSON-RPC request: {error}"),
                ),
            ));
        }
    };
    if let Err(error) = request.validate() {
        return Some(json_rpc_error(id, &error));
    }
    let is_notification = request.id.is_none();
    let request_id = request.id.clone();

    let result = match request.method.as_str() {
        "system.hello" => match HelloResponse::negotiate(request.params, &state.hello_server) {
            Ok(response) => {
                state.hello_complete = true;
                serde_json::to_value(response).map_err(|error| {
                    ControlError::new(
                        ControlErrorKind::Internal,
                        format!("failed to serialize hello response: {error}"),
                    )
                })
            }
            Err(error) => {
                if error.kind == ControlErrorKind::AuthenticationFailed {
                    state.close_after_response = true;
                }
                Err(error)
            }
        },
        _ if !state.hello_complete => Err(ControlError::new(
            ControlErrorKind::HandshakeRequired,
            "system.hello must be the first request on a control connection",
        )),
        "system.get_capabilities" => Ok(json!({
            "protocol_version": crate::control::PROTOCOL_VERSION,
            "capabilities": registry::capabilities(),
        })),
        "system.list_methods" | "system.describe_methods" => Ok(json!({
            "protocol_version": crate::control::PROTOCOL_VERSION,
            "methods": registry::catalog_json(),
        })),
        "system.describe_events" => Ok(json!({
            "protocol_version": crate::control::PROTOCOL_VERSION,
            "notification_method": "events.event",
            "envelope_fields": [
                "event", "sequence", "revision", "source", "data",
                "initiating_session_id", "initiating_request_id"
            ],
            "families": [
                "application.*", "project.*", "viewer.camera.*", "viewer.channels.*",
                "viewer.layers.*", "viewer.selection.*", "viewer.readiness.*",
                "data.resources.*", "tasks.*", "ui.extensions.*", "ui.contributions.*",
                "ui.extension:<extension-id>.*"
            ]
        })),
        "ui.describe_schema" => Ok(json!({
            "schema_version": 1,
            "max_components": 512,
            "max_depth": 16,
            "locations": [
                "left.sections", "right.tabs", "top_bar.actions", "canvas.controls",
                "status_bar", "project.cards"
            ],
            "components": [
                "panel", "column", "row", "grid", "tabs", "scroll", "group", "collapsible",
                "text", "markdown", "status", "warning", "error", "spinner", "separator",
                "spacer", "button", "toggle", "checkbox", "slider", "number", "integer",
                "text_input", "select", "radio", "multi_select", "color", "progress"
            ],
            "actions": ["emit", "command", "bind"],
            "event_policies": ["commit", "immediate", "throttle", "debounce"]
        })),
        "system.batch" => run_batch(&request.params, tx, ctx, state),
        "system.get_diagnostics" => Ok(json!({
            "control_queue_depth": tx.len(),
            "events": state.event_hub.diagnostics(),
            "tasks": state.task_registry.list(true),
            "data_resource_count": state.resource_registry.list_resources().len(),
            "external_layer_count": state.resource_registry.list_layers().len(),
            "extensions": state.ui_registry.list_extensions(),
            "contribution_count": state.ui_registry.list_contributions().len(),
        })),
        "events.subscribe" => subscribe_events(&request.params, state),
        "events.unsubscribe" => unsubscribe_events(&request.params, state),
        "events.get_status" => Ok(state.event_hub.status(&state.hello_server.session_id)),
        "tasks.start" => start_task(&request.params, tx, ctx, state),
        "tasks.get" => get_task(&request.params, state),
        "tasks.list" => list_tasks(&request.params, state),
        "tasks.cancel" => cancel_task(&request.params, state),
        "tasks.forget" => forget_task(&request.params, state),
        "data.resources.register" => register_resource(request.params, state),
        "data.resources.list" => Ok(json!({
            "resources": state.resource_registry.list_resources(),
            "revision": state.event_hub.revision(),
        })),
        "data.resources.get" => get_resource(&request.params, state),
        "data.resources.remove" => remove_resource(&request.params, state),
        "viewer.layers.add" => add_layer(request.params, state),
        "viewer.layers.list" => Ok(json!({
            "layers": state.resource_registry.list_layers(),
            "revision": state.event_hub.revision(),
        })),
        "viewer.layers.get" => get_layer(&request.params, state),
        "viewer.layers.update" => update_layer(&request.params, state),
        "viewer.layers.remove" => remove_layer(&request.params, state),
        "viewer.layers.reorder" => reorder_layers(&request.params, state),
        "ui.extensions.register" => register_extension(request.params, state),
        "ui.extensions.list" => Ok(json!({
            "extensions": state.ui_registry.list_extensions(),
            "revision": state.event_hub.revision(),
        })),
        "ui.extensions.remove" => remove_extension(&request.params, state),
        "ui.contributions.register" => register_contribution(request.params, state),
        "ui.contributions.list" => Ok(json!({
            "contributions": state.ui_registry.list_contributions(),
            "revision": state.event_hub.revision(),
        })),
        "ui.contributions.patch_values" => patch_ui_values(&request.params, state),
        "ui.contributions.remove" => remove_contribution(&request.params, state),
        method => dispatch_to_app(method, request.params, tx, ctx, state, request_id),
    };

    if is_notification {
        return None;
    }
    Some(match result {
        Ok(value) => json_rpc_result(id, value),
        Err(error) => json_rpc_error(id, &error),
    })
}

fn handle_legacy_request(
    value: Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
) -> Value {
    let Some(method) = value.get("method").and_then(Value::as_str) else {
        return json!({"ok": false, "error": "missing method"});
    };
    let params = value.get("params").cloned().unwrap_or(Value::Null);
    let mut state = ConnectionState::unauthenticated_test();
    match dispatch_to_app(method, params, tx, ctx, &mut state, None) {
        Ok(value) => json!({"ok": true, "result": value}),
        Err(error) => json!({"ok": false, "error": error.message}),
    }
}

fn dispatch_to_app(
    method: &str,
    params: Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &mut ConnectionState,
    request_id: Option<Value>,
) -> Result<Value, ControlError> {
    let command = ControlCommand::decode(method, params)?;
    let (reply_tx, reply_rx) = crossbeam_channel::bounded::<Result<Value, ControlError>>(1);
    match tx.try_send(OdonControlRequest {
        command,
        reply: reply_tx,
        session_id: state.hello_server.session_id.clone(),
        request_id,
        event_hub: Arc::clone(&state.event_hub),
        task_registry: Arc::clone(&state.task_registry),
        task_id: None,
    }) {
        Ok(()) => {}
        Err(crossbeam_channel::TrySendError::Full(_)) => {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon control queue is full; retry later",
            ));
        }
        Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon app is not accepting control requests",
            ));
        }
    }
    ctx.request_repaint();
    match reply_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(value)) => {
            if let Some(message) = value.get("error").and_then(Value::as_str) {
                let kind = if message.contains("No dataset viewer")
                    || message.contains("available in single-image mode")
                {
                    ControlErrorKind::WrongMode
                } else if message.contains("transitioning") {
                    ControlErrorKind::NotReady
                } else {
                    ControlErrorKind::Application
                };
                Err(ControlError::new(kind, message).with_data(json!({
                    "method": method,
                })))
            } else {
                Ok(value)
            }
        }
        Ok(Err(error)) => Err(error),
        Err(_) => Err(ControlError::new(
            ControlErrorKind::Timeout,
            "Odon app did not respond in time",
        )
        .with_data(json!({"method": method}))),
    }
}

fn run_batch(
    params: &Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &mut ConnectionState,
) -> Result<Value, ControlError> {
    if params
        .get("atomic")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(ControlError::new(
            ControlErrorKind::Unsupported,
            "atomic batches are not supported in protocol v1",
        ));
    }
    let operations = params
        .get("operations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ControlError::invalid_params("system.batch", "operations must be an array")
        })?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(ControlError::invalid_params(
            "system.batch",
            "operations must contain between 1 and 128 commands",
        ));
    }
    let mut results = Vec::with_capacity(operations.len());
    for (index, operation) in operations.iter().enumerate() {
        let method = operation
            .get("method")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ControlError::invalid_params(
                    "system.batch",
                    format!("operation {index} has no method"),
                )
            })?;
        if method.starts_with("system.")
            || method.starts_with("events.")
            || method.starts_with("tasks.")
            || method.starts_with("ui.")
            || method.starts_with("data.")
            || method.starts_with("viewer.layers.")
        {
            return Err(ControlError::invalid_params(
                "system.batch",
                format!("operation {index} is not an application command"),
            ));
        }
        let result = dispatch_to_app(
            method,
            operation
                .get("params")
                .cloned()
                .unwrap_or_else(|| json!({})),
            tx,
            ctx,
            state,
            None,
        )?;
        results.push(json!({"method": method, "result": result}));
    }
    Ok(json!({
        "atomic": false,
        "results": results,
        "revision": state.event_hub.revision(),
    }))
}

fn start_task(
    params: &Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    let method = params
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| ControlError::invalid_params("tasks.start", "method is required"))?;
    if method.starts_with("system.")
        || method.starts_with("events.")
        || method.starts_with("tasks.")
    {
        return Err(ControlError::invalid_params(
            "tasks.start",
            "only application control methods can run as tasks",
        ));
    }
    let command = ControlCommand::decode(
        method,
        params.get("params").cloned().unwrap_or_else(|| json!({})),
    )?;
    let operation_method = registry::canonical_method(method).to_string();
    let snapshot = state.task_registry.create(
        params
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or(method),
        state.hello_server.session_id.clone(),
        true,
    )?;
    let task_id = snapshot.task_id.clone();
    let (reply_tx, reply_rx) = crossbeam_channel::bounded::<Result<Value, ControlError>>(1);
    tx.try_send(OdonControlRequest {
        command,
        reply: reply_tx,
        session_id: state.hello_server.session_id.clone(),
        request_id: None,
        event_hub: Arc::clone(&state.event_hub),
        task_registry: Arc::clone(&state.task_registry),
        task_id: Some(task_id.clone()),
    })
    .map_err(|error| match error {
        crossbeam_channel::TrySendError::Full(_) => ControlError::new(
            ControlErrorKind::NotReady,
            "Odon control queue is full; retry later",
        ),
        crossbeam_channel::TrySendError::Disconnected(_) => ControlError::new(
            ControlErrorKind::NotReady,
            "Odon app is not accepting control requests",
        ),
    })?;
    ctx.request_repaint();
    let tasks = Arc::clone(&state.task_registry);
    let app_tx = tx.clone();
    let app_ctx = ctx.clone();
    let session_id = state.hello_server.session_id.clone();
    let event_hub = Arc::clone(&state.event_hub);
    thread::Builder::new()
        .name("odon-control-task".to_string())
        .spawn(move || match reply_rx.recv() {
            Ok(Ok(value)) => {
                let settled = if matches!(
                    operation_method.as_str(),
                    "open_project"
                        | "open_ome_zarr"
                        | "open_tiff"
                        | "open_mosaic_samplesheet"
                        | "open_roi"
                ) {
                    let _ = tasks.progress(&task_id, None, "waiting for viewer readiness");
                    wait_for_application_ready(
                        &operation_method,
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method.starts_with("capture_") {
                    let _ = tasks.progress(&task_id, None, "waiting for screenshot output");
                    wait_for_output_path(&value, &tasks, &task_id)
                } else {
                    Ok(())
                };
                match settled {
                    Ok(()) => {
                        let _ = tasks.complete(&task_id, value);
                    }
                    Err(error) => {
                        if error.kind != ControlErrorKind::Cancelled {
                            let _ = tasks.fail(&task_id, &error);
                        }
                    }
                }
            }
            Ok(Err(error)) => {
                if error.kind != ControlErrorKind::Cancelled {
                    let _ = tasks.fail(&task_id, &error);
                }
            }
            Err(_) => {
                let _ = tasks.fail(
                    &task_id,
                    &ControlError::new(
                        ControlErrorKind::NotReady,
                        "Odon closed before the task completed",
                    ),
                );
            }
        })
        .map_err(|error| {
            ControlError::new(
                ControlErrorKind::Internal,
                format!("failed to start task monitor: {error}"),
            )
        })?;
    serde_json::to_value(snapshot).map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize task: {error}"),
        )
    })
}

fn wait_for_application_ready(
    operation: &str,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &Arc<TaskRegistry>,
    task_id: &str,
) -> Result<(), ControlError> {
    let expected_mode = match operation {
        "open_project" => "project",
        "open_mosaic_samplesheet" => "mosaic",
        _ => "single",
    };
    thread::sleep(Duration::from_millis(50));
    loop {
        ensure_task_not_cancelled(tasks, task_id)?;
        let command = ControlCommand::decode("get_loading_state", json!({}))?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: Arc::clone(tasks),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon stopped accepting readiness checks",
            )
        })?;
        ctx.request_repaint();
        let state = response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon closed during a readiness check",
            )
        })??;
        let mode_matches = state.get("mode").and_then(Value::as_str) == Some(expected_mode);
        let busy = state
            .get("loading")
            .and_then(|loading| loading.get("busy"))
            .and_then(Value::as_bool)
            .or_else(|| state.get("busy").and_then(Value::as_bool))
            .unwrap_or(true);
        if mode_matches && !busy {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn wait_for_output_path(
    value: &Value,
    tasks: &Arc<TaskRegistry>,
    task_id: &str,
) -> Result<(), ControlError> {
    let Some(path) = find_string_field(value, "path") else {
        return Ok(());
    };
    let path = std::path::PathBuf::from(path);
    loop {
        ensure_task_not_cancelled(tasks, task_id)?;
        if std::fs::metadata(&path).is_ok_and(|metadata| metadata.len() > 0) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn ensure_task_not_cancelled(tasks: &TaskRegistry, task_id: &str) -> Result<(), ControlError> {
    if tasks.get(task_id)?.state == TaskState::Cancelled {
        Err(
            ControlError::new(ControlErrorKind::Cancelled, "task was cancelled")
                .with_data(json!({"task_id": task_id})),
        )
    } else {
        Ok(())
    }
}

fn find_string_field<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    match value {
        Value::Object(object) => object.get(field).and_then(Value::as_str).or_else(|| {
            object
                .values()
                .find_map(|value| find_string_field(value, field))
        }),
        Value::Array(values) => values
            .iter()
            .find_map(|value| find_string_field(value, field)),
        _ => None,
    }
}

fn task_id<'a>(method: &str, params: &'a Value) -> Result<&'a str, ControlError> {
    params
        .get("task_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ControlError::invalid_params(method, "task_id is required"))
}

fn get_task(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serde_json::to_value(state.task_registry.get(task_id("tasks.get", params)?)?).map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize task: {error}"),
        )
    })
}

fn list_tasks(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let include_finished = params
        .get("include_finished")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    Ok(json!({"tasks": state.task_registry.list(include_finished)}))
}

fn cancel_task(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serde_json::to_value(
        state
            .task_registry
            .cancel(task_id("tasks.cancel", params)?)?,
    )
    .map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize task: {error}"),
        )
    })
}

fn forget_task(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = task_id("tasks.forget", params)?;
    state.task_registry.forget(id)?;
    Ok(json!({"task_id": id, "forgotten": true}))
}

fn serialize_control(value: impl serde::Serialize) -> Result<Value, ControlError> {
    serde_json::to_value(value).map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize control resource: {error}"),
        )
    })
}

fn required_id<'a>(method: &str, field: &str, params: &'a Value) -> Result<&'a str, ControlError> {
    params
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ControlError::invalid_params(method, format!("{field} is required")))
}

fn register_resource(params: Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serialize_control(
        state
            .resource_registry
            .register_resource(params, &state.hello_server.session_id)?,
    )
}

fn get_resource(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serialize_control(state.resource_registry.get_resource(required_id(
        "data.resources.get",
        "resource_id",
        params,
    )?)?)
}

fn remove_resource(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = required_id("data.resources.remove", "resource_id", params)?;
    state
        .resource_registry
        .remove_resource(id, &state.hello_server.session_id)?;
    Ok(json!({"resource_id": id, "removed": true}))
}

fn add_layer(params: Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serialize_control(
        state
            .resource_registry
            .add_layer(params, &state.hello_server.session_id)?,
    )
}

fn get_layer(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serialize_control(state.resource_registry.get_layer(required_id(
        "viewer.layers.get",
        "layer_id",
        params,
    )?)?)
}

fn update_layer(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = required_id("viewer.layers.update", "layer_id", params)?;
    let mut patch = params.clone();
    patch
        .as_object_mut()
        .map(|object| object.remove("layer_id"));
    serialize_control(state.resource_registry.update_layer(
        id,
        &patch,
        &state.hello_server.session_id,
    )?)
}

fn remove_layer(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = required_id("viewer.layers.remove", "layer_id", params)?;
    state
        .resource_registry
        .remove_layer(id, &state.hello_server.session_id)?;
    Ok(json!({"layer_id": id, "removed": true}))
}

fn reorder_layers(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let order = params
        .get("order")
        .and_then(Value::as_array)
        .ok_or_else(|| ControlError::invalid_params("viewer.layers.reorder", "order is required"))?
        .iter()
        .map(|id| {
            id.as_str().map(str::to_string).ok_or_else(|| {
                ControlError::invalid_params("viewer.layers.reorder", "layer IDs must be strings")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "layers": state.resource_registry.reorder_layers(
            &order,
            &state.hello_server.session_id,
        )?,
        "revision": state.event_hub.revision(),
    }))
}

fn register_extension(params: Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serialize_control(
        state
            .ui_registry
            .register_extension(params, &state.hello_server.session_id)?,
    )
}

fn remove_extension(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = required_id("ui.extensions.remove", "extension_id", params)?;
    state
        .ui_registry
        .remove_extension(id, &state.hello_server.session_id)?;
    Ok(json!({"extension_id": id, "removed": true}))
}

fn register_contribution(params: Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serialize_control(
        state
            .ui_registry
            .register_contribution(params, &state.hello_server.session_id)?,
    )
}

fn patch_ui_values(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = required_id("ui.contributions.patch_values", "contribution_id", params)?;
    let values = params
        .get("values")
        .cloned()
        .ok_or_else(|| {
            ControlError::invalid_params("ui.contributions.patch_values", "values is required")
        })
        .and_then(|value| {
            serde_json::from_value(value).map_err(|error| {
                ControlError::invalid_params(
                    "ui.contributions.patch_values",
                    format!("values must be an object: {error}"),
                )
            })
        })?;
    let if_revision = params.get("if_revision").and_then(Value::as_u64);
    serialize_control(state.ui_registry.patch_values(
        id,
        &values,
        if_revision,
        &state.hello_server.session_id,
    )?)
}

fn remove_contribution(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = required_id("ui.contributions.remove", "contribution_id", params)?;
    state
        .ui_registry
        .remove_contribution(id, &state.hello_server.session_id)?;
    Ok(json!({"contribution_id": id, "removed": true}))
}

fn subscribe_events(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let events = params
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| ControlError::invalid_params("events.subscribe", "events must be an array"))?
        .iter()
        .map(|event| {
            event.as_str().map(str::to_string).ok_or_else(|| {
                ControlError::invalid_params("events.subscribe", "event patterns must be strings")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let subscriptions = state
        .event_hub
        .subscribe(&state.hello_server.session_id, events)
        .map_err(|message| ControlError::invalid_params("events.subscribe", message))?;
    Ok(json!({
        "subscription_id": format!("subscription:{}", state.hello_server.session_id),
        "events": subscriptions,
        "revision": state.event_hub.revision(),
    }))
}

fn unsubscribe_events(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let patterns = match params.get("events") {
        Some(Value::Array(events)) => Some(
            events
                .iter()
                .map(|event| {
                    event.as_str().map(str::to_string).ok_or_else(|| {
                        ControlError::invalid_params(
                            "events.unsubscribe",
                            "event patterns must be strings",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Some(_) => {
            return Err(ControlError::invalid_params(
                "events.unsubscribe",
                "events must be an array when provided",
            ));
        }
        None => None,
    };
    let remaining = state
        .event_hub
        .unsubscribe(&state.hello_server.session_id, patterns.as_deref());
    Ok(json!({"events": remaining, "revision": state.event_hub.revision()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn read_json(reader: &mut BufReader<TcpStream>) -> Value {
        let mut line = String::new();
        reader.read_line(&mut line).expect("read bridge response");
        serde_json::from_str(line.trim()).expect("parse bridge response")
    }

    #[test]
    fn tcp_bridge_validates_envelopes_and_roundtrips_app_replies() {
        let bridge = OdonControlBridge::spawn("127.0.0.1:0", egui::Context::default())
            .expect("spawn bridge on ephemeral port");
        let mut stream = TcpStream::connect(bridge.local_addr()).expect("connect bridge client");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set read timeout");
        let mut reader = BufReader::new(stream.try_clone().expect("clone bridge socket"));

        writeln!(stream, "{{").expect("write malformed JSON");
        let malformed = read_json(&mut reader);
        assert_eq!(malformed["jsonrpc"], "2.0");
        assert!(
            malformed["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid JSON")
        );
        assert_eq!(malformed["error"]["code"], -32700);

        writeln!(stream, "{}", json!({"params": {}})).expect("write missing method");
        let missing = read_json(&mut reader);
        assert_eq!(missing, json!({"ok": false, "error": "missing method"}));

        writeln!(
            stream,
            "{}",
            json!({"method": "set_camera", "params": {"center_x": 12.5}})
        )
        .expect("write valid request");
        stream.flush().expect("flush valid request");
        let deadline = Instant::now() + Duration::from_secs(2);
        let request = loop {
            match bridge.try_recv() {
                Ok(request) => break request,
                Err(crossbeam_channel::TryRecvError::Empty) if Instant::now() < deadline => {
                    std::thread::yield_now();
                }
                Err(error) => panic!("bridge request not delivered: {error}"),
            }
        };
        assert_eq!(request.command.method(), "set_camera");
        assert_eq!(request.command.params()["center_x"], 12.5);
        request
            .reply
            .send(Ok(json!({"center_world_lvl0": [12.5, 0.0]})))
            .expect("reply from app");
        let response = read_json(&mut reader);
        assert_eq!(response["ok"], true);
        assert_eq!(response["result"]["center_world_lvl0"], json!([12.5, 0.0]));
    }

    #[test]
    fn json_rpc_requires_hello_and_exposes_introspection() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let ctx = egui::Context::default();
        let mut state = ConnectionState::unauthenticated_test();

        let before_hello = handle_control_line(
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "get_current_view",
                "params": {}
            })
            .to_string(),
            &tx,
            &ctx,
            &mut state,
        )
        .expect("request response");
        assert_eq!(before_hello["error"]["data"]["kind"], "HANDSHAKE_REQUIRED");

        let hello = handle_control_line(
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "system.hello",
                "params": {
                    "client": {"name": "bridge-test", "version": "1.0.0"},
                    "protocol_versions": [1]
                }
            })
            .to_string(),
            &tx,
            &ctx,
            &mut state,
        )
        .expect("hello response");
        assert_eq!(hello["result"]["protocol_version"], 1);
        assert!(state.hello_complete);

        let methods = handle_control_line(
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "system.list_methods",
                "params": {}
            })
            .to_string(),
            &tx,
            &ctx,
            &mut state,
        )
        .expect("method response");
        assert!(
            methods["result"]["methods"]
                .as_array()
                .is_some_and(|methods| methods.iter().any(|method| method["name"] == "get_camera"))
        );
    }

    #[test]
    fn authenticated_connections_execute_requests_concurrently() {
        let bridge = OdonControlBridge::spawn("127.0.0.1:0", egui::Context::default())
            .expect("spawn bridge on ephemeral port");
        let mut stream = TcpStream::connect(bridge.local_addr()).expect("connect bridge client");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set read timeout");
        let mut reader = BufReader::new(stream.try_clone().expect("clone bridge socket"));

        writeln!(
            stream,
            "{}",
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "system.hello",
                "params": {
                    "client": {"name": "concurrency-test", "version": "1"},
                    "protocol_versions": [1]
                }
            })
        )
        .expect("write hello");
        stream.flush().expect("flush hello");
        assert_eq!(read_json(&mut reader)["id"], 1);

        writeln!(
            stream,
            "{}",
            json!({"jsonrpc": "2.0", "id": 2, "method": "get_camera", "params": {}})
        )
        .expect("write first request");
        writeln!(
            stream,
            "{}",
            json!({"jsonrpc": "2.0", "id": 3, "method": "list_channels", "params": {}})
        )
        .expect("write second request");
        stream.flush().expect("flush concurrent requests");

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut requests = Vec::new();
        while requests.len() < 2 && Instant::now() < deadline {
            match bridge.try_recv() {
                Ok(request) => requests.push(request),
                Err(crossbeam_channel::TryRecvError::Empty) => std::thread::yield_now(),
                Err(error) => panic!("bridge request channel failed: {error}"),
            }
        }
        assert_eq!(
            requests.len(),
            2,
            "both requests should reach the app concurrently"
        );

        let first_index = requests
            .iter()
            .position(|request| request.request_id == Some(json!(2)))
            .expect("request id 2");
        let second_index = requests
            .iter()
            .position(|request| request.request_id == Some(json!(3)))
            .expect("request id 3");
        requests[second_index]
            .reply
            .send(Ok(json!({"channels": []})))
            .expect("reply to second request first");
        assert_eq!(read_json(&mut reader)["id"], 3);
        requests[first_index]
            .reply
            .send(Ok(json!({"zoom": 1.0})))
            .expect("reply to first request second");
        assert_eq!(read_json(&mut reader)["id"], 2);
    }

    #[test]
    fn protocol_registries_roundtrip_data_layers_and_declarative_ui() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let ctx = egui::Context::default();
        let mut state = ConnectionState::unauthenticated_test();

        let call = |state: &mut ConnectionState, id: u64, method: &str, params: Value| {
            handle_control_line(
                &json!({
                    "jsonrpc": "2.0", "id": id, "method": method, "params": params
                })
                .to_string(),
                &tx,
                &ctx,
                state,
            )
            .expect("JSON-RPC response")
        };
        assert!(
            call(
                &mut state,
                1,
                "system.hello",
                json!({
                    "client": {"name": "conformance", "version": "1"},
                    "protocol_versions": [1]
                }),
            )["result"]
                .is_object()
        );
        let methods = call(&mut state, 2, "system.list_methods", json!({}));
        assert!(
            methods["result"]["methods"]
                .as_array()
                .is_some_and(|items| {
                    items.iter().any(|item| item["name"] == "viewer.camera.fit")
                })
        );
        let resource = call(
            &mut state,
            3,
            "data.resources.register",
            json!({
                "resource_id": "resource:test", "uri": "file:///tmp/test.zarr",
                "format": "ome-zarr",
                "coordinate_space": {"axes": ["y", "x"], "scale": [1.0, 1.0]}
            }),
        );
        assert_eq!(resource["result"]["resource_id"], "resource:test");
        let layer = call(
            &mut state,
            4,
            "viewer.layers.add",
            json!({
                "layer_id": "layer:test", "name": "Test", "kind": "labels",
                "data_resource_id": "resource:test"
            }),
        );
        assert_eq!(layer["result"]["layer_id"], "layer:test");
        let extension = call(
            &mut state,
            5,
            "ui.extensions.register",
            json!({
                "id": "org.example.test", "name": "Test", "version": "1",
                "capabilities": ["ui.panels"]
            }),
        );
        assert_eq!(extension["result"]["id"], "org.example.test");
        let contribution = call(
            &mut state,
            6,
            "ui.contributions.register",
            json!({
                "extension_id": "org.example.test", "location": "right.tabs",
                "root": {"id": "root", "type": "panel", "children": [
                    {"id": "run", "type": "button", "label": "Run",
                     "action": {"type": "emit", "event": "run"}}
                ]}
            }),
        );
        assert_eq!(contribution["result"]["extension_id"], "org.example.test");
    }
}
