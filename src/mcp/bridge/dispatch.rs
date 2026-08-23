use super::*;

pub(super) fn handle_control_line(
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
        "system.get_application_surface" => crate::control::application_surface_json(),
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
        "system.get_diagnostics" => state.task_service.list(true).map(|tasks| {
            json!({
                "control_queue_depth": tx.len(),
                "dispatch": crate::control::actor::execution_diagnostics(&state.actor_diagnostics),
                "events": state.event_hub.diagnostics(),
                "tasks": tasks,
                "data_resource_count": state.resource_registry.list_resources().len(),
                "external_layer_count": state.resource_registry.list_layers().len(),
                "extensions": state.ui_registry.list_extensions(),
                "contribution_count": state.ui_registry.list_contributions().len(),
            })
        }),
        "events.subscribe" => subscribe_events(&request.params, state),
        "events.unsubscribe" => unsubscribe_events(&request.params, state),
        "events.get_status" => Ok(state.event_hub.status(&state.hello_server.session_id)),
        "tasks.start" => start_task(&request.params, tx, ctx, state),
        "tasks.get" => get_task(&request.params, state),
        "tasks.list" => list_tasks(&request.params, state),
        "tasks.cancel" => cancel_task(&request.params, state),
        "tasks.forget" => forget_task(&request.params, state),
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

pub(super) fn dispatch_to_app(
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
