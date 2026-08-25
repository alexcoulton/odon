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
    if state.hello_complete
        && request.method != "system.hello"
        && let Err(error) = validate_session_method_capability(&request.method, state)
    {
        return (!is_notification).then(|| json_rpc_error(id, &error));
    }

    let result = match request.method.as_str() {
        "system.hello" => match HelloResponse::negotiate(request.params, &state.hello_server) {
            Ok(response) => {
                state.ui_registry.set_session_capabilities(
                    &state.hello_server.session_id,
                    &response.granted_capabilities,
                );
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
            "granted_capabilities":state.ui_registry.session_capabilities(
                &state.hello_server.session_id
            ),
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
                "ui.commands.*", "ui.menus.*", "ui.toolbars.*", "ui.palette.*", "ui.extension:<extension-id>.*"
            ]
        })),
        "system.get_application_surface" => crate::control::application_surface_json(),
        "ui.describe_schema" => Ok(json!({
            "schema_version": 1,
            "max_components": 512,
            "max_depth": 16,
            "locations": [
                "shell", "left.sections", "right.tabs", "top_bar.actions", "canvas.controls",
                "status_bar", "project.cards"
            ],
            "components": [
                "panel", "column", "row", "grid", "tabs", "scroll", "group", "collapsible",
                "text", "markdown", "status", "warning", "error", "spinner", "separator",
                "spacer", "button", "toggle", "checkbox", "slider", "number", "integer",
                "text_input", "select", "radio", "multi_select", "color", "progress"
            ],
            "actions": ["emit", "command", "bind"],
            "event_policies": ["commit", "immediate", "throttle", "debounce"],
            "state_bindings": {
                "properties":["visible","enabled"],
                "type":"command_state",
                "command_states":["visible","enabled","checked"],
                "missing_command_policy":"false",
                "evaluation":"actor_projection"
            },
            "interaction_backpressure": {
                "coalescing_key":"extension_id:component_id",
                "deferred_retention":"latest_value",
                "minimum_cadence_ms":33,
                "subscriber_pressure":"bounded_drop"
            },
            "extension_layouts": {
                "document_format":"odon.shell-layout",
                "normalized_schema_version":1,
                "accepted_schema_versions":[0,1],
                "max_per_extension":64,
                "required_capability":"ui.panels",
                "disconnect_policies":["remove","disable","retain"],
                "apply_method":"ui.shell.import_layout"
            }
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
        "ui.extensions.register" => (|| -> Result<Value, ControlError> {
            let result = register_extension(request.params, state)?;
            sync_registered_extension_commands(tx, ctx, state, &result)?;
            Ok(result)
        })(),
        "ui.extensions.list" => Ok(json!({
            "extensions": state.ui_registry.list_extensions(),
            "revision": state.event_hub.revision(),
        })),
        "ui.extensions.remove" => (|| -> Result<Value, ControlError> {
            let extension_id = request
                .params
                .get("extension_id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ControlError::invalid_params("ui.extensions.remove", "extension_id is required")
                })?;
            let cleanup = state
                .ui_registry
                .extension_cleanup_context(extension_id, &state.hello_server.session_id)?;
            call_actor_for_state_cleanup(
                tx,
                state,
                "ui.commands.cleanup_extensions",
                json!({"extensions":[cleanup]}),
            )?;
            ctx.request_repaint();
            remove_extension(&request.params, state)
        })(),
        "ui.extensions.set_readiness" => (|| -> Result<Value, ControlError> {
            let result = set_extension_readiness(request.params, state)?;
            sync_registered_extension_commands(tx, ctx, state, &result)?;
            Ok(result)
        })(),
        "ui.extensions.layouts.register" => register_extension_layout(request.params, state),
        "ui.extensions.layouts.list" => list_extension_layouts(&request.params, state),
        "ui.extensions.layouts.remove" => remove_extension_layout(&request.params, state),
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

fn validate_session_method_capability(
    method: &str,
    state: &ConnectionState,
) -> Result<(), ControlError> {
    let Some(required) = registry::capability_for(method) else {
        return Ok(());
    };
    if !required.starts_with("ui.shell.") {
        return Ok(());
    }
    let granted = state
        .ui_registry
        .session_capabilities(&state.hello_server.session_id);
    if granted.iter().any(|capability| {
        capability == required
            || capability == "ui.shell.application_control"
            || (required == "ui.shell.compose" && capability == "ui.shell.extension_place")
    }) {
        return Ok(());
    }
    Err(ControlError::new(
        ControlErrorKind::PermissionDenied,
        format!("{method} requires the '{required}' session capability"),
    )
    .with_data(json!({
        "method":method,
        "required_capability":required,
        "granted_capabilities":granted,
        "resolution":"request the capability during system.hello",
    })))
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
    let component_catalog_mode = (method == "ui.shell.components.list")
        .then(|| {
            params
                .get("mode")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .flatten();
    if method == "ui.shell.replace_layout" {
        if let Some(desired_tree) = params.get("desired_tree") {
            state
                .ui_registry
                .validate_shell_layout_access(desired_tree, &state.hello_server.session_id)?;
        }
    }
    if method == "ui.shell.import_layout"
        && let Some(document) = params.get("document")
        && let Some(desired_tree) = document
            .get("layout")
            .or_else(|| document.get("desired_tree"))
    {
        state
            .ui_registry
            .validate_shell_layout_access(desired_tree, &state.hello_server.session_id)?;
    }
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
        Ok(Ok(mut value)) => {
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
                if value.get("layout").is_some() {
                    state
                        .ui_registry
                        .annotate_shell_snapshot_ownership(&mut value);
                }
                if method == "ui.shell.components.list"
                    && let Some(components) =
                        value.get_mut("components").and_then(Value::as_array_mut)
                {
                    components.extend(
                        state
                            .ui_registry
                            .shell_component_descriptors(component_catalog_mode.as_deref()),
                    );
                    components
                        .sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
                }
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

/// Dispatches a protected native lifecycle request without coupling the TCP transport to command
/// decoding. Disconnect cleanup is best effort, so this deliberately uses a shorter timeout than
/// an ordinary client request.
pub(super) fn call_actor_for_cleanup(
    tx: &Sender<OdonControlRequest>,
    identity: &ControlServerIdentity,
    method: &str,
    params: Value,
) -> Result<Value, ControlError> {
    let command = ControlCommand::decode(method, params)?;
    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    tx.try_send(OdonControlRequest {
        command,
        reply: reply_tx,
        session_id: "native-ui".to_string(),
        request_id: None,
        event_hub: Arc::clone(&identity.event_hub),
        task_registry: Arc::clone(&identity.task_registry),
        task_id: None,
    })
    .map_err(|_| {
        ControlError::new(
            ControlErrorKind::NotReady,
            "application shell could not reconcile extension disconnect focus",
        )
    })?;
    reply_rx.recv_timeout(Duration::from_secs(2)).map_err(|_| {
        ControlError::new(
            ControlErrorKind::Timeout,
            "application shell focus reconciliation timed out",
        )
    })?
}

fn call_actor_for_state_cleanup(
    tx: &Sender<OdonControlRequest>,
    state: &ConnectionState,
    method: &str,
    params: Value,
) -> Result<Value, ControlError> {
    let command = ControlCommand::decode(method, params)?;
    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    tx.try_send(OdonControlRequest {
        command,
        reply: reply_tx,
        session_id: "native-ui".to_string(),
        request_id: None,
        event_hub: Arc::clone(&state.event_hub),
        task_registry: Arc::clone(&state.task_registry),
        task_id: None,
    })
    .map_err(|_| {
        ControlError::new(
            ControlErrorKind::NotReady,
            "application command lifecycle could not be reconciled",
        )
    })?;
    reply_rx.recv_timeout(Duration::from_secs(2)).map_err(|_| {
        ControlError::new(
            ControlErrorKind::Timeout,
            "application command lifecycle reconciliation timed out",
        )
    })?
}

fn sync_registered_extension_commands(
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &ConnectionState,
    extension: &Value,
) -> Result<(), ControlError> {
    let has_actions = extension
        .get("granted_capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|capability| capability.as_str() == Some("ui.actions"));
    if !has_actions {
        return Ok(());
    }
    let extension_id = extension.get("id").and_then(Value::as_str).ok_or_else(|| {
        ControlError::new(ControlErrorKind::Internal, "extension snapshot has no ID")
    })?;
    let context = state
        .ui_registry
        .extension_command_context(extension_id, &state.hello_server.session_id)?;
    call_actor_for_state_cleanup(
        tx,
        state,
        "ui.commands.sync_extension",
        json!({"context":context}),
    )?;
    ctx.request_repaint();
    Ok(())
}
