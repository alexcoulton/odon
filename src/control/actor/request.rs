use super::*;

pub(super) fn reject_worker_submission(
    request: OdonControlRequest,
    diagnostics: &ActorDiagnostics,
) {
    diagnostics
        .rejected_requests
        .fetch_add(1, Ordering::Relaxed);
    diagnostics.record_reply_time(request.command.queue_age());
    let _ = request.reply.send(Err(ControlError::new(
        ControlErrorKind::NotReady,
        "Odon's resource worker queue is unavailable; retry later",
    )));
}

pub(super) fn finish_request(
    request: OdonControlRequest,
    mut response: Value,
    diagnostics: &ActorDiagnostics,
) {
    let mutates = request.command.mutates();
    let revision = if mutates {
        request.event_hub.next_revision()
    } else {
        request.event_hub.revision()
    };
    if let Some(object) = response.as_object_mut() {
        object.insert("_control".to_string(), json!({"revision": revision}));
    }
    if mutates {
        let event_data = response.clone();
        let method = request.command.method();
        let source = request
            .command
            .params()
            .get("viewport_id")
            .and_then(Value::as_str)
            .map(|id| format!("viewport:{id}"))
            .unwrap_or_else(|| {
                if method.starts_with("ui.shell.")
                    || method.starts_with("ui.commands.")
                    || method.starts_with("ui.menus.")
                    || method.starts_with("ui.toolbars.")
                    || method.starts_with("ui.palette.")
                {
                    "application:shell".to_string()
                } else if method.starts_with("datasets.") {
                    "application".to_string()
                } else if method.starts_with("project.") {
                    "project:active".to_string()
                } else {
                    "viewer:active".to_string()
                }
            });
        let primary_event = request
            .command
            .event_name()
            .unwrap_or("application.state.changed");
        request.event_hub.publish(
            primary_event,
            &source,
            revision,
            json!({"method": method, "result": event_data}),
            Some(request.session_id.clone()),
            request.request_id.clone(),
        );
        if primary_event != "ui.shell.changed"
            && let Some(change) = response.get("shell_change").cloned()
        {
            request.event_hub.publish(
                "ui.shell.changed",
                "application:shell",
                revision,
                json!({"method":method,"change":change}),
                Some(request.session_id.clone()),
                request.request_id.clone(),
            );
        }
        if response
            .get("active_viewport_changed")
            .and_then(Value::as_bool)
            == Some(true)
            && let Some(legacy_event) = active_viewport_compatibility_event(method)
            && legacy_event != primary_event
        {
            request.event_hub.publish(
                legacy_event,
                "viewer:active",
                revision,
                json!({
                    "method": method,
                    "result": response,
                    "caused_by_event": primary_event,
                }),
                Some(request.session_id.clone()),
                request.request_id.clone(),
            );
        }
    }
    diagnostics.record_reply_time(request.command.queue_age());
    let _ = request.reply.send(Ok(response));
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
        "viewer.viewports.layers.set"
        | "viewer.viewports.layers.set_visibility"
        | "viewer.viewports.layers.set_order"
        | "viewer.viewports.layers.set_active"
        | "viewer.viewports.layers.state.replace" => Some("viewer.layers.changed"),
        _ => None,
    }
}

pub(super) fn expand_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}
