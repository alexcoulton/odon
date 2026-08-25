use super::*;

fn task_id<'a>(method: &str, params: &'a Value) -> Result<&'a str, ControlError> {
    params
        .get("task_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ControlError::invalid_params(method, "task_id is required"))
}

pub(super) fn get_task(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serde_json::to_value(state.task_service.get(task_id("tasks.get", params)?)?).map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize task: {error}"),
        )
    })
}

pub(super) fn list_tasks(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let include_finished = params
        .get("include_finished")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    Ok(json!({"tasks": state.task_service.list(include_finished)?}))
}

pub(super) fn cancel_task(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    serde_json::to_value(
        state
            .task_service
            .cancel(task_id("tasks.cancel", params)?)?,
    )
    .map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize task: {error}"),
        )
    })
}

pub(super) fn forget_task(params: &Value, state: &ConnectionState) -> Result<Value, ControlError> {
    let id = task_id("tasks.forget", params)?;
    state.task_service.forget(id)?;
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

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtensionLayoutListRequest {
    extension_id: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtensionLayoutRemoveRequest {
    extension_id: String,
    name: String,
}

fn extension_layout_id(method: &str, id: String) -> Result<String, ControlError> {
    if id.trim().is_empty() || id.len() > 256 {
        Err(ControlError::invalid_params(
            method,
            "extension_id must contain 1 to 256 characters",
        ))
    } else {
        Ok(id)
    }
}

pub(super) fn register_extension(
    params: Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    serialize_control(
        state
            .ui_registry
            .register_extension(params, &state.hello_server.session_id)?,
    )
}

pub(super) fn remove_extension(
    params: &Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    let id = required_id("ui.extensions.remove", "extension_id", params)?;
    state
        .ui_registry
        .remove_extension(id, &state.hello_server.session_id)?;
    Ok(json!({"extension_id": id, "removed": true}))
}

pub(super) fn set_extension_readiness(
    params: Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    serialize_control(
        state
            .ui_registry
            .set_extension_readiness(params, &state.hello_server.session_id)?,
    )
}

pub(super) fn register_extension_layout(
    params: Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    serialize_control(
        state
            .ui_registry
            .register_extension_layout(params, &state.hello_server.session_id)?,
    )
}

pub(super) fn list_extension_layouts(
    params: &Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    let request: ExtensionLayoutListRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(
                "ui.extensions.layouts.list",
                format!("invalid extension layout list request: {error}"),
            )
        })?;
    let id = extension_layout_id("ui.extensions.layouts.list", request.extension_id)?;
    let layouts = state
        .ui_registry
        .list_extension_layouts(&id, &state.hello_server.session_id)?;
    Ok(json!({
        "extension_id":id,
        "layouts":layouts,
        "revision":state.event_hub.revision(),
    }))
}

pub(super) fn remove_extension_layout(
    params: &Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    let request: ExtensionLayoutRemoveRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(
                "ui.extensions.layouts.remove",
                format!("invalid extension layout removal request: {error}"),
            )
        })?;
    let id = extension_layout_id("ui.extensions.layouts.remove", request.extension_id)?;
    state.ui_registry.remove_extension_layout(
        &id,
        &request.name,
        &state.hello_server.session_id,
    )?;
    Ok(json!({"extension_id":id,"name":request.name,"removed":true}))
}

pub(super) fn register_contribution(
    params: Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    serialize_control(
        state
            .ui_registry
            .register_contribution(params, &state.hello_server.session_id)?,
    )
}

pub(super) fn patch_ui_values(
    params: &Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
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

pub(super) fn remove_contribution(
    params: &Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    let id = required_id("ui.contributions.remove", "contribution_id", params)?;
    state
        .ui_registry
        .remove_contribution(id, &state.hello_server.session_id)?;
    Ok(json!({"contribution_id": id, "removed": true}))
}

pub(super) fn subscribe_events(
    params: &Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
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

pub(super) fn unsubscribe_events(
    params: &Value,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
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
