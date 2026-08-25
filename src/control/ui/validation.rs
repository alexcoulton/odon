use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use super::*;

const MAX_COMPONENTS: usize = 512;
const MAX_DEPTH: usize = 16;

pub(super) fn validate_tree(root: &Component) -> Result<(), ControlError> {
    let mut ids = HashSet::new();
    let mut count = 0;
    validate_component(root, 0, &mut count, &mut ids)
}

fn validate_component(
    component: &Component,
    depth: usize,
    count: &mut usize,
    ids: &mut HashSet<String>,
) -> Result<(), ControlError> {
    *count += 1;
    if depth > MAX_DEPTH || *count > MAX_COMPONENTS {
        return Err(ControlError::new(
            ControlErrorKind::ResourceLimit,
            "component tree exceeds safety limits",
        )
        .with_data(json!({
            "max_components": MAX_COMPONENTS,
            "max_depth": MAX_DEPTH,
        })));
    }
    if component.id.trim().is_empty() || !ids.insert(component.id.clone()) {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            "component IDs must be non-empty and unique",
        ));
    }
    if !matches!(
        component.kind.as_str(),
        "panel"
            | "column"
            | "row"
            | "grid"
            | "tabs"
            | "scroll"
            | "group"
            | "collapsible"
            | "text"
            | "markdown"
            | "status"
            | "warning"
            | "error"
            | "spinner"
            | "separator"
            | "spacer"
            | "button"
            | "toggle"
            | "checkbox"
            | "slider"
            | "number"
            | "integer"
            | "text_input"
            | "select"
            | "radio"
            | "multi_select"
            | "color"
            | "progress"
    ) {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            format!("unsupported component type '{}'", component.kind),
        ));
    }
    if component.kind == "grid"
        && component
            .columns
            .is_some_and(|value| !(1..=16).contains(&value))
    {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            "grid columns must be between 1 and 16",
        ));
    }
    if matches!(component.kind.as_str(), "slider" | "number" | "integer")
        && component
            .minimum
            .zip(component.maximum)
            .is_some_and(|(min, max)| !min.is_finite() || !max.is_finite() || min >= max)
    {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            "numeric component bounds must be finite and increasing",
        ));
    }
    if let Some(policy) = component.event_policy.as_ref() {
        let kind = policy.get("type").and_then(Value::as_str).ok_or_else(|| {
            ControlError::invalid_params(
                "ui.contributions.register",
                "event_policy.type is required",
            )
        })?;
        if !matches!(kind, "commit" | "immediate" | "throttle" | "debounce") {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "unsupported event policy",
            ));
        }
        if matches!(kind, "throttle" | "debounce")
            && !policy
                .get("milliseconds")
                .and_then(Value::as_u64)
                .is_some_and(|value| (1..=60_000).contains(&value))
        {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "throttle/debounce milliseconds must be between 1 and 60000",
            ));
        }
    }
    validate_state_bindings(component)?;
    for child in &component.children {
        validate_component(child, depth + 1, count, ids)?;
    }
    Ok(())
}

fn validate_state_bindings(component: &Component) -> Result<(), ControlError> {
    for (property, binding) in &component.state_bindings {
        if !matches!(property.as_str(), "visible" | "enabled") {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "component state bindings support only visible and enabled properties",
            ));
        }
        let object = binding.as_object().ok_or_else(|| {
            ControlError::invalid_params(
                "ui.contributions.register",
                format!("{property} state binding must be an object"),
            )
        })?;
        if object
            .keys()
            .any(|key| !matches!(key.as_str(), "type" | "command_id" | "state" | "equals"))
        {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                format!("{property} state binding contains an unknown field"),
            ));
        }
        if object.get("type").and_then(Value::as_str) != Some("command_state") {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                format!("{property} state binding type must be 'command_state'"),
            ));
        }
        let command_id = object
            .get("command_id")
            .and_then(Value::as_str)
            .filter(|value| {
                !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
            })
            .ok_or_else(|| {
                ControlError::invalid_params(
                    "ui.contributions.register",
                    format!("{property} state binding requires a bounded command_id"),
                )
            })?;
        let _ = command_id;
        if !object
            .get("state")
            .and_then(Value::as_str)
            .is_some_and(|state| matches!(state, "visible" | "enabled" | "checked"))
        {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                format!("{property} state binding state must be visible, enabled, or checked"),
            ));
        }
        if object
            .get("equals")
            .is_some_and(|value| !value.is_boolean())
        {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                format!("{property} state binding equals must be a boolean"),
            ));
        }
    }
    Ok(())
}

pub(super) fn component_ids(root: &Component) -> HashSet<String> {
    let mut ids = HashSet::new();
    fn collect(component: &Component, ids: &mut HashSet<String>) {
        ids.insert(component.id.clone());
        for child in &component.children {
            collect(child, ids);
        }
    }
    collect(root, &mut ids);
    ids
}

pub(super) fn patch_component_values(component: &mut Component, values: &HashMap<String, Value>) {
    if let Some(value) = values.get(&component.id) {
        component.value = value.clone();
    }
    for child in &mut component.children {
        patch_component_values(child, values);
    }
}

pub(super) fn sync_component_binding(
    component: &mut Component,
    native_state: &Value,
    layers: &[crate::control::LayerSnapshot],
) {
    for (property, binding) in &component.state_bindings {
        let command_id = binding.get("command_id").and_then(Value::as_str);
        let state = binding.get("state").and_then(Value::as_str);
        let expected = binding
            .get("equals")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let value = native_state
            .pointer("/shell/_command_surface/commands")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|command| command.get("id").and_then(Value::as_str) == command_id)
            .and_then(|command| state.and_then(|state| command.pointer(&format!("/state/{state}"))))
            .and_then(Value::as_bool)
            .is_some_and(|actual| actual == expected);
        match property.as_str() {
            "visible" => component.visible = value,
            "enabled" => component.enabled = value,
            _ => unreachable!("validated component state binding property"),
        }
    }
    let binding = component
        .action
        .as_ref()
        .filter(|action| action.get("type").and_then(Value::as_str) == Some("bind"));
    if let Some(binding) = binding {
        let target = binding.get("target").and_then(Value::as_str);
        let property = binding.get("property").and_then(Value::as_str);
        let value = match (target, property) {
            (Some("viewer.layers"), Some(property)) => binding
                .get("layer_id")
                .and_then(Value::as_str)
                .and_then(|id| layers.iter().find(|layer| layer.layer_id == id))
                .and_then(|layer| match property {
                    "visible" => Some(json!(layer.visible)),
                    "opacity" => Some(json!(layer.opacity)),
                    _ => None,
                }),
            (Some("viewer.channels"), Some("active")) => native_state
                .pointer("/channels/channels")
                .and_then(Value::as_array)
                .and_then(|channels| {
                    channels.iter().find(|channel| {
                        channel.get("selected").and_then(Value::as_bool) == Some(true)
                    })
                })
                .and_then(|channel| channel.get("name").cloned()),
            (Some("viewer.channels"), Some("visible")) => native_state
                .pointer("/channels/channels")
                .and_then(Value::as_array)
                .map(|channels| {
                    Value::Array(
                        channels
                            .iter()
                            .filter(|channel| {
                                channel.get("visible").and_then(Value::as_bool) == Some(true)
                            })
                            .filter_map(|channel| channel.get("name").cloned())
                            .collect(),
                    )
                }),
            (Some("viewer.camera"), Some("zoom")) => native_state
                .pointer("/camera/camera/zoom_screen_per_lvl0_px")
                .cloned(),
            (Some("viewer"), Some("smooth_pixels")) => native_state
                .pointer("/smooth/smooth_pixels/smooth")
                .cloned(),
            _ => None,
        };
        if let Some(value) = value {
            component.value = value;
        }
    }
    for child in &mut component.children {
        sync_component_binding(child, native_state, layers);
    }
}

pub(super) fn ensure_contribution_capabilities(
    root: &Component,
    extension: &ExtensionSnapshot,
) -> Result<(), ControlError> {
    let granted = &extension.granted_capabilities;
    if !granted.iter().any(|capability| capability == "ui.panels") {
        return Err(ControlError::new(
            ControlErrorKind::PermissionDenied,
            "the extension was not granted ui.panels",
        ));
    }
    fn check(component: &Component, granted: &[String]) -> Result<(), ControlError> {
        if let Some(action) = component.action.as_ref() {
            let kind = action.get("type").and_then(Value::as_str).ok_or_else(|| {
                ControlError::invalid_params(
                    "ui.contributions.register",
                    "component action.type is required",
                )
            })?;
            match kind {
                "emit" => {
                    if !action
                        .get("event")
                        .and_then(Value::as_str)
                        .is_some_and(|event| !event.trim().is_empty())
                    {
                        return Err(ControlError::invalid_params(
                            "ui.contributions.register",
                            "emit actions require a non-empty event",
                        ));
                    }
                }
                "command" => {
                    let method = action
                        .get("method")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            ControlError::invalid_params(
                                "ui.contributions.register",
                                "command actions require a method",
                            )
                        })?;
                    if crate::control::registry::method(method).is_none() {
                        return Err(ControlError::invalid_params(
                            "ui.contributions.register",
                            format!("unknown native command '{method}'"),
                        ));
                    }
                    if !granted
                        .iter()
                        .any(|capability| capability == "viewer.write")
                    {
                        return Err(ControlError::new(
                            ControlErrorKind::PermissionDenied,
                            "native command actions require viewer.write",
                        ));
                    }
                }
                "bind" => {
                    let target = action.get("target").and_then(Value::as_str);
                    if !matches!(
                        target,
                        Some("viewer.layers" | "viewer.channels" | "viewer.camera" | "viewer")
                    ) {
                        return Err(ControlError::new(
                            ControlErrorKind::Unsupported,
                            "unsupported native binding target",
                        ));
                    }
                    let property = action.get("property").and_then(Value::as_str);
                    let property_supported = match target {
                        Some("viewer.layers") => {
                            matches!(property, Some("opacity" | "visible"))
                                && action
                                    .get("layer_id")
                                    .and_then(Value::as_str)
                                    .is_some_and(|id| !id.trim().is_empty())
                        }
                        Some("viewer.channels") => matches!(property, Some("active" | "visible")),
                        Some("viewer.camera") => property == Some("zoom"),
                        Some("viewer") => property == Some("smooth_pixels"),
                        _ => false,
                    };
                    if !property_supported {
                        return Err(ControlError::new(
                            ControlErrorKind::Unsupported,
                            "unsupported native binding property or missing layer_id",
                        ));
                    }
                    let capability = if target == Some("viewer.layers") {
                        "viewer.layers.write"
                    } else {
                        "viewer.write"
                    };
                    if !granted.iter().any(|granted| granted == capability) {
                        return Err(ControlError::new(
                            ControlErrorKind::PermissionDenied,
                            format!("native bindings to this target require {capability}"),
                        ));
                    }
                }
                _ => {
                    return Err(ControlError::invalid_params(
                        "ui.contributions.register",
                        "unsupported component action type",
                    ));
                }
            }
        }
        for child in &component.children {
            check(child, granted)?;
        }
        Ok(())
    }
    check(root, granted)
}
