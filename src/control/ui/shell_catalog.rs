use std::collections::{BTreeSet, HashMap};

use serde_json::{Value, json};

use super::*;

impl UiRegistry {
    pub fn shell_component_descriptors(&self, mode: Option<&str>) -> Vec<Value> {
        let state = self.state.lock().expect("UI registry poisoned");
        let mut descriptors = state
            .contributions
            .iter()
            .filter_map(|contribution| {
                let modes = contribution_modes(&contribution.location);
                if mode.is_some_and(|mode| !modes.contains(&mode)) {
                    return None;
                }
                let extension = state.extensions.get(&contribution.extension_id);
                let owner_session_id =
                    extension.map(|extension| extension.owner_session_id.clone());
                Some(json!({
                    "id":contribution.shell_mount,
                    "version":1,
                    "title":contribution.root.title.clone()
                        .or_else(|| contribution.root.label.clone())
                        .unwrap_or_else(|| contribution.extension_id.clone()),
                    "kind":contribution_kind(&contribution.location),
                    "modes":modes,
                    "readiness":[format!("extension_{}", contribution.readiness)],
                    "legal_parent_types":contribution_legal_parent_types(&contribution.location),
                    "singleton":true,
                    "configuration_schema":{
                        "$schema":"https://json-schema.org/draft/2020-12/schema",
                        "type":"object",
                        "properties":{},
                        "additionalProperties":false
                    },
                    "commands":[],
                    "events":[format!("ui.extension:{}.*", contribution.extension_id)],
                    "minimum_size":{"width":120.0,"height":40.0},
                    "recommended_size":{"width":320.0,"height":240.0},
                    "persistence":"session",
                    "ownership":{
                        "scope":"extension",
                        "owner_id":contribution.extension_id,
                        "owner_session_id":owner_session_id,
                        "protected":false,
                    },
                }))
            })
            .collect::<Vec<_>>();
        descriptors.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
        descriptors
    }

    pub fn validate_shell_layout_access(
        &self,
        desired_tree: &Value,
        session_id: &str,
    ) -> Result<(), ControlError> {
        let state = self.state.lock().expect("UI registry poisoned");
        let application_controller = state
            .session_capabilities
            .get(session_id)
            .is_some_and(|capabilities| capabilities.contains("ui.shell.application_control"));
        validate_shell_layout_nodes(
            &state,
            desired_tree,
            (!application_controller).then_some(session_id),
            None,
        )
    }

    pub fn validate_extension_shell_layout_access(
        &self,
        desired_tree: &Value,
        session_id: &str,
        extension_id: &str,
    ) -> Result<(), ControlError> {
        let state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(extension_id)
            .ok_or_else(|| not_found("extension", extension_id))?;
        ensure_owner(extension, session_id)?;
        ensure_session_capability(
            &state,
            session_id,
            "ui.shell.extension_place",
            "ui.extensions.layouts.register",
        )?;
        validate_shell_layout_nodes(&state, desired_tree, Some(session_id), Some(extension_id))
    }

    pub fn validate_shell_mutation_access(
        &self,
        method: &str,
        params: &Value,
        current_shell: &Value,
        candidate_layout: Option<&Value>,
        session_id: &str,
    ) -> Result<(), ControlError> {
        if session_id == "native-ui" {
            return Ok(());
        }
        let state = self.state.lock().expect("UI registry poisoned");
        let owned_extensions = state
            .extensions
            .values()
            .filter(|extension| extension.owner_session_id == session_id)
            .map(|extension| extension.id.as_str())
            .collect::<BTreeSet<_>>();
        let granted_capabilities = state
            .session_capabilities
            .get(session_id)
            .cloned()
            .unwrap_or_default();
        let application_controller = granted_capabilities.contains("ui.shell.application_control");
        let owner_guard = (!application_controller).then_some(session_id);
        if let Some(candidate) = candidate_layout {
            validate_shell_layout_nodes(&state, candidate, owner_guard, None)?;
        }
        if application_controller
            || (method == "ui.shell.recover" && granted_capabilities.contains("ui.shell.recovery"))
        {
            return Ok(());
        }
        let current_nodes = current_shell
            .pointer("/layout/nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::Internal,
                    "current shell snapshot has no desired layout nodes",
                )
            })?;
        if method == "ui.shell.patch_layout" {
            for field in [
                "visibility",
                "selected",
                "sizes",
                "splits",
                "collapsed",
                "configurations",
            ] {
                for node_id in params
                    .get(field)
                    .and_then(Value::as_object)
                    .into_iter()
                    .flat_map(|values| values.keys())
                {
                    if let Some(node) = current_nodes
                        .iter()
                        .find(|node| node.get("id").and_then(Value::as_str) == Some(node_id))
                    {
                        ensure_shell_node_owner(
                            node,
                            &owned_extensions,
                            &granted_capabilities,
                            method,
                        )?;
                    }
                }
            }
            for field in ["active_region_id", "focused_node_id"] {
                if let Some(node_id) = params.get(field).and_then(Value::as_str)
                    && let Some(node) = current_nodes
                        .iter()
                        .find(|node| node.get("id").and_then(Value::as_str) == Some(node_id))
                {
                    ensure_shell_node_owner(
                        node,
                        &owned_extensions,
                        &granted_capabilities,
                        method,
                    )?;
                }
            }
            if params
                .get("clear_focus")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && let Some(node_id) = current_shell.get("focused_node_id").and_then(Value::as_str)
                && let Some(node) = current_nodes
                    .iter()
                    .find(|node| node.get("id").and_then(Value::as_str) == Some(node_id))
            {
                ensure_shell_node_owner(node, &owned_extensions, &granted_capabilities, method)?;
            }
            return Ok(());
        }
        if method == "ui.shell.patch" {
            let legacy_nodes = current_shell
                .get("nodes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten();
            let targeted = ["visibility", "orders", "selected"]
                .into_iter()
                .flat_map(|field| {
                    params
                        .get(field)
                        .and_then(Value::as_object)
                        .into_iter()
                        .flat_map(|values| values.keys())
                })
                .collect::<BTreeSet<_>>();
            for node_id in targeted {
                if let Some(node) = legacy_nodes
                    .clone()
                    .find(|node| node.get("id").and_then(Value::as_str) == Some(node_id))
                {
                    ensure_shell_node_owner(
                        node,
                        &owned_extensions,
                        &granted_capabilities,
                        method,
                    )?;
                }
            }
            return Ok(());
        }
        if matches!(
            method,
            "ui.shell.replace_layout"
                | "ui.shell.import_layout"
                | "ui.shell.reset"
                | "ui.shell.recover"
                | "ui.shell.profiles.load"
        ) {
            for node in current_nodes {
                ensure_shell_node_owner(node, &owned_extensions, &granted_capabilities, method)?;
            }
        }
        Ok(())
    }

    pub fn annotate_shell_snapshot_ownership(&self, snapshot: &mut Value) {
        let state = self.state.lock().expect("UI registry poisoned");
        let owners = state
            .contributions
            .iter()
            .filter_map(|contribution| {
                let extension = state.extensions.get(&contribution.extension_id)?;
                Some((
                    contribution.shell_mount.as_str(),
                    (
                        contribution.extension_id.as_str(),
                        extension.owner_session_id.as_str(),
                        contribution.readiness.as_str(),
                        extension.readiness_reason.as_deref(),
                        contribution.extension_version.as_str(),
                        extension.version.as_str(),
                    ),
                ))
            })
            .collect::<HashMap<_, _>>();
        let Some(nodes) = snapshot
            .pointer_mut("/layout/nodes")
            .and_then(Value::as_array_mut)
        else {
            return;
        };
        for node in nodes {
            let Some(mount) = node.get("mount").and_then(Value::as_str) else {
                continue;
            };
            let extension_mount =
                node.get("type").and_then(Value::as_str) == Some("extension_mount");
            let readiness = owners.get(mount).copied();
            let object = node
                .as_object_mut()
                .expect("shell layout nodes are objects");
            if let Some((
                extension_id,
                owner_session_id,
                readiness,
                reason,
                expected_version,
                current_version,
            )) = readiness
            {
                object.insert(
                    "ownership".to_string(),
                    json!({
                        "scope":"extension",
                        "owner_id":extension_id,
                        "owner_session_id":owner_session_id,
                        "protected":false,
                    }),
                );
                object.insert(
                    "readiness".to_string(),
                    json!({
                        "state":readiness,
                        "reason":reason,
                        "expected_extension_version":expected_version,
                        "current_extension_version":current_version,
                    }),
                );
            } else if extension_mount {
                object.insert(
                    "readiness".to_string(),
                    json!({
                        "state":"missing",
                        "reason":"the retained extension contribution is not registered",
                        "expected_extension_version":Value::Null,
                        "current_extension_version":Value::Null,
                    }),
                );
            }
        }
    }
}

fn validate_shell_layout_nodes(
    state: &State,
    desired_tree: &Value,
    owner_session_id: Option<&str>,
    required_extension_id: Option<&str>,
) -> Result<(), ControlError> {
    let nodes = desired_tree
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ControlError::invalid_params(
                "ui.shell.replace_layout",
                "desired_tree.nodes must be an array",
            )
        })?;
    let mode = nodes
        .iter()
        .find_map(|node| match node.get("mount").and_then(Value::as_str) {
            Some("builtin:project-workspace") => Some("project"),
            Some("builtin:viewer-canvas") => Some("single"),
            Some("builtin:mosaic-canvas") => Some("mosaic"),
            _ => None,
        });
    let node_types = nodes
        .iter()
        .filter_map(|node| {
            Some((
                node.get("id")?.as_str()?.to_string(),
                node.get("type")?.as_str()?.to_string(),
            ))
        })
        .collect::<HashMap<_, _>>();
    let mut derived_parents = HashMap::new();
    for parent in nodes {
        let Some(parent_id) = parent.get("id").and_then(Value::as_str) else {
            continue;
        };
        for child in parent
            .get("children")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            derived_parents.insert(child.to_string(), parent_id.to_string());
        }
    }
    for node in nodes.iter().filter(|node| {
        node.get("type").and_then(Value::as_str) == Some("extension_mount")
            && node
                .get("mount")
                .and_then(Value::as_str)
                .is_some_and(|mount| mount.contains('/'))
    }) {
        let mount = node.get("mount").and_then(Value::as_str).unwrap();
        let contribution = state
            .contributions
            .iter()
            .find(|contribution| contribution.shell_mount == mount)
            .ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::ResourceNotFound,
                    format!("extension shell mount '{mount}' is not registered"),
                )
            })?;
        let extension = state
            .extensions
            .get(&contribution.extension_id)
            .ok_or_else(|| not_found("extension", &contribution.extension_id))?;
        if owner_session_id
            .is_some_and(|owner_session_id| extension.owner_session_id != owner_session_id)
        {
            return Err(shell_ownership_error(
                node,
                &contribution.extension_id,
                "ui.shell.replace_layout",
            ));
        }
        if required_extension_id.is_some_and(|required| required != contribution.extension_id) {
            return Err(shell_ownership_error(
                node,
                &contribution.extension_id,
                "ui.extensions.layouts.register",
            ));
        }
        if let Some(mode) = mode
            && !contribution_modes(&contribution.location).contains(&mode)
        {
            return Err(ControlError::invalid_params(
                "ui.shell.replace_layout",
                format!("extension shell mount '{mount}' is not available in {mode} mode"),
            ));
        }
        let node_id = node.get("id").and_then(Value::as_str).unwrap_or_default();
        let parent_id = node
            .get("parent_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| derived_parents.get(node_id).cloned());
        let parent_kind = parent_id
            .as_deref()
            .and_then(|parent| node_types.get(parent))
            .map(String::as_str)
            .unwrap_or_default();
        if !contribution_legal_parent_types(&contribution.location).contains(&parent_kind) {
            return Err(ControlError::invalid_params(
                "ui.shell.replace_layout",
                format!(
                    "extension shell mount '{mount}' cannot be placed in parent type '{parent_kind}'"
                ),
            ));
        }
    }
    Ok(())
}

fn ensure_shell_node_owner(
    node: &Value,
    owned_extensions: &BTreeSet<&str>,
    granted_capabilities: &BTreeSet<String>,
    method: &str,
) -> Result<(), ControlError> {
    if node.get("type").and_then(Value::as_str) != Some("extension_mount") {
        let required = shell_application_required_capability(node, method);
        if granted_capabilities.contains(required)
            || granted_capabilities.contains("ui.shell.application_control")
        {
            return Ok(());
        }
        return Err(shell_application_ownership_error(node, method));
    }
    let Some(mount) = node.get("mount").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(extension_id) = mount
        .strip_prefix("extension:")
        .and_then(|value| value.split('/').next())
    else {
        return Ok(());
    };
    if owned_extensions.contains(extension_id)
        && granted_capabilities.contains("ui.shell.extension_place")
    {
        Ok(())
    } else if owned_extensions.contains(extension_id) {
        Err(shell_session_capability_error(
            node,
            method,
            "ui.shell.extension_place",
        ))
    } else if granted_capabilities.contains("ui.shell.application_control") {
        Ok(())
    } else {
        Err(shell_ownership_error(node, extension_id, method))
    }
}

fn ensure_session_capability(
    state: &State,
    session_id: &str,
    capability: &str,
    method: &str,
) -> Result<(), ControlError> {
    if state
        .session_capabilities
        .get(session_id)
        .is_some_and(|capabilities| capabilities.contains(capability))
    {
        return Ok(());
    }
    Err(ControlError::new(
        ControlErrorKind::PermissionDenied,
        format!("{method} requires the '{capability}' session capability"),
    )
    .with_data(json!({
        "method":method,
        "required_capability":capability,
        "resolution":"request the capability during system.hello",
    })))
}

fn shell_session_capability_error(node: &Value, method: &str, capability: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::PermissionDenied,
        format!("{method} requires the '{capability}' session capability"),
    )
    .with_data(json!({
        "method":method,
        "node_id":node.get("id").cloned().unwrap_or(Value::Null),
        "mount":node.get("mount").cloned().unwrap_or(Value::Null),
        "required_capability":capability,
        "resolution":"request the capability during system.hello",
    }))
}

fn shell_application_ownership_error(node: &Value, method: &str) -> ControlError {
    let required_capability = shell_application_required_capability(node, method);
    ControlError::new(
        ControlErrorKind::PermissionDenied,
        format!("{method} cannot change an application-owned shell node"),
    )
    .with_data(json!({
        "method":method,
        "node_id":node.get("id").cloned().unwrap_or(Value::Null),
        "mount":node.get("mount").cloned().unwrap_or(Value::Null),
        "owner":{"scope":"application","owner_id":"odon"},
        "required_capability":required_capability,
        "resolution":"request an explicit application capability during system.hello; extensions may mutate only their own extension_mount nodes",
    }))
}

fn shell_application_required_capability(node: &Value, method: &str) -> &'static str {
    let mount = node.get("mount").and_then(Value::as_str);
    let kind = node.get("type").and_then(Value::as_str);
    if mount.is_some_and(|mount| mount.ends_with("-top-bar"))
        || mount.is_some_and(|mount| {
            matches!(
                mount,
                "builtin:extension-host.top-bar-actions"
                    | "builtin:extension-host.status-bar"
                    | "builtin:extension-host.canvas-controls"
            )
        })
        || matches!(kind, Some("toolbar" | "status_bar" | "menu_host"))
    {
        "ui.shell.chrome"
    } else if method == "ui.shell.recover" {
        "ui.shell.recovery"
    } else if node
        .pointer("/ownership/protected")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        "ui.shell.application_control"
    } else {
        "ui.shell.compose"
    }
}

fn shell_ownership_error(node: &Value, extension_id: &str, method: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::PermissionDenied,
        format!("{method} cannot change a shell node owned by extension '{extension_id}'"),
    )
    .with_data(json!({
        "method":method,
        "node_id":node.get("id").cloned().unwrap_or(Value::Null),
        "mount":node.get("mount").cloned().unwrap_or(Value::Null),
        "owner":{"scope":"extension","owner_id":extension_id},
        "required_capability":"ui.shell.application_control",
        "resolution":"use an application-controller session or ask the owning extension to mutate the node",
    }))
}
