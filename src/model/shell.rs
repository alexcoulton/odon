//! Actor-owned application-shell composition.
//!
//! The shell model owns stable built-in node identity, visibility for structural chrome, and
//! child ordering. Existing viewer/mosaic panel visibility and selected-tab fields remain the
//! canonical domain values during the first shell milestone; `AppModel` overlays them when it
//! builds a snapshot and applies shell mutations through the same fields.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::control::{ControlError, ControlErrorKind};

use super::app::ModelMode;

mod document;
mod layout;

use layout::{
    MAX_CONFIGURATION_BYTES_PER_NODE, MAX_CONFIGURATION_BYTES_TOTAL, MAX_CONFIGURATION_DEPTH,
    MAX_CONFIGURATION_VALUES_PER_NODE, ShellLayout, component_catalog, component_catalog_schema,
    component_minimum_size, default_layout, layout_schema,
};

use document::shell_document_schema;

const SHELL_SCHEMA_VERSION: u64 = 1;

#[derive(Debug, Clone)]
pub(crate) struct ShellModel {
    revision: u64,
    modes: BTreeMap<String, ShellMode>,
}

#[derive(Debug, Clone)]
struct ShellMode {
    root_id: String,
    nodes: BTreeMap<String, ShellNode>,
    layout: ShellLayout,
    active_region_id: String,
    focused_node_id: Option<String>,
}

#[derive(Debug, Clone)]
struct ShellNode {
    kind: &'static str,
    parent: Option<String>,
    content: Option<&'static str>,
    visible: bool,
    mutable_visibility: bool,
    mutable_order: bool,
    mutable_selection: bool,
    children: Vec<String>,
    selected: Option<String>,
}

impl Default for ShellModel {
    fn default() -> Self {
        Self {
            revision: 1,
            modes: BTreeMap::from([
                ("project".to_string(), project_shell()),
                ("single".to_string(), single_shell()),
                ("mosaic".to_string(), mosaic_shell()),
            ]),
        }
    }
}

impl ShellModel {
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn sync_overlay(
        &mut self,
        mode: ModelMode,
        visibility: &[(&str, bool)],
        selection: &[(&str, &str)],
        sync_desired_layout: bool,
    ) -> Result<(), ControlError> {
        let mode_name = editable_mode_name(mode)?;
        let shell = self
            .modes
            .get_mut(mode_name)
            .ok_or_else(|| internal(format!("shell mode '{mode_name}' is missing")))?;
        let mut changed = false;
        for (id, visible) in visibility {
            let node = shell.nodes.get_mut(*id).ok_or_else(|| {
                internal(format!(
                    "shell visibility overlay references unknown node '{id}'"
                ))
            })?;
            changed |= node.visible != *visible;
            node.visible = *visible;
            if sync_desired_layout {
                changed |= sync_layout_visibility(shell, id, *visible);
            }
        }
        for (id, selected) in selection {
            let node = shell.nodes.get_mut(*id).ok_or_else(|| {
                internal(format!(
                    "shell selection overlay references unknown node '{id}'"
                ))
            })?;
            changed |= node.selected.as_deref() != Some(*selected);
            node.selected = Some((*selected).to_string());
            if sync_desired_layout
                && let Some(mount) = shell.nodes.get(*selected).and_then(|node| node.content)
            {
                changed |= shell.layout.select_mount(mount);
            }
        }
        if changed {
            self.touch();
        }
        Ok(())
    }

    pub(crate) fn snapshot(
        &self,
        mode: ModelMode,
        visibility: &[(&str, bool)],
        selection: &[(&str, &str)],
    ) -> Result<Value, ControlError> {
        let mode_name = editable_mode_name(mode)?;
        let mut shell = self
            .modes
            .get(mode_name)
            .cloned()
            .ok_or_else(|| internal(format!("shell mode '{mode_name}' is missing")))?;
        for (node_id, visible) in visibility {
            let node = shell.nodes.get_mut(*node_id).ok_or_else(|| {
                internal(format!(
                    "shell visibility overlay references unknown node '{node_id}'"
                ))
            })?;
            node.visible = *visible;
        }
        for (container_id, selected_id) in selection {
            let node = shell.nodes.get_mut(*container_id).ok_or_else(|| {
                internal(format!(
                    "shell selection overlay references unknown node '{container_id}'"
                ))
            })?;
            node.selected = Some((*selected_id).to_string());
        }
        Ok(shell.to_json(mode_name, self.revision))
    }

    pub(crate) fn patch(&mut self, params: &Value, mode: ModelMode) -> Result<(), ControlError> {
        let mode_name = requested_mode(params, mode)?;
        if let Some(expected) = params.get("if_shell_revision") {
            let expected = expected.as_u64().ok_or_else(|| {
                invalid(
                    "ui.shell.patch",
                    "if_shell_revision must be an unsigned integer",
                )
            })?;
            if expected != self.revision {
                return Err(ControlError::new(
                    ControlErrorKind::Conflict,
                    format!(
                        "shell revision conflict: expected {expected}, current revision is {}",
                        self.revision
                    ),
                )
                .with_data(json!({
                    "expected_revision": expected,
                    "current_revision": self.revision,
                    "conflicting_domain":"application_shell",
                    "snapshot_method":"ui.shell.get",
                    "retry_strategy":"refetch_merge_retry",
                })));
            }
        }
        let current = self
            .modes
            .get(mode_name)
            .ok_or_else(|| internal(format!("shell mode '{mode_name}' is missing")))?;
        let mut candidate = current.clone();
        apply_visibility_patch(&mut candidate, params)?;
        apply_order_patch(&mut candidate, params)?;
        apply_selection_patch(&mut candidate, params)?;
        apply_layout_compatibility_patch(&mut candidate, params);
        candidate.validate("ui.shell.patch")?;
        if &candidate != current {
            self.modes.insert(mode_name.to_string(), candidate);
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        Ok(())
    }

    pub(crate) fn reset(
        &mut self,
        params: &Value,
        mode: ModelMode,
    ) -> Result<String, ControlError> {
        if let Some(expected) = params.get("if_shell_revision") {
            let expected = expected.as_u64().ok_or_else(|| {
                invalid(
                    "ui.shell.reset",
                    "if_shell_revision must be an unsigned integer",
                )
            })?;
            if expected != self.revision {
                return Err(ControlError::new(
                    ControlErrorKind::Conflict,
                    format!(
                        "shell revision conflict: expected {expected}, current revision is {}",
                        self.revision
                    ),
                )
                .with_data(json!({
                    "expected_revision": expected,
                    "current_revision": self.revision,
                    "conflicting_domain":"application_shell",
                    "snapshot_method":"ui.shell.get",
                    "retry_strategy":"refetch_merge_retry",
                })));
            }
        }
        let mode_name = requested_mode(params, mode)?;
        let replacement = default_mode(mode_name);
        let changed = self.modes.get(mode_name) != Some(&replacement);
        self.modes.insert(mode_name.to_string(), replacement);
        if changed {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        Ok(mode_name.to_string())
    }

    pub(crate) fn replace_layout(
        &mut self,
        params: &Value,
        mode: ModelMode,
    ) -> Result<(), ControlError> {
        let mode_name = requested_mode(params, mode)?;
        validate_revision_guard(params, self.revision, "ui.shell.replace_layout")?;
        let desired_tree = params
            .get("desired_tree")
            .ok_or_else(|| invalid("ui.shell.replace_layout", "desired_tree is required"))?;
        let candidate =
            ShellLayout::from_value(desired_tree, mode_name, "ui.shell.replace_layout")?;
        let shell = self
            .modes
            .get_mut(mode_name)
            .ok_or_else(|| internal(format!("shell mode '{mode_name}' is missing")))?;
        if shell.layout != candidate {
            shell.layout = candidate;
            shell.active_region_id = shell.layout.preferred_active_region_id().to_string();
            shell.focused_node_id = None;
            self.touch();
        }
        Ok(())
    }

    pub(crate) fn patch_layout(
        &mut self,
        params: &Value,
        mode: ModelMode,
    ) -> Result<(), ControlError> {
        let mode_name = requested_mode(params, mode)?;
        validate_revision_guard(params, self.revision, "ui.shell.patch_layout")?;
        let current = self
            .modes
            .get(mode_name)
            .ok_or_else(|| internal(format!("shell mode '{mode_name}' is missing")))?;
        let mut candidate = current.clone();
        candidate
            .layout
            .patch_state(params, mode_name, "ui.shell.patch_layout")?;
        candidate.patch_focus_state(params, "ui.shell.patch_layout")?;
        candidate.validate("ui.shell.patch_layout")?;
        if &candidate != current {
            self.modes.insert(mode_name.to_string(), candidate);
            self.touch();
        }
        Ok(())
    }

    pub(crate) fn mode_state(&self, mode_name: &str) -> Result<Value, ControlError> {
        let shell = self
            .modes
            .get(mode_name)
            .ok_or_else(|| internal(format!("shell mode '{mode_name}' is missing")))?;
        Ok(shell.to_json(mode_name, self.revision))
    }
}

impl PartialEq for ShellMode {
    fn eq(&self, other: &Self) -> bool {
        self.root_id == other.root_id
            && self.nodes == other.nodes
            && self.layout == other.layout
            && self.active_region_id == other.active_region_id
            && self.focused_node_id == other.focused_node_id
    }
}

impl PartialEq for ShellNode {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.parent == other.parent
            && self.content == other.content
            && self.visible == other.visible
            && self.mutable_visibility == other.mutable_visibility
            && self.mutable_order == other.mutable_order
            && self.mutable_selection == other.mutable_selection
            && self.children == other.children
            && self.selected == other.selected
    }
}

impl ShellMode {
    fn to_json(&self, mode: &str, revision: u64) -> Value {
        let nodes = self
            .nodes
            .iter()
            .map(|(id, node)| {
                let protected = id == &self.root_id
                    || node.content.is_some_and(|content| {
                        matches!(
                            content,
                            "builtin:project-workspace"
                                | "builtin:viewer-canvas"
                                | "builtin:mosaic-canvas"
                        )
                    });
                json!({
                    "id": id,
                    "type": node.kind,
                    "parent_id": node.parent,
                    "content": node.content,
                    "visible": node.visible,
                    "mutable": {
                        "visibility": node.mutable_visibility,
                        "order": node.mutable_order,
                        "selection": node.mutable_selection,
                    },
                    "children": node.children,
                    "selected_id": node.selected,
                    "ownership":{
                        "scope":"application",
                        "owner_id":"odon",
                        "protected":protected,
                    },
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version": SHELL_SCHEMA_VERSION,
            "revision": revision,
            "mode": mode,
            "root_id": self.root_id,
            "nodes": nodes,
            "layout": self.layout.to_snapshot_json(),
            "active_region_id":self.active_region_id,
            "focused_node_id":self.focused_node_id,
        })
    }

    fn patch_focus_state(&mut self, params: &Value, method: &str) -> Result<(), ControlError> {
        if let Some(active_region) = params.get("active_region_id") {
            let active_region = active_region
                .as_str()
                .ok_or_else(|| invalid(method, "active_region_id must be a layout node ID"))?;
            if !self.layout.contains_node(active_region) {
                return Err(invalid(
                    method,
                    format!("unknown active-region layout node '{active_region}'"),
                ));
            }
            self.active_region_id = active_region.to_string();
        }
        if let Some(focused) = params.get("focused_node_id") {
            let focused = focused
                .as_str()
                .ok_or_else(|| invalid(method, "focused_node_id must be a layout node ID"))?;
            if !self.layout.contains_node(focused) {
                return Err(invalid(
                    method,
                    format!("unknown focused layout node '{focused}'"),
                ));
            }
            self.focused_node_id = Some(focused.to_string());
        }
        if params
            .get("clear_focus")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            self.focused_node_id = None;
        }
        Ok(())
    }

    fn validate(&self, method: &str) -> Result<(), ControlError> {
        if !self.nodes.contains_key(&self.root_id) {
            return Err(invalid(
                method,
                "shell root_id does not reference a known node",
            ));
        }
        let mut referenced = BTreeSet::new();
        for (id, node) in &self.nodes {
            if let Some(parent) = node.parent.as_deref()
                && !self.nodes.contains_key(parent)
            {
                return Err(invalid(
                    method,
                    format!("shell node '{id}' references unknown parent '{parent}'"),
                ));
            }
            let expected = self
                .nodes
                .iter()
                .filter_map(|(candidate_id, candidate)| {
                    (candidate.parent.as_deref() == Some(id.as_str())).then_some(candidate_id)
                })
                .cloned()
                .collect::<BTreeSet<_>>();
            let actual = node.children.iter().cloned().collect::<BTreeSet<_>>();
            if actual.len() != node.children.len() || actual != expected {
                return Err(invalid(
                    method,
                    format!("children for shell node '{id}' must contain each direct child once"),
                ));
            }
            referenced.extend(node.children.iter().cloned());
            if let Some(selected) = node.selected.as_deref()
                && !node.children.iter().any(|child| child == selected)
            {
                return Err(invalid(
                    method,
                    format!("selected node '{selected}' is not a child of '{id}'"),
                ));
            }
        }
        if referenced.contains(&self.root_id) {
            return Err(invalid(method, "shell root cannot be a child"));
        }
        if !self.layout.contains_node(&self.active_region_id) {
            return Err(invalid(
                method,
                format!(
                    "active region '{}' is not a desired-layout node",
                    self.active_region_id
                ),
            ));
        }
        if let Some(focused) = self.focused_node_id.as_deref()
            && !self.layout.contains_node(focused)
        {
            return Err(invalid(
                method,
                format!("focused node '{focused}' is not a desired-layout node"),
            ));
        }
        Ok(())
    }
}

fn apply_visibility_patch(shell: &mut ShellMode, params: &Value) -> Result<(), ControlError> {
    let Some(visibility) = params.get("visibility") else {
        return Ok(());
    };
    let visibility = visibility.as_object().ok_or_else(|| {
        invalid(
            "ui.shell.patch",
            "visibility must be an object of node IDs to booleans",
        )
    })?;
    for (id, value) in visibility {
        let visible = value.as_bool().ok_or_else(|| {
            invalid(
                "ui.shell.patch",
                format!("visibility for '{id}' must be a boolean"),
            )
        })?;
        let node = shell
            .nodes
            .get_mut(id)
            .ok_or_else(|| invalid("ui.shell.patch", format!("unknown shell node '{id}'")))?;
        if !node.mutable_visibility {
            return Err(invalid(
                "ui.shell.patch",
                format!("visibility for required shell node '{id}' cannot be changed"),
            ));
        }
        node.visible = visible;
    }
    Ok(())
}

fn apply_order_patch(shell: &mut ShellMode, params: &Value) -> Result<(), ControlError> {
    let Some(orders) = params.get("orders") else {
        return Ok(());
    };
    let orders = orders.as_object().ok_or_else(|| {
        invalid(
            "ui.shell.patch",
            "orders must be an object of parent IDs to child-ID arrays",
        )
    })?;
    for (id, value) in orders {
        let children = value.as_array().ok_or_else(|| {
            invalid(
                "ui.shell.patch",
                format!("order for '{id}' must be an array"),
            )
        })?;
        let children = children
            .iter()
            .map(|child| {
                child.as_str().map(str::to_string).ok_or_else(|| {
                    invalid(
                        "ui.shell.patch",
                        format!("order for '{id}' must contain strings"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let node = shell
            .nodes
            .get_mut(id)
            .ok_or_else(|| invalid("ui.shell.patch", format!("unknown shell container '{id}'")))?;
        if !node.mutable_order {
            return Err(invalid(
                "ui.shell.patch",
                format!("shell node '{id}' does not support child reordering"),
            ));
        }
        node.children = children;
    }
    Ok(())
}

fn apply_selection_patch(shell: &mut ShellMode, params: &Value) -> Result<(), ControlError> {
    let Some(selected) = params.get("selected") else {
        return Ok(());
    };
    let selected = selected.as_object().ok_or_else(|| {
        invalid(
            "ui.shell.patch",
            "selected must be an object of tab-container IDs to child IDs",
        )
    })?;
    for (id, value) in selected {
        let child = value.as_str().ok_or_else(|| {
            invalid(
                "ui.shell.patch",
                format!("selected value for '{id}' must be a string"),
            )
        })?;
        let node = shell
            .nodes
            .get_mut(id)
            .ok_or_else(|| invalid("ui.shell.patch", format!("unknown shell container '{id}'")))?;
        if !node.mutable_selection {
            return Err(invalid(
                "ui.shell.patch",
                format!("shell node '{id}' does not support selection"),
            ));
        }
        node.selected = Some(child.to_string());
    }
    Ok(())
}

fn apply_layout_compatibility_patch(shell: &mut ShellMode, params: &Value) {
    if let Some(visibility) = params.get("visibility").and_then(Value::as_object) {
        for (id, visible) in visibility {
            if let Some(visible) = visible.as_bool() {
                sync_layout_visibility(shell, id, visible);
            }
        }
    }
    if let Some(orders) = params.get("orders").and_then(Value::as_object) {
        for children in orders.values().filter_map(Value::as_array) {
            let mounts = children
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|child| shell.nodes.get(child).and_then(|node| node.content))
                .collect::<Vec<_>>();
            shell.layout.reorder_mounts(&mounts);
        }
    }
    if let Some(selected) = params.get("selected").and_then(Value::as_object) {
        for child in selected.values().filter_map(Value::as_str) {
            if let Some(mount) = shell.nodes.get(child).and_then(|node| node.content) {
                shell.layout.select_mount(mount);
            }
        }
    }
}

fn sync_layout_visibility(shell: &mut ShellMode, id: &str, visible: bool) -> bool {
    let mounts: Vec<&str> = match id {
        "builtin:single.left-panel" | "builtin:mosaic.left-panel" => {
            vec!["builtin:layers", "builtin:project"]
        }
        "builtin:single.right-panel" => vec![
            "builtin:properties",
            "builtin:views",
            "builtin:analysis",
            "builtin:measurements",
            "builtin:memory",
            "builtin:roi-selector",
        ],
        "builtin:mosaic.right-panel" => vec![
            "builtin:properties",
            "builtin:views",
            "builtin:mosaic-layout",
            "builtin:memory",
        ],
        _ => shell
            .nodes
            .get(id)
            .and_then(|node| node.content)
            .map(|mount| vec![mount])
            .unwrap_or_default(),
    };
    mounts.iter().fold(false, |changed, mount| {
        shell.layout.set_mount_visible(mount, visible) || changed
    })
}

fn requested_mode(params: &Value, current: ModelMode) -> Result<&str, ControlError> {
    if let Some(mode) = params.get("mode") {
        let mode = mode
            .as_str()
            .ok_or_else(|| invalid("ui.shell", "mode must be a string"))?;
        if matches!(mode, "project" | "single" | "mosaic") {
            return Ok(mode);
        }
        return Err(invalid(
            "ui.shell",
            "mode must be project, single, or mosaic",
        ));
    }
    editable_mode_name(current)
}

fn validate_revision_guard(
    params: &Value,
    revision: u64,
    method: &str,
) -> Result<(), ControlError> {
    let Some(expected) = params.get("if_shell_revision") else {
        return Ok(());
    };
    let expected = expected
        .as_u64()
        .ok_or_else(|| invalid(method, "if_shell_revision must be an unsigned integer"))?;
    if expected == revision {
        return Ok(());
    }
    Err(ControlError::new(
        ControlErrorKind::Conflict,
        format!("shell revision conflict: expected {expected}, current revision is {revision}"),
    )
    .with_data(json!({
        "expected_revision":expected,
        "current_revision":revision,
        "conflicting_domain":"application_shell",
        "snapshot_method":"ui.shell.get",
        "retry_strategy":"refetch_merge_retry",
    })))
}

fn editable_mode_name(mode: ModelMode) -> Result<&'static str, ControlError> {
    match mode {
        ModelMode::Project => Ok("project"),
        ModelMode::Single => Ok("single"),
        ModelMode::Mosaic => Ok("mosaic"),
        ModelMode::Transition => Err(ControlError::new(
            ControlErrorKind::NotReady,
            "shell mode must be explicit while Odon is transitioning",
        )),
    }
}

fn default_mode(mode: &str) -> ShellMode {
    match mode {
        "project" => project_shell(),
        "single" => single_shell(),
        "mosaic" => mosaic_shell(),
        _ => unreachable!("validated shell mode"),
    }
}

fn project_shell() -> ShellMode {
    shell_mode(
        "builtin:project.root",
        [
            node(
                "builtin:project.root",
                "application",
                None,
                None,
                true,
                false,
            ),
            node(
                "builtin:project.top-bar",
                "toolbar",
                Some("builtin:project.root"),
                Some("builtin:project-top-bar"),
                true,
                true,
            ),
            node(
                "builtin:project.workspace",
                "workspace",
                Some("builtin:project.root"),
                Some("builtin:project-workspace"),
                true,
                false,
            ),
            extension_host("project", "top-bar", "builtin:project.root"),
            extension_host("project", "status-bar", "builtin:project.root"),
            extension_host("project", "project-cards", "builtin:project.root"),
        ],
        [
            "builtin:project.top-bar",
            "extension:project.top-bar",
            "builtin:project.workspace",
            "extension:project.project-cards",
            "extension:project.status-bar",
        ],
    )
}

fn single_shell() -> ShellMode {
    shell_mode(
        "builtin:single.root",
        [
            node(
                "builtin:single.root",
                "application",
                None,
                None,
                true,
                false,
            ),
            node(
                "builtin:single.top-bar",
                "toolbar",
                Some("builtin:single.root"),
                Some("builtin:viewer-top-bar"),
                true,
                true,
            ),
            node(
                "builtin:single.left-panel",
                "panel",
                Some("builtin:single.root"),
                None,
                true,
                true,
            ),
            tabs(
                "builtin:single.left-tabs",
                "builtin:single.left-panel",
                "builtin:single.layers",
                ["builtin:single.layers", "builtin:single.project"],
            ),
            leaf(
                "builtin:single.layers",
                "builtin:single.left-tabs",
                "builtin:layers",
            ),
            leaf(
                "builtin:single.project",
                "builtin:single.left-tabs",
                "builtin:project",
            ),
            node(
                "builtin:single.canvas",
                "canvas_host",
                Some("builtin:single.root"),
                Some("builtin:viewer-canvas"),
                true,
                false,
            ),
            node(
                "builtin:single.right-panel",
                "panel",
                Some("builtin:single.root"),
                None,
                true,
                true,
            ),
            tabs(
                "builtin:single.right-tabs",
                "builtin:single.right-panel",
                "builtin:single.properties",
                [
                    "builtin:single.properties",
                    "builtin:single.views",
                    "builtin:single.analysis",
                    "builtin:single.measurements",
                    "builtin:single.memory",
                    "builtin:single.roi-selector",
                ],
            ),
            leaf(
                "builtin:single.properties",
                "builtin:single.right-tabs",
                "builtin:properties",
            ),
            leaf(
                "builtin:single.views",
                "builtin:single.right-tabs",
                "builtin:views",
            ),
            leaf(
                "builtin:single.analysis",
                "builtin:single.right-tabs",
                "builtin:analysis",
            ),
            leaf(
                "builtin:single.measurements",
                "builtin:single.right-tabs",
                "builtin:measurements",
            ),
            leaf(
                "builtin:single.memory",
                "builtin:single.right-tabs",
                "builtin:memory",
            ),
            leaf(
                "builtin:single.roi-selector",
                "builtin:single.right-tabs",
                "builtin:roi-selector",
            ),
            extension_host("single", "top-bar", "builtin:single.root"),
            extension_host("single", "status-bar", "builtin:single.root"),
            extension_host("single", "left-panel", "builtin:single.root"),
            extension_host("single", "right-panel", "builtin:single.root"),
            extension_host("single", "canvas-controls", "builtin:single.root"),
        ],
        [
            "builtin:single.top-bar",
            "extension:single.top-bar",
            "builtin:single.left-panel",
            "extension:single.left-panel",
            "builtin:single.canvas",
            "extension:single.canvas-controls",
            "builtin:single.right-panel",
            "extension:single.right-panel",
            "extension:single.status-bar",
        ],
    )
}

fn mosaic_shell() -> ShellMode {
    shell_mode(
        "builtin:mosaic.root",
        [
            node(
                "builtin:mosaic.root",
                "application",
                None,
                None,
                true,
                false,
            ),
            node(
                "builtin:mosaic.top-bar",
                "toolbar",
                Some("builtin:mosaic.root"),
                Some("builtin:mosaic-top-bar"),
                true,
                true,
            ),
            node(
                "builtin:mosaic.left-panel",
                "panel",
                Some("builtin:mosaic.root"),
                None,
                true,
                true,
            ),
            tabs(
                "builtin:mosaic.left-tabs",
                "builtin:mosaic.left-panel",
                "builtin:mosaic.layers",
                ["builtin:mosaic.layers", "builtin:mosaic.project"],
            ),
            leaf(
                "builtin:mosaic.layers",
                "builtin:mosaic.left-tabs",
                "builtin:layers",
            ),
            leaf(
                "builtin:mosaic.project",
                "builtin:mosaic.left-tabs",
                "builtin:project",
            ),
            node(
                "builtin:mosaic.canvas",
                "canvas_host",
                Some("builtin:mosaic.root"),
                Some("builtin:mosaic-canvas"),
                true,
                false,
            ),
            node(
                "builtin:mosaic.right-panel",
                "panel",
                Some("builtin:mosaic.root"),
                None,
                true,
                true,
            ),
            tabs(
                "builtin:mosaic.right-tabs",
                "builtin:mosaic.right-panel",
                "builtin:mosaic.properties",
                [
                    "builtin:mosaic.properties",
                    "builtin:mosaic.views",
                    "builtin:mosaic.layout",
                    "builtin:mosaic.memory",
                ],
            ),
            leaf(
                "builtin:mosaic.properties",
                "builtin:mosaic.right-tabs",
                "builtin:properties",
            ),
            leaf(
                "builtin:mosaic.views",
                "builtin:mosaic.right-tabs",
                "builtin:views",
            ),
            leaf(
                "builtin:mosaic.layout",
                "builtin:mosaic.right-tabs",
                "builtin:mosaic-layout",
            ),
            leaf(
                "builtin:mosaic.memory",
                "builtin:mosaic.right-tabs",
                "builtin:memory",
            ),
            extension_host("mosaic", "top-bar", "builtin:mosaic.root"),
            extension_host("mosaic", "status-bar", "builtin:mosaic.root"),
            extension_host("mosaic", "left-panel", "builtin:mosaic.root"),
            extension_host("mosaic", "right-panel", "builtin:mosaic.root"),
            extension_host("mosaic", "canvas-controls", "builtin:mosaic.root"),
        ],
        [
            "builtin:mosaic.top-bar",
            "extension:mosaic.top-bar",
            "builtin:mosaic.left-panel",
            "extension:mosaic.left-panel",
            "builtin:mosaic.canvas",
            "extension:mosaic.canvas-controls",
            "builtin:mosaic.right-panel",
            "extension:mosaic.right-panel",
            "extension:mosaic.status-bar",
        ],
    )
}

fn shell_mode<const N: usize, const R: usize>(
    root_id: &str,
    nodes: [(&str, ShellNode); N],
    root_children: [&str; R],
) -> ShellMode {
    let mut nodes = nodes
        .into_iter()
        .map(|(id, node)| (id.to_string(), node))
        .collect::<BTreeMap<_, _>>();
    nodes
        .get_mut(root_id)
        .expect("shell root is declared")
        .children = root_children.into_iter().map(str::to_string).collect();
    let mut children_by_parent = BTreeMap::<String, Vec<String>>::new();
    for (id, node) in &nodes {
        if let Some(parent) = node.parent.as_ref() {
            children_by_parent
                .entry(parent.clone())
                .or_default()
                .push(id.clone());
        }
    }
    for (parent, children) in children_by_parent {
        let node = nodes
            .get_mut(&parent)
            .expect("shell parent references a declared node");
        if node.children.is_empty() {
            node.children = children;
        }
    }
    let layout = default_layout(
        root_id
            .split('.')
            .next()
            .and_then(|prefix| prefix.strip_prefix("builtin:"))
            .expect("mode root ID"),
    );
    let active_region_id = layout.preferred_active_region_id().to_string();
    let shell = ShellMode {
        root_id: root_id.to_string(),
        nodes,
        layout,
        active_region_id,
        focused_node_id: None,
    };
    shell
        .validate("default shell")
        .expect("valid default shell");
    shell
}

fn node<'a>(
    id: &'a str,
    kind: &'static str,
    parent: Option<&str>,
    content: Option<&'static str>,
    visible: bool,
    mutable_visibility: bool,
) -> (&'a str, ShellNode) {
    (
        id,
        ShellNode {
            kind,
            parent: parent.map(str::to_string),
            content,
            visible,
            mutable_visibility,
            mutable_order: false,
            mutable_selection: false,
            children: Vec::new(),
            selected: None,
        },
    )
}

fn tabs<'a, const N: usize>(
    id: &'a str,
    parent: &str,
    selected: &str,
    children: [&str; N],
) -> (&'a str, ShellNode) {
    let (id, mut node) = node(id, "tabs", Some(parent), None, true, false);
    node.mutable_order = true;
    node.mutable_selection = true;
    node.children = children.into_iter().map(str::to_string).collect();
    node.selected = Some(selected.to_string());
    (id, node)
}

fn leaf<'a>(id: &'a str, parent: &str, content: &'static str) -> (&'a str, ShellNode) {
    node(id, "builtin", Some(parent), Some(content), true, false)
}

fn extension_host(mode: &str, name: &str, parent: &str) -> (&'static str, ShellNode) {
    let id = match (mode, name) {
        ("project", "top-bar") => "extension:project.top-bar",
        ("project", "status-bar") => "extension:project.status-bar",
        ("project", "project-cards") => "extension:project.project-cards",
        ("single", "top-bar") => "extension:single.top-bar",
        ("single", "status-bar") => "extension:single.status-bar",
        ("single", "left-panel") => "extension:single.left-panel",
        ("single", "right-panel") => "extension:single.right-panel",
        ("single", "canvas-controls") => "extension:single.canvas-controls",
        ("mosaic", "top-bar") => "extension:mosaic.top-bar",
        ("mosaic", "status-bar") => "extension:mosaic.status-bar",
        ("mosaic", "left-panel") => "extension:mosaic.left-panel",
        ("mosaic", "right-panel") => "extension:mosaic.right-panel",
        ("mosaic", "canvas-controls") => "extension:mosaic.canvas-controls",
        _ => unreachable!("known extension shell host"),
    };
    let content = match name {
        "top-bar" => "builtin:extension-host.top-bar-actions",
        "status-bar" => "builtin:extension-host.status-bar",
        "left-panel" => "builtin:extension-host.left-sections",
        "right-panel" => "builtin:extension-host.right-tabs",
        "canvas-controls" => "builtin:extension-host.canvas-controls",
        "project-cards" => "builtin:extension-host.project-cards",
        _ => unreachable!("known extension shell host"),
    };
    node(
        id,
        "extension_host",
        Some(parent),
        Some(content),
        true,
        true,
    )
}

fn invalid(method: &str, message: impl Into<String>) -> ControlError {
    ControlError::invalid_params(method, message)
}

fn internal(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::Internal, message)
}

pub(crate) fn visibility_patch(params: &Value) -> Option<&Map<String, Value>> {
    params.get("visibility").and_then(Value::as_object)
}

pub(crate) fn selection_patch(params: &Value) -> Option<&Map<String, Value>> {
    params.get("selected").and_then(Value::as_object)
}

pub(crate) fn shell_schema() -> Value {
    let mode = json!({"type":"string","enum":["project","single","mosaic"]});
    let node_id = json!({"type":"string","minLength":1,"maxLength":256});
    let mutability = json!({
        "type":"object",
        "properties":{
            "visibility":{"type":"boolean"},
            "order":{"type":"boolean"},
            "selection":{"type":"boolean"}
        },
        "required":["visibility","order","selection"],
        "additionalProperties":false
    });
    let ownership = json!({
        "type":"object",
        "properties":{
            "scope":{"type":"string","enum":["application","extension"]},
            "owner_id":{"type":"string","minLength":1},
            "owner_session_id":{"type":"string","minLength":1},
            "protected":{"type":"boolean"}
        },
        "required":["scope","owner_id","protected"],
        "additionalProperties":false
    });
    let node = json!({
        "type":"object",
        "properties":{
            "id":node_id,
            "type":{"type":"string","enum":[
                "application","toolbar","workspace","panel","tabs","builtin",
                "canvas_host","extension_host"
            ]},
            "parent_id":{"type":["string","null"]},
            "content":{"type":["string","null"]},
            "visible":{"type":"boolean"},
            "mutable":mutability,
            "children":{"type":"array","items":{"type":"string","minLength":1},"uniqueItems":true},
            "selected_id":{"type":["string","null"]}
            ,"ownership":ownership
        },
        "required":[
            "id","type","parent_id","content","visible","mutable","children","selected_id","ownership"
        ],
        "additionalProperties":false
    });
    let property_change = json!({
        "type":"object",
        "properties":{
            "node_id":{"type":"string","minLength":1},
            "property":{"type":"string","enum":[
                "visibility","order","selection","size","split","collapse","layout",
                "configuration","active_region","focus"
            ]},
            "before":{},
            "after":{}
        },
        "required":["node_id","property","before","after"],
        "additionalProperties":false
    });
    let change = json!({
        "type":"object",
        "properties":{
            "operation":{"type":"string","enum":["patch","reset","native_sync","replace_layout","patch_layout","import_layout","recover","load_profile"]},
            "mode":mode,
            "previous_revision":{"type":"integer","minimum":1},
            "revision":{"type":"integer","minimum":1},
            "changed":{"type":"boolean"},
            "changes":{"type":"array","items":property_change},
            "transaction_id":{"type":["string","null"],"minLength":1,"maxLength":128}
        },
        "required":[
            "operation","mode","previous_revision","revision","changed","changes","transaction_id"
        ],
        "additionalProperties":false
    });
    let snapshot = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-snapshot-v1.json",
        "title":"Odon application shell snapshot",
        "type":"object",
        "properties":{
            "schema_version":{"const":SHELL_SCHEMA_VERSION},
            "revision":{"type":"integer","minimum":1},
            "mode":mode,
            "root_id":{"type":"string","minLength":1},
            "nodes":{"type":"array","items":node,"uniqueItems":true},
            "layout":layout_schema(),
            "active_region_id":{"type":"string","minLength":1,"maxLength":256},
            "focused_node_id":{"type":["string","null"],"minLength":1,"maxLength":256},
            "change":change,
            "import":{
                "type":"object",
                "properties":{
                    "mode":mode,
                    "source_schema_version":{"type":"integer","minimum":0},
                    "schema_version":{"const":1},
                    "migrated":{"type":"boolean"}
                },
                "required":["mode","source_schema_version","schema_version","migrated"],
                "additionalProperties":false
            },
            "recovery":{
                "type":"object",
                "properties":{"protected":{"const":true},"mode":mode},
                "required":["protected","mode"],
                "additionalProperties":false
            },
            "profile":{
                "type":"object",
                "properties":{
                    "name":{"type":"string","minLength":1,"maxLength":128},
                    "scope":{"type":"string","enum":["session","application","project"]}
                },
                "required":["name","scope"],
                "additionalProperties":false
            },
            "_control":{"type":"object"}
        },
        "required":["schema_version","revision","mode","root_id","nodes","layout","active_region_id","focused_node_id"],
        "additionalProperties":false
    });
    let patch = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-patch-v1.json",
        "title":"Odon application shell patch",
        "type":"object",
        "properties":{
            "mode":mode,
            "if_shell_revision":{"type":"integer","minimum":1},
            "if_revision":{"type":"integer","minimum":0},
            "transaction_id":{"type":"string","minLength":1,"maxLength":128},
            "visibility":{
                "type":"object",
                "propertyNames":{"type":"string","minLength":1},
                "additionalProperties":{"type":"boolean"}
            },
            "orders":{
                "type":"object",
                "propertyNames":{"type":"string","minLength":1},
                "additionalProperties":{
                    "type":"array","items":{"type":"string","minLength":1},"uniqueItems":true
                }
            },
            "selected":{
                "type":"object",
                "propertyNames":{"type":"string","minLength":1},
                "additionalProperties":{"type":"string","minLength":1}
            }
        },
        "additionalProperties":false
    });
    let get = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-get-v1.json",
        "type":"object",
        "properties":{"mode":mode},
        "additionalProperties":false
    });
    let reset = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-reset-v1.json",
        "type":"object",
        "properties":{
            "mode":mode,
            "if_shell_revision":{"type":"integer","minimum":1},
            "if_revision":{"type":"integer","minimum":0},
            "transaction_id":{"type":"string","minLength":1,"maxLength":128}
        },
        "additionalProperties":false
    });
    let replace_layout = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-replace-layout-v1.json",
        "type":"object",
        "properties":{
            "mode":mode,
            "if_shell_revision":{"type":"integer","minimum":1},
            "if_revision":{"type":"integer","minimum":0},
            "transaction_id":{"type":"string","minLength":1,"maxLength":128},
            "desired_tree":layout_schema()
        },
        "required":["desired_tree"],
        "additionalProperties":false
    });
    let patch_layout = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-patch-layout-v1.json",
        "type":"object",
        "properties":{
            "mode":mode,
            "if_shell_revision":{"type":"integer","minimum":1},
            "if_revision":{"type":"integer","minimum":0},
            "transaction_id":{"type":"string","minLength":1,"maxLength":128},
            "visibility":{"type":"object","propertyNames":{"type":"string","minLength":1,"maxLength":256},"additionalProperties":{"type":"boolean"}},
            "selected":{"type":"object","propertyNames":{"type":"string","minLength":1,"maxLength":256},"additionalProperties":{"type":"string","minLength":1,"maxLength":256}},
            "sizes":{"type":"object","propertyNames":{"type":"string","minLength":1,"maxLength":256},"additionalProperties":{"type":"object"}},
            "splits":{"type":"object","propertyNames":{"type":"string","minLength":1,"maxLength":256},"additionalProperties":{"type":"object"}},
            "collapsed":{"type":"object","propertyNames":{"type":"string","minLength":1,"maxLength":256},"additionalProperties":{"type":"boolean"}},
            "configurations":{"type":"object","propertyNames":{"type":"string","minLength":1,"maxLength":256},"additionalProperties":{"type":"object"}},
            "active_region_id":{"type":"string","minLength":1,"maxLength":256},
            "focused_node_id":{"type":"string","minLength":1,"maxLength":256},
            "clear_focus":{"type":"boolean"}
        },
        "additionalProperties":false
    });
    let import_layout = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-import-layout-v1.json",
        "type":"object",
        "properties":{
            "mode":mode,
            "if_shell_revision":{"type":"integer","minimum":1},
            "if_revision":{"type":"integer","minimum":0},
            "transaction_id":{"type":"string","minLength":1,"maxLength":128},
            "document":shell_document_schema()
        },
        "required":["document"],
        "additionalProperties":false
    });
    let profile_scope =
        json!({"type":"string","enum":["session","application","project"],"default":"session"});
    let profile_name = json!({"type":"string","minLength":1,"maxLength":128});
    let profiles_list = json!({
        "type":"object",
        "properties":{"scope":profile_scope},
        "additionalProperties":false
    });
    let profiles_save = json!({
        "type":"object",
        "properties":{"name":profile_name,"scope":profile_scope,"mode":mode},
        "required":["name"],
        "additionalProperties":false
    });
    let profiles_load = json!({
        "type":"object",
        "properties":{
            "name":profile_name,
            "scope":profile_scope,
            "mode":mode,
            "if_shell_revision":{"type":"integer","minimum":1},
            "if_revision":{"type":"integer","minimum":0},
            "transaction_id":{"type":"string","minLength":1,"maxLength":128}
        },
        "required":["name"],
        "additionalProperties":false
    });
    let profiles_remove = json!({
        "type":"object",
        "properties":{"name":profile_name,"scope":profile_scope},
        "required":["name"],
        "additionalProperties":false
    });
    json!({
        "schema_version":SHELL_SCHEMA_VERSION,
        "json_schema_draft":"2020-12",
        "mutation_scope":"active_mode_only",
        "inactive_modes":"inspectable",
        "no_op_revision":"unchanged",
        "ownership_policy":{
            "snapshot_field":"ownership",
            "scopes":["application","extension"],
            "application_controller_capability":"ui.shell.application_control",
            "extension_rule":"a session that owns registered extensions may mutate its own extension nodes but not another extension's nodes",
            "application_controller_rule":"a session explicitly granted ui.shell.application_control may compose all registered mounts",
            "grant_negotiation":"system.hello.requested_capabilities -> granted_capabilities",
            "native_recovery_authority":"native-ui",
            "permission_error_fields":["method","node_id","mount","owner","required_capability","resolution"]
        },
        "capability_policy":{
            "read":"ui.shell.read",
            "composition":"ui.shell.compose",
            "extension_placement":"ui.shell.extension_place",
            "persistence":"ui.shell.persistence",
            "protected_recovery":"ui.shell.recovery",
            "application_chrome":"ui.shell.chrome",
            "platform_windows":"ui.shell.window_control",
            "shortcuts":"ui.shell.shortcuts",
            "application_controller":"ui.shell.application_control",
            "enforced_capabilities":["ui.shell.read","ui.shell.compose","ui.shell.extension_place","ui.shell.persistence","ui.shell.recovery","ui.shell.chrome","ui.shell.shortcuts","ui.shell.application_control"],
            "future_capabilities":["ui.shell.window_control"]
        },
        "snapshot_schema":snapshot,
        "get_schema":get,
        "export_layout_schema":get,
        "layout_document_schema":shell_document_schema(),
        "import_layout_schema":import_layout,
        "profiles_list_schema":profiles_list,
        "profiles_save_schema":profiles_save,
        "profiles_load_schema":profiles_load,
        "profiles_remove_schema":profiles_remove,
        "patch_schema":patch,
        "reset_schema":reset,
        "recover_schema":reset,
        "replace_layout_schema":replace_layout,
        "patch_layout_schema":patch_layout,
        "component_catalog_schema":component_catalog_schema(),
        "startup_restore":{
            "scope":"application",
            "settings_field":"shell_layout_startup_profiles",
            "activation":"once_per_mode_per_process",
            "failure_policy":"protected_recovery",
            "diagnostics_method":"app.settings.get"
        },
        "layout_limits":{
            "max_nodes":256,
            "max_depth":32,
            "configuration_bytes_per_node":MAX_CONFIGURATION_BYTES_PER_NODE,
            "configuration_bytes_total":MAX_CONFIGURATION_BYTES_TOTAL,
            "configuration_max_depth":MAX_CONFIGURATION_DEPTH,
            "configuration_values_per_node":MAX_CONFIGURATION_VALUES_PER_NODE
        },
        "document_migrations":{"current":1,"accepted":[0,1]}
    })
}

#[doc(hidden)]
pub fn shell_component_catalog(mode: &str) -> Value {
    component_catalog(Some(mode))
}

pub(super) fn shell_component_minimum_size(id: &str) -> Option<[f32; 2]> {
    component_minimum_size(id)
}

pub(crate) fn validate_layout_document(document: &Value) -> Result<(), ControlError> {
    document::validate_layout_document(document)
}

pub(crate) fn validate_layout_document_for(
    document: &Value,
    method: &str,
) -> Result<(), ControlError> {
    document::validate_layout_document_for(document, method)
}

pub(crate) fn normalize_layout_document(document: &Value) -> Result<Value, ControlError> {
    document::normalize_layout_document(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_patch_validates_identity_order_and_revision() {
        let mut shell = ShellModel::default();
        shell
            .patch(
                &json!({
                    "mode":"single",
                    "orders":{"builtin:single.left-tabs":[
                        "builtin:single.project", "builtin:single.layers"
                    ]},
                    "selected":{"builtin:single.left-tabs":"builtin:single.project"},
                    "visibility":{"extension:single.status-bar":false},
                }),
                ModelMode::Project,
            )
            .unwrap();
        let snapshot = shell
            .snapshot(
                ModelMode::Single,
                &[
                    ("builtin:single.left-panel", true),
                    ("builtin:single.right-panel", true),
                ],
                &[("builtin:single.left-tabs", "builtin:single.project")],
            )
            .unwrap();
        let tabs = snapshot["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == "builtin:single.left-tabs")
            .unwrap();
        assert_eq!(tabs["children"][0], "builtin:single.project");
        assert_eq!(tabs["selected_id"], "builtin:single.project");
        assert_eq!(tabs["mutable"]["order"], true);
        let layout_tabs = snapshot["layout"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == "layout:single.left-tabs")
            .unwrap();
        assert_eq!(layout_tabs["children"][0], "layout:single.project");
        assert_eq!(layout_tabs["selected_id"], "layout:single.project");
        let extension_status = snapshot["layout"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["mount"] == "builtin:extension-host.status-bar")
            .unwrap();
        assert_eq!(extension_status["visible"], false);
        let root = snapshot["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == "builtin:single.root")
            .unwrap();
        assert_eq!(root["mutable"]["order"], false);
        assert!(
            shell
                .patch(
                    &json!({"mode":"single","visibility":{"builtin:single.canvas":false}}),
                    ModelMode::Single,
                )
                .is_err()
        );
        assert!(
            shell
                .patch(
                    &json!({"mode":"single","orders":{"builtin:single.root":root["children"]}}),
                    ModelMode::Single,
                )
                .is_err()
        );
        assert!(
            shell
                .patch(
                    &json!({"mode":"mosaic","orders":{"builtin:mosaic.right-tabs":["builtin:mosaic.views"]}}),
                    ModelMode::Project,
                )
                .is_err()
        );
    }

    #[test]
    fn shell_reset_restores_defaults() {
        let mut shell = ShellModel::default();
        shell
            .patch(
                &json!({
                    "mode":"project",
                    "visibility":{"builtin:project.top-bar":false}
                }),
                ModelMode::Project,
            )
            .unwrap();
        let revision = shell.revision();
        shell
            .reset(
                &json!({"mode":"project","if_shell_revision":revision}),
                ModelMode::Project,
            )
            .unwrap();
        let snapshot = shell.mode_state("project").unwrap();
        let top_bar = snapshot["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == "builtin:project.top-bar")
            .unwrap();
        assert_eq!(top_bar["visible"], true);
        assert!(
            shell
                .patch(
                    &json!({"mode":"project","if_shell_revision":99}),
                    ModelMode::Project,
                )
                .is_err()
        );
    }

    #[test]
    fn shell_no_ops_preserve_revision_and_schema_is_machine_readable() {
        let mut shell = ShellModel::default();
        let revision = shell.revision();
        shell.patch(&json!({}), ModelMode::Project).unwrap();
        assert_eq!(shell.revision(), revision);
        shell.reset(&json!({}), ModelMode::Project).unwrap();
        assert_eq!(shell.revision(), revision);

        let schema = shell_schema();
        assert_eq!(schema["schema_version"], SHELL_SCHEMA_VERSION);
        assert_eq!(schema["mutation_scope"], "active_mode_only");
        assert_eq!(schema["no_op_revision"], "unchanged");
        assert_eq!(
            schema["snapshot_schema"]["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(
            schema["patch_schema"]["properties"]["orders"]["additionalProperties"]["uniqueItems"],
            true
        );
        assert_eq!(
            schema["component_catalog_schema"]["properties"]["components"]["type"],
            "array"
        );
        assert_eq!(
            schema["layout_limits"]["configuration_bytes_per_node"],
            MAX_CONFIGURATION_BYTES_PER_NODE
        );
        assert_eq!(
            schema["layout_limits"]["configuration_bytes_total"],
            MAX_CONFIGURATION_BYTES_TOTAL
        );
    }

    #[test]
    fn component_catalog_only_advertises_registered_commands() {
        for mode in ["project", "single", "mosaic"] {
            let catalog = shell_component_catalog(mode);
            for component in catalog["components"].as_array().unwrap() {
                for command in component["commands"].as_array().unwrap() {
                    let command = command.as_str().unwrap();
                    assert!(
                        crate::control::registry::method(command).is_some(),
                        "component {} advertises unknown command {command}",
                        component["id"]
                    );
                }
            }
        }
        let single = shell_component_catalog("single");
        let top_bar = single["components"]
            .as_array()
            .unwrap()
            .iter()
            .find(|component| component["id"] == "builtin:viewer-top-bar")
            .unwrap();
        assert_eq!(
            top_bar["configuration_schema"]["properties"]["show_panel_controls"]["default"],
            true
        );
        assert_eq!(
            top_bar["configuration_schema"]["additionalProperties"],
            false
        );
        let inspector = single["components"]
            .as_array()
            .unwrap()
            .iter()
            .find(|component| component["id"] == "builtin:shell-inspector")
            .unwrap();
        assert_eq!(inspector["kind"], "panel");
        assert!(
            inspector["legal_parent_types"]
                .as_array()
                .unwrap()
                .contains(&json!("tabs"))
        );
        let command_toolbar = single["components"]
            .as_array()
            .unwrap()
            .iter()
            .find(|component| component["id"] == "builtin:command-toolbar")
            .unwrap();
        assert_eq!(command_toolbar["kind"], "toolbar");
        assert_eq!(command_toolbar["commands"], json!(["ui.commands.execute"]));
    }
}
