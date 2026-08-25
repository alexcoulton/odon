//! Validated actor-owned desired application layout trees.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::control::ControlError;

const MAX_LAYOUT_NODES: usize = 256;
const MAX_LAYOUT_DEPTH: usize = 32;
const MAX_TEXT_LENGTH: usize = 256;
pub(super) const MAX_CONFIGURATION_BYTES_PER_NODE: usize = 16 * 1024;
pub(super) const MAX_CONFIGURATION_BYTES_TOTAL: usize = 256 * 1024;
pub(super) const MAX_CONFIGURATION_DEPTH: usize = 16;
pub(super) const MAX_CONFIGURATION_VALUES_PER_NODE: usize = 1024;
const MAX_CONFIGURATION_KEY_LENGTH: usize = 256;
const MAX_CONFIGURATION_STRING_BYTES: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ShellLayout {
    pub(super) root_id: String,
    pub(super) nodes: BTreeMap<String, LayoutNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LayoutNode {
    kind: LayoutKind,
    parent_id: Option<String>,
    children: Vec<String>,
    visible: bool,
    title: Option<String>,
    mount: Option<String>,
    selected_id: Option<String>,
    size: LayoutSize,
    split: Option<SplitOptions>,
    collapsed: bool,
    configuration: Value,
    state_bindings: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum LayoutKind {
    Application,
    Row,
    Column,
    Split,
    Tabs,
    Panel,
    Collapsible,
    Toolbar,
    StatusBar,
    MenuHost,
    CanvasSlot,
    BuiltinMount,
    ExtensionMount,
}

impl LayoutKind {
    fn name(self) -> &'static str {
        match self {
            Self::Application => "application",
            Self::Row => "row",
            Self::Column => "column",
            Self::Split => "split",
            Self::Tabs => "tabs",
            Self::Panel => "panel",
            Self::Collapsible => "collapsible",
            Self::Toolbar => "toolbar",
            Self::StatusBar => "status_bar",
            Self::MenuHost => "menu_host",
            Self::CanvasSlot => "canvas_slot",
            Self::BuiltinMount => "builtin_mount",
            Self::ExtensionMount => "extension_mount",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LayoutSize {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    width: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    height: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_width: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_height: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_width: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_height: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flex: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SplitOptions {
    #[serde(default)]
    axis: SplitAxis,
    ratio: f32,
    #[serde(default = "default_resizable")]
    resizable: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum SplitAxis {
    #[default]
    Horizontal,
    Vertical,
}

fn default_resizable() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DesiredLayout {
    root_id: String,
    nodes: Vec<DesiredLayoutNode>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DesiredLayoutNode {
    id: String,
    #[serde(rename = "type")]
    kind: LayoutKind,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    children: Vec<String>,
    #[serde(default = "default_visible")]
    visible: bool,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    mount: Option<String>,
    #[serde(default)]
    selected_id: Option<String>,
    #[serde(default)]
    size: LayoutSize,
    #[serde(default)]
    split: Option<SplitOptions>,
    #[serde(default)]
    collapsed: bool,
    #[serde(default = "empty_configuration")]
    configuration: Value,
    #[serde(default)]
    state_bindings: BTreeMap<String, Value>,
}

fn default_visible() -> bool {
    true
}

fn empty_configuration() -> Value {
    json!({})
}

impl ShellLayout {
    pub(super) fn contains_node(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    pub(super) fn preferred_active_region_id(&self) -> &str {
        self.nodes
            .iter()
            .find_map(|(id, node)| {
                node.mount
                    .as_deref()
                    .is_some_and(protected_mount)
                    .then_some(id.as_str())
            })
            .unwrap_or(self.root_id.as_str())
    }

    pub(super) fn from_value(
        value: &Value,
        mode: &str,
        method: &str,
    ) -> Result<Self, ControlError> {
        let desired: DesiredLayout = serde_json::from_value(value.clone()).map_err(|error| {
            ControlError::invalid_params(method, format!("invalid desired_tree: {error}"))
        })?;
        if desired.nodes.is_empty() {
            return Err(invalid(method, "desired_tree.nodes must not be empty"));
        }
        if desired.nodes.len() > MAX_LAYOUT_NODES {
            return Err(invalid(
                method,
                format!("desired_tree exceeds the {MAX_LAYOUT_NODES}-node limit"),
            ));
        }
        validate_id(method, "root_id", &desired.root_id)?;

        let mut nodes = BTreeMap::new();
        for desired_node in desired.nodes {
            validate_id(method, "node id", &desired_node.id)?;
            if let Some(parent_id) = desired_node.parent_id.as_deref() {
                validate_id(method, "parent_id", parent_id)?;
            }
            validate_optional_text(method, "title", desired_node.title.as_deref())?;
            validate_optional_text(method, "mount", desired_node.mount.as_deref())?;
            for child in &desired_node.children {
                validate_id(method, "child id", child)?;
            }
            if desired_node.children.iter().collect::<BTreeSet<_>>().len()
                != desired_node.children.len()
            {
                return Err(invalid(
                    method,
                    format!(
                        "layout node '{}' contains a duplicate child ID",
                        desired_node.id
                    ),
                ));
            }
            validate_size(method, &desired_node.id, &desired_node.size)?;
            if !desired_node.configuration.is_object() {
                return Err(invalid(
                    method,
                    format!(
                        "configuration for layout node '{}' must be an object",
                        desired_node.id
                    ),
                ));
            }
            let id = desired_node.id;
            let node = LayoutNode {
                kind: desired_node.kind,
                parent_id: desired_node.parent_id,
                children: desired_node.children,
                visible: desired_node.visible,
                title: desired_node.title,
                mount: desired_node.mount,
                selected_id: desired_node.selected_id,
                size: desired_node.size,
                split: desired_node.split,
                collapsed: desired_node.collapsed,
                configuration: desired_node.configuration,
                state_bindings: desired_node.state_bindings,
            };
            validate_layout_state_bindings(method, &id, &node.state_bindings)?;
            if nodes.insert(id.clone(), node).is_some() {
                return Err(invalid(method, format!("duplicate layout node ID '{id}'")));
            }
        }

        let mut layout = Self {
            root_id: desired.root_id,
            nodes,
        };
        layout.validate_and_link(mode, method)?;
        Ok(layout)
    }

    pub(super) fn to_json(&self) -> Value {
        let nodes = self
            .nodes
            .iter()
            .map(|(id, node)| {
                json!({
                    "id":id,
                    "type":node.kind,
                    "parent_id":node.parent_id,
                    "children":node.children,
                    "visible":node.visible,
                    "title":node.title,
                    "mount":node.mount,
                    "selected_id":node.selected_id,
                    "size":node.size,
                    "split":node.split,
                    "collapsed":node.collapsed,
                    "configuration":node.configuration,
                    "state_bindings":node.state_bindings,
                })
            })
            .collect::<Vec<_>>();
        json!({"root_id":self.root_id,"nodes":nodes})
    }

    pub(super) fn to_snapshot_json(&self) -> Value {
        let mut value = self.to_json();
        if let Some(nodes) = value.get_mut("nodes").and_then(Value::as_array_mut) {
            for node in nodes {
                let id = node.get("id").and_then(Value::as_str).unwrap_or_default();
                let kind = node.get("type").and_then(Value::as_str).unwrap_or_default();
                let mount = node
                    .get("mount")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let (scope, owner_id) = if kind == "extension_mount" {
                    (
                        "extension",
                        mount
                            .as_deref()
                            .and_then(extension_owner_id)
                            .unwrap_or("unregistered")
                            .to_string(),
                    )
                } else {
                    ("application", "odon".to_string())
                };
                let protected = id == self.root_id || mount.as_deref().is_some_and(protected_mount);
                node.as_object_mut()
                    .expect("serialized layout nodes are objects")
                    .insert(
                        "ownership".to_string(),
                        json!({
                            "scope":scope,
                            "owner_id":owner_id,
                            "protected":protected,
                        }),
                    );
            }
        }
        value
    }

    pub(super) fn set_mount_visible(&mut self, mount: &str, visible: bool) -> bool {
        let Some(node) = self
            .nodes
            .values_mut()
            .find(|node| node.mount.as_deref() == Some(mount))
        else {
            return false;
        };
        let changed = node.visible != visible;
        node.visible = visible;
        changed
    }

    pub(super) fn select_mount(&mut self, mount: &str) -> bool {
        let Some(mut current) = self
            .nodes
            .iter()
            .find_map(|(id, node)| (node.mount.as_deref() == Some(mount)).then(|| id.clone()))
        else {
            return false;
        };
        loop {
            let Some(parent_id) = self
                .nodes
                .get(&current)
                .and_then(|node| node.parent_id.clone())
            else {
                return false;
            };
            let parent = self.nodes.get_mut(&parent_id).expect("validated parent");
            if parent.kind == LayoutKind::Tabs {
                let changed = parent.selected_id.as_deref() != Some(current.as_str());
                parent.selected_id = Some(current);
                return changed;
            }
            current = parent_id;
        }
    }

    pub(super) fn reorder_mounts(&mut self, mounts: &[&str]) -> bool {
        if mounts.is_empty() {
            return false;
        }
        let mount_nodes = mounts
            .iter()
            .filter_map(|mount| {
                self.nodes.iter().find_map(|(id, node)| {
                    (node.mount.as_deref() == Some(*mount)).then(|| id.clone())
                })
            })
            .collect::<Vec<_>>();
        if mount_nodes.len() != mounts.len() {
            return false;
        }
        let Some(parent_id) = mount_nodes
            .first()
            .and_then(|id| self.nodes.get(id).and_then(|node| node.parent_id.clone()))
        else {
            return false;
        };
        if !mount_nodes.iter().all(|id| {
            self.nodes
                .get(id)
                .and_then(|node| node.parent_id.as_deref())
                == Some(parent_id.as_str())
        }) {
            return false;
        }
        let parent = self.nodes.get_mut(&parent_id).expect("validated parent");
        if parent.kind != LayoutKind::Tabs
            || !mount_nodes.iter().all(|id| parent.children.contains(id))
        {
            return false;
        }
        // Version-1 compatibility orders only enumerate the legacy built-in tabs. Preserve
        // first-class extension-host children in their existing slots while reordering the
        // requested subset.
        let requested = mount_nodes.iter().cloned().collect::<BTreeSet<_>>();
        let mut replacements = mount_nodes.into_iter();
        let reordered = parent
            .children
            .iter()
            .map(|child| {
                if requested.contains(child) {
                    replacements.next().expect("requested child replacement")
                } else {
                    child.clone()
                }
            })
            .collect::<Vec<_>>();
        let changed = parent.children != reordered;
        parent.children = reordered;
        changed
    }

    pub(super) fn patch_state(
        &mut self,
        params: &Value,
        mode: &str,
        method: &str,
    ) -> Result<(), ControlError> {
        if let Some(visibility) = params.get("visibility") {
            let visibility = visibility.as_object().ok_or_else(|| {
                invalid(method, "visibility must map layout node IDs to booleans")
            })?;
            for (id, value) in visibility {
                let visible = value.as_bool().ok_or_else(|| {
                    invalid(method, format!("visibility for '{id}' must be a boolean"))
                })?;
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| invalid(method, format!("unknown layout node '{id}'")))?;
                node.visible = visible;
            }
        }
        if let Some(selected) = params.get("selected") {
            let selected = selected.as_object().ok_or_else(|| {
                invalid(
                    method,
                    "selected must map tabs node IDs to direct child IDs",
                )
            })?;
            for (id, value) in selected {
                let child = value.as_str().ok_or_else(|| {
                    invalid(
                        method,
                        format!("selected value for '{id}' must be a string"),
                    )
                })?;
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| invalid(method, format!("unknown layout node '{id}'")))?;
                if node.kind != LayoutKind::Tabs || !node.children.iter().any(|id| id == child) {
                    return Err(invalid(
                        method,
                        format!("layout tabs node '{id}' cannot select '{child}'"),
                    ));
                }
                node.selected_id = Some(child.to_string());
            }
        }
        if let Some(sizes) = params.get("sizes") {
            let sizes = sizes
                .as_object()
                .ok_or_else(|| invalid(method, "sizes must map layout node IDs to size objects"))?;
            for (id, value) in sizes {
                let size: LayoutSize = serde_json::from_value(value.clone()).map_err(|error| {
                    invalid(
                        method,
                        format!("invalid size for layout node '{id}': {error}"),
                    )
                })?;
                validate_size(method, id, &size)?;
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| invalid(method, format!("unknown layout node '{id}'")))?;
                node.size = size;
            }
        }
        if let Some(splits) = params.get("splits") {
            let splits = splits.as_object().ok_or_else(|| {
                invalid(method, "splits must map split node IDs to split options")
            })?;
            for (id, value) in splits {
                let split: SplitOptions =
                    serde_json::from_value(value.clone()).map_err(|error| {
                        invalid(method, format!("invalid split options for '{id}': {error}"))
                    })?;
                if !split.ratio.is_finite() || !(0.05..=0.95).contains(&split.ratio) {
                    return Err(invalid(
                        method,
                        format!("split ratio for '{id}' must be between 0.05 and 0.95"),
                    ));
                }
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| invalid(method, format!("unknown layout node '{id}'")))?;
                if node.kind != LayoutKind::Split {
                    return Err(invalid(
                        method,
                        format!("layout node '{id}' is not a split"),
                    ));
                }
                node.split = Some(split);
            }
        }
        if let Some(collapsed) = params.get("collapsed") {
            let collapsed = collapsed.as_object().ok_or_else(|| {
                invalid(
                    method,
                    "collapsed must map collapsible node IDs to booleans",
                )
            })?;
            for (id, value) in collapsed {
                let collapsed = value.as_bool().ok_or_else(|| {
                    invalid(
                        method,
                        format!("collapsed value for '{id}' must be a boolean"),
                    )
                })?;
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| invalid(method, format!("unknown layout node '{id}'")))?;
                if node.kind != LayoutKind::Collapsible {
                    return Err(invalid(
                        method,
                        format!("layout node '{id}' is not collapsible"),
                    ));
                }
                node.collapsed = collapsed;
            }
        }
        if let Some(configurations) = params.get("configurations") {
            let configurations = configurations.as_object().ok_or_else(|| {
                invalid(
                    method,
                    "configurations must map layout node IDs to configuration objects",
                )
            })?;
            for (id, configuration) in configurations {
                if !configuration.is_object() {
                    return Err(invalid(
                        method,
                        format!("configuration for layout node '{id}' must be an object"),
                    ));
                }
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| invalid(method, format!("unknown layout node '{id}'")))?;
                node.configuration = configuration.clone();
            }
        }
        self.validate_and_link(mode, method)
    }

    fn validate_and_link(&mut self, mode: &str, method: &str) -> Result<(), ControlError> {
        let Some(root) = self.nodes.get(&self.root_id) else {
            return Err(invalid(method, "desired_tree.root_id is not a known node"));
        };
        if root.kind != LayoutKind::Application {
            return Err(invalid(
                method,
                "desired_tree root must have type 'application'",
            ));
        }

        let mut parents = BTreeMap::<String, String>::new();
        for (id, node) in &self.nodes {
            validate_node_shape(method, id, node)?;
            for child in &node.children {
                if !self.nodes.contains_key(child) {
                    return Err(invalid(
                        method,
                        format!("layout node '{id}' references unknown child '{child}'"),
                    ));
                }
                if let Some(first_parent) = parents.insert(child.clone(), id.clone()) {
                    return Err(invalid(
                        method,
                        format!(
                            "layout node '{child}' has multiple parents ('{first_parent}' and '{id}')"
                        ),
                    ));
                }
            }
        }
        validate_configuration_quotas(method, &self.nodes)?;
        if parents.contains_key(&self.root_id) {
            return Err(invalid(method, "desired_tree root cannot be a child"));
        }
        for (id, node) in &mut self.nodes {
            let derived = parents.get(id).cloned();
            if node.parent_id.is_some() && node.parent_id != derived {
                return Err(invalid(
                    method,
                    format!("declared parent_id for layout node '{id}' does not match its parent"),
                ));
            }
            node.parent_id = derived;
        }

        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        self.visit(&self.root_id, 1, &mut visiting, &mut visited, method)?;
        if visited.len() != self.nodes.len() {
            let unreachable = self
                .nodes
                .keys()
                .find(|id| !visited.contains(*id))
                .expect("node count differs")
                .clone();
            return Err(invalid(
                method,
                format!("layout node '{unreachable}' is not reachable from root_id"),
            ));
        }

        validate_mounts(mode, method, &self.nodes)
    }

    fn visit(
        &self,
        id: &str,
        depth: usize,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
        method: &str,
    ) -> Result<(), ControlError> {
        if depth > MAX_LAYOUT_DEPTH {
            return Err(invalid(
                method,
                format!("desired_tree exceeds the maximum depth of {MAX_LAYOUT_DEPTH}"),
            ));
        }
        if !visiting.insert(id.to_string()) {
            return Err(invalid(
                method,
                format!("layout contains a cycle at '{id}'"),
            ));
        }
        let node = self.nodes.get(id).expect("validated child identity");
        for child in &node.children {
            self.visit(child, depth + 1, visiting, visited, method)?;
        }
        visiting.remove(id);
        visited.insert(id.to_string());
        Ok(())
    }
}

fn validate_configuration_quotas(
    method: &str,
    nodes: &BTreeMap<String, LayoutNode>,
) -> Result<(), ControlError> {
    let mut total_bytes = 0usize;
    for (node_id, node) in nodes {
        let encoded_bytes = serde_json::to_vec(&node.configuration)
            .map_err(|error| invalid(method, format!("cannot encode configuration: {error}")))?
            .len();
        if encoded_bytes > MAX_CONFIGURATION_BYTES_PER_NODE {
            return Err(invalid(
                method,
                format!(
                    "configuration for layout node '{node_id}' exceeds the {MAX_CONFIGURATION_BYTES_PER_NODE}-byte per-node limit"
                ),
            ));
        }
        total_bytes = total_bytes.saturating_add(encoded_bytes);
        if total_bytes > MAX_CONFIGURATION_BYTES_TOTAL {
            return Err(invalid(
                method,
                format!(
                    "layout configuration exceeds the {MAX_CONFIGURATION_BYTES_TOTAL}-byte total limit"
                ),
            ));
        }

        let mut values = 0usize;
        let mut pending = vec![(&node.configuration, 1usize)];
        while let Some((value, depth)) = pending.pop() {
            if depth > MAX_CONFIGURATION_DEPTH {
                return Err(invalid(
                    method,
                    format!(
                        "configuration for layout node '{node_id}' exceeds the maximum depth of {MAX_CONFIGURATION_DEPTH}"
                    ),
                ));
            }
            values = values.saturating_add(1);
            if values > MAX_CONFIGURATION_VALUES_PER_NODE {
                return Err(invalid(
                    method,
                    format!(
                        "configuration for layout node '{node_id}' exceeds the {MAX_CONFIGURATION_VALUES_PER_NODE}-value limit"
                    ),
                ));
            }
            match value {
                Value::Object(object) => {
                    for (key, child) in object {
                        if key.is_empty()
                            || key.len() > MAX_CONFIGURATION_KEY_LENGTH
                            || key.chars().any(char::is_control)
                        {
                            return Err(invalid(
                                method,
                                format!(
                                    "configuration key for layout node '{node_id}' must contain 1 to {MAX_CONFIGURATION_KEY_LENGTH} non-control bytes"
                                ),
                            ));
                        }
                        pending.push((child, depth + 1));
                    }
                }
                Value::Array(array) => {
                    pending.extend(array.iter().map(|child| (child, depth + 1)));
                }
                Value::String(text) if text.len() > MAX_CONFIGURATION_STRING_BYTES => {
                    return Err(invalid(
                        method,
                        format!(
                            "configuration string for layout node '{node_id}' exceeds the {MAX_CONFIGURATION_STRING_BYTES}-byte limit"
                        ),
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_node_shape(method: &str, id: &str, node: &LayoutNode) -> Result<(), ControlError> {
    let count = node.children.len();
    let expected = match node.kind {
        LayoutKind::Application | LayoutKind::Row | LayoutKind::Column => count >= 1,
        LayoutKind::Split => count == 2,
        LayoutKind::Tabs => count >= 1,
        LayoutKind::Panel | LayoutKind::Collapsible => count == 1,
        LayoutKind::Toolbar | LayoutKind::StatusBar | LayoutKind::MenuHost => count >= 1,
        LayoutKind::CanvasSlot | LayoutKind::BuiltinMount | LayoutKind::ExtensionMount => {
            count == 0
        }
    };
    if !expected {
        return Err(invalid(
            method,
            format!("layout node '{id}' has an invalid child count for its type"),
        ));
    }
    if node.kind == LayoutKind::Tabs {
        let Some(selected) = node.selected_id.as_deref() else {
            return Err(invalid(
                method,
                format!("tabs node '{id}' requires selected_id"),
            ));
        };
        if !node.children.iter().any(|child| child == selected) {
            return Err(invalid(
                method,
                format!("tabs node '{id}' selects non-child '{selected}'"),
            ));
        }
    } else if node.selected_id.is_some() {
        return Err(invalid(
            method,
            format!("only tabs nodes may define selected_id (node '{id}')"),
        ));
    }
    if node.kind == LayoutKind::Split {
        let Some(split) = node.split.as_ref() else {
            return Err(invalid(
                method,
                format!("split node '{id}' requires split options"),
            ));
        };
        if !split.ratio.is_finite() || !(0.05..=0.95).contains(&split.ratio) {
            return Err(invalid(
                method,
                format!("split ratio for '{id}' must be between 0.05 and 0.95"),
            ));
        }
    } else if node.split.is_some() {
        return Err(invalid(
            method,
            format!("only split nodes may define split options (node '{id}')"),
        ));
    }
    let is_mount = matches!(
        node.kind,
        LayoutKind::CanvasSlot | LayoutKind::BuiltinMount | LayoutKind::ExtensionMount
    );
    if is_mount != node.mount.is_some() {
        return Err(invalid(
            method,
            format!(
                "layout node '{id}' {} define mount",
                if is_mount { "must" } else { "must not" }
            ),
        ));
    }
    if !is_mount
        && node
            .configuration
            .as_object()
            .is_some_and(|configuration| !configuration.is_empty())
    {
        return Err(invalid(
            method,
            format!("only mount nodes may define configuration (node '{id}')"),
        ));
    }
    if node.collapsed && node.kind != LayoutKind::Collapsible {
        return Err(invalid(
            method,
            format!("only collapsible nodes may be collapsed (node '{id}')"),
        ));
    }
    Ok(())
}

fn validate_layout_state_bindings(
    method: &str,
    node_id: &str,
    bindings: &BTreeMap<String, Value>,
) -> Result<(), ControlError> {
    for (property, binding) in bindings {
        if property != "visible" {
            return Err(invalid(
                method,
                format!("layout node '{node_id}' can bind only its visible property"),
            ));
        }
        let object = binding.as_object().ok_or_else(|| {
            invalid(
                method,
                format!("layout node '{node_id}' visible binding must be an object"),
            )
        })?;
        if object
            .keys()
            .any(|key| !matches!(key.as_str(), "type" | "command_id" | "state" | "equals"))
        {
            return Err(invalid(
                method,
                format!("layout node '{node_id}' visible binding contains an unknown field"),
            ));
        }
        if object.get("type").and_then(Value::as_str) != Some("command_state") {
            return Err(invalid(
                method,
                format!("layout node '{node_id}' binding type must be 'command_state'"),
            ));
        }
        validate_id(
            method,
            "layout command-state binding command_id",
            object
                .get("command_id")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        )?;
        if !object
            .get("state")
            .and_then(Value::as_str)
            .is_some_and(|state| matches!(state, "visible" | "enabled" | "checked"))
        {
            return Err(invalid(
                method,
                format!(
                    "layout node '{node_id}' binding state must be visible, enabled, or checked"
                ),
            ));
        }
        if object
            .get("equals")
            .is_some_and(|value| !value.is_boolean())
        {
            return Err(invalid(
                method,
                format!("layout node '{node_id}' binding equals must be a boolean"),
            ));
        }
    }
    Ok(())
}

fn validate_mounts(
    mode: &str,
    method: &str,
    nodes: &BTreeMap<String, LayoutNode>,
) -> Result<(), ControlError> {
    let required = required_mount(mode);
    let mut seen = BTreeSet::new();
    for (id, node) in nodes {
        let Some(mount) = node.mount.as_deref() else {
            continue;
        };
        match node.kind {
            LayoutKind::CanvasSlot | LayoutKind::BuiltinMount => {
                let Some(component) = component_descriptor(mount) else {
                    return Err(invalid(
                        method,
                        format!("unknown built-in component mount '{mount}'"),
                    ));
                };
                let available_in_mode = component["modes"]
                    .as_array()
                    .is_some_and(|modes| modes.iter().any(|candidate| candidate == mode));
                if !available_in_mode {
                    return Err(invalid(
                        method,
                        format!("built-in mount '{mount}' is not available in {mode} mode"),
                    ));
                }
                let singleton = component["singleton"].as_bool().unwrap_or(true);
                let duplicate = !seen.insert(mount);
                if singleton && duplicate {
                    return Err(invalid(
                        method,
                        format!("singleton built-in mount '{mount}' appears more than once"),
                    ));
                }
                let parent_kind = node
                    .parent_id
                    .as_deref()
                    .and_then(|parent| nodes.get(parent))
                    .map(|parent| parent.kind.name())
                    .ok_or_else(|| {
                        invalid(
                            method,
                            format!("built-in mount '{mount}' requires a parent container"),
                        )
                    })?;
                let legal_parent = component["legal_parent_types"]
                    .as_array()
                    .is_some_and(|parents| parents.iter().any(|parent| parent == parent_kind));
                if !legal_parent {
                    return Err(invalid(
                        method,
                        format!(
                            "built-in mount '{mount}' cannot be placed in parent type '{parent_kind}'"
                        ),
                    ));
                }
                let component_kind = component["kind"].as_str().unwrap_or_default();
                if node.kind == LayoutKind::CanvasSlot && component_kind != "canvas" {
                    return Err(invalid(
                        method,
                        format!("canvas slot '{id}' cannot mount non-canvas component '{mount}'"),
                    ));
                }
                if node.kind == LayoutKind::BuiltinMount && component_kind == "canvas" {
                    return Err(invalid(
                        method,
                        format!("canvas component '{mount}' requires a canvas_slot node"),
                    ));
                }
                if node.kind == LayoutKind::CanvasSlot && mount != required {
                    return Err(invalid(
                        method,
                        format!("canvas slot '{id}' must mount '{required}' in {mode} mode"),
                    ));
                }
                validate_component_configuration(method, id, mount, &node.configuration)?;
            }
            LayoutKind::ExtensionMount => {
                if !mount.starts_with("extension:") || mount.len() <= "extension:".len() {
                    return Err(invalid(
                        method,
                        format!("extension mount '{mount}' must use the extension: namespace"),
                    ));
                }
            }
            _ => unreachable!("mount presence validated by node type"),
        }
    }
    if !seen.contains(required) {
        return Err(invalid(
            method,
            format!("desired_tree must retain required mount '{required}'"),
        ));
    }
    let required_id = nodes
        .iter()
        .find_map(|(id, node)| (node.mount.as_deref() == Some(required)).then_some(id.as_str()))
        .expect("required mount was seen");
    let mut current = required_id;
    loop {
        let node = nodes
            .get(current)
            .expect("validated required mount ancestry");
        if !node.visible || (node.kind == LayoutKind::Collapsible && node.collapsed) {
            return Err(invalid(
                method,
                format!("required mount '{required}' must remain usable and visible"),
            ));
        }
        let Some(parent) = node.parent_id.as_deref() else {
            break;
        };
        let parent_node = nodes.get(parent).expect("validated required mount parent");
        if parent_node.kind == LayoutKind::Tabs
            && parent_node.selected_id.as_deref() != Some(current)
        {
            return Err(invalid(
                method,
                format!("required mount '{required}' must remain selected and usable"),
            ));
        }
        current = parent;
    }
    Ok(())
}

fn validate_component_configuration(
    method: &str,
    node_id: &str,
    mount: &str,
    configuration: &Value,
) -> Result<(), ControlError> {
    let object = configuration.as_object().ok_or_else(|| {
        invalid(
            method,
            format!("configuration for layout node '{node_id}' must be an object"),
        )
    })?;
    let allowed = match mount {
        "builtin:project-top-bar" => &["show_title"][..],
        "builtin:viewer-top-bar" => &[
            "show_title",
            "show_navigation",
            "show_panel_controls",
            "show_viewport_controls",
            "show_rendering_controls",
            "show_contrast_controls",
        ][..],
        "builtin:mosaic-top-bar" => &[
            "show_title",
            "show_navigation",
            "show_status",
            "show_panel_controls",
            "show_rendering_controls",
            "show_contrast_controls",
        ][..],
        _ => &[][..],
    };
    for (key, value) in object {
        if !allowed.contains(&key.as_str()) {
            return Err(invalid(
                method,
                format!("unknown configuration property '{key}' for mount '{mount}'"),
            ));
        }
        if !value.is_boolean() {
            return Err(invalid(
                method,
                format!("configuration property '{key}' for mount '{mount}' must be boolean"),
            ));
        }
    }
    Ok(())
}

fn required_mount(mode: &str) -> &'static str {
    match mode {
        "project" => "builtin:project-workspace",
        "single" => "builtin:viewer-canvas",
        "mosaic" => "builtin:mosaic-canvas",
        _ => unreachable!("validated mode"),
    }
}

fn component_descriptor(id: &str) -> Option<Value> {
    component_descriptors()
        .into_iter()
        .find(|component| component["id"] == id)
}

pub(super) fn component_minimum_size(id: &str) -> Option<[f32; 2]> {
    let component = component_descriptor(id)?;
    Some([
        component.pointer("/minimum_size/width")?.as_f64()? as f32,
        component.pointer("/minimum_size/height")?.as_f64()? as f32,
    ])
}

fn validate_size(method: &str, id: &str, size: &LayoutSize) -> Result<(), ControlError> {
    for (name, value, allow_zero) in [
        ("width", size.width, false),
        ("height", size.height, false),
        ("min_width", size.min_width, true),
        ("min_height", size.min_height, true),
        ("max_width", size.max_width, false),
        ("max_height", size.max_height, false),
        ("flex", size.flex, false),
    ] {
        if let Some(value) = value
            && (!value.is_finite()
                || if allow_zero {
                    value < 0.0
                } else {
                    value <= 0.0
                })
        {
            return Err(invalid(
                method,
                format!("layout size {name} for '{id}' must be finite and positive"),
            ));
        }
    }
    if size
        .min_width
        .zip(size.max_width)
        .is_some_and(|(min, max)| min > max)
        || size
            .min_height
            .zip(size.max_height)
            .is_some_and(|(min, max)| min > max)
    {
        return Err(invalid(
            method,
            format!("layout size minimum exceeds maximum for '{id}'"),
        ));
    }
    Ok(())
}

fn validate_id(method: &str, label: &str, value: &str) -> Result<(), ControlError> {
    if value.is_empty() || value.len() > MAX_TEXT_LENGTH || value.chars().any(char::is_control) {
        return Err(invalid(
            method,
            format!("{label} must contain 1 to {MAX_TEXT_LENGTH} non-control characters"),
        ));
    }
    Ok(())
}

fn validate_optional_text(
    method: &str,
    label: &str,
    value: Option<&str>,
) -> Result<(), ControlError> {
    if let Some(value) = value {
        validate_id(method, label, value)?;
    }
    Ok(())
}

fn invalid(method: &str, message: impl Into<String>) -> ControlError {
    ControlError::invalid_params(method, message)
}

pub(super) fn default_layout(mode: &str) -> ShellLayout {
    let value = match mode {
        "project" => json!({
            "root_id":"layout:project.root",
            "nodes":[
                node("layout:project.root", "application", &["layout:project.top-host", "layout:project.body", "layout:project.status-host"]),
                node("layout:project.top-host", "toolbar", &["layout:project.top", "layout:project.top-actions"]),
                mount("layout:project.top", "builtin_mount", "builtin:project-top-bar"),
                titled_mount("layout:project.top-actions", "builtin_mount", "builtin:extension-host.top-bar-actions", "Extensions"),
                node("layout:project.body", "row", &["layout:project.workspace", "layout:project.extensions"]),
                mount("layout:project.workspace", "builtin_mount", "builtin:project-workspace"),
                sized_titled_mount("layout:project.extensions", "builtin_mount", "builtin:extension-host.project-cards", "Extensions", 320.0, 180.0),
                node("layout:project.status-host", "status_bar", &["layout:project.status"]),
                titled_mount("layout:project.status", "builtin_mount", "builtin:extension-host.status-bar", "Extension status")
            ]
        }),
        "single" => json!({
            "root_id":"layout:single.root",
            "nodes":[
                node("layout:single.root", "application", &["layout:single.top-host", "layout:single.body", "layout:single.status-host"]),
                node("layout:single.top-host", "toolbar", &["layout:single.top", "layout:single.top-actions"]),
                mount("layout:single.top", "builtin_mount", "builtin:viewer-top-bar"),
                titled_mount("layout:single.top-actions", "builtin_mount", "builtin:extension-host.top-bar-actions", "Extensions"),
                split("layout:single.body", "layout:single.left", "layout:single.center-right", "horizontal", 0.26),
                split("layout:single.center-right", "layout:single.canvas-column", "layout:single.right", "horizontal", 0.63),
                panel("layout:single.left", "layout:single.left-tabs", 360.0),
                tabs("layout:single.left-tabs", &["layout:single.layers", "layout:single.project", "layout:single.left-extensions"], "layout:single.layers"),
                mount("layout:single.layers", "builtin_mount", "builtin:layers"),
                mount("layout:single.project", "builtin_mount", "builtin:project"),
                titled_mount("layout:single.left-extensions", "builtin_mount", "builtin:extension-host.left-sections", "Extensions"),
                node("layout:single.canvas-column", "column", &["layout:single.canvas-controls", "layout:single.canvas"]),
                titled_mount("layout:single.canvas-controls", "builtin_mount", "builtin:extension-host.canvas-controls", "Canvas controls"),
                sized_mount("layout:single.canvas", "canvas_slot", "builtin:viewer-canvas", 256.0, 256.0),
                panel("layout:single.right", "layout:single.right-tabs", 380.0),
                tabs("layout:single.right-tabs", &["layout:single.properties", "layout:single.views", "layout:single.analysis", "layout:single.measurements", "layout:single.memory", "layout:single.roi-selector", "layout:single.right-extensions"], "layout:single.properties"),
                mount("layout:single.properties", "builtin_mount", "builtin:properties"),
                mount("layout:single.views", "builtin_mount", "builtin:views"),
                mount("layout:single.analysis", "builtin_mount", "builtin:analysis"),
                mount("layout:single.measurements", "builtin_mount", "builtin:measurements"),
                mount("layout:single.memory", "builtin_mount", "builtin:memory"),
                mount("layout:single.roi-selector", "builtin_mount", "builtin:roi-selector"),
                titled_mount("layout:single.right-extensions", "builtin_mount", "builtin:extension-host.right-tabs", "Extensions"),
                node("layout:single.status-host", "status_bar", &["layout:single.status"]),
                titled_mount("layout:single.status", "builtin_mount", "builtin:extension-host.status-bar", "Extension status")
            ]
        }),
        "mosaic" => json!({
            "root_id":"layout:mosaic.root",
            "nodes":[
                node("layout:mosaic.root", "application", &["layout:mosaic.top-host", "layout:mosaic.body", "layout:mosaic.status-host"]),
                node("layout:mosaic.top-host", "toolbar", &["layout:mosaic.top", "layout:mosaic.top-actions"]),
                mount("layout:mosaic.top", "builtin_mount", "builtin:mosaic-top-bar"),
                titled_mount("layout:mosaic.top-actions", "builtin_mount", "builtin:extension-host.top-bar-actions", "Extensions"),
                split("layout:mosaic.body", "layout:mosaic.left", "layout:mosaic.center-right", "horizontal", 0.26),
                split("layout:mosaic.center-right", "layout:mosaic.canvas-column", "layout:mosaic.right", "horizontal", 0.63),
                panel("layout:mosaic.left", "layout:mosaic.left-tabs", 360.0),
                tabs("layout:mosaic.left-tabs", &["layout:mosaic.layers", "layout:mosaic.project", "layout:mosaic.left-extensions"], "layout:mosaic.layers"),
                mount("layout:mosaic.layers", "builtin_mount", "builtin:layers"),
                mount("layout:mosaic.project", "builtin_mount", "builtin:project"),
                titled_mount("layout:mosaic.left-extensions", "builtin_mount", "builtin:extension-host.left-sections", "Extensions"),
                node("layout:mosaic.canvas-column", "column", &["layout:mosaic.canvas-controls", "layout:mosaic.canvas"]),
                titled_mount("layout:mosaic.canvas-controls", "builtin_mount", "builtin:extension-host.canvas-controls", "Canvas controls"),
                sized_mount("layout:mosaic.canvas", "canvas_slot", "builtin:mosaic-canvas", 256.0, 256.0),
                panel("layout:mosaic.right", "layout:mosaic.right-tabs", 380.0),
                tabs("layout:mosaic.right-tabs", &["layout:mosaic.properties", "layout:mosaic.views", "layout:mosaic.layout", "layout:mosaic.memory", "layout:mosaic.right-extensions"], "layout:mosaic.properties"),
                mount("layout:mosaic.properties", "builtin_mount", "builtin:properties"),
                mount("layout:mosaic.views", "builtin_mount", "builtin:views"),
                mount("layout:mosaic.layout", "builtin_mount", "builtin:mosaic-layout"),
                mount("layout:mosaic.memory", "builtin_mount", "builtin:memory"),
                titled_mount("layout:mosaic.right-extensions", "builtin_mount", "builtin:extension-host.right-tabs", "Extensions"),
                node("layout:mosaic.status-host", "status_bar", &["layout:mosaic.status"]),
                titled_mount("layout:mosaic.status", "builtin_mount", "builtin:extension-host.status-bar", "Extension status")
            ]
        }),
        _ => unreachable!("validated mode"),
    };
    ShellLayout::from_value(&value, mode, "default shell layout")
        .expect("built-in desired layout is valid")
}

pub(super) fn recovery_layout(mode: &str) -> ShellLayout {
    let (root_id, content_id, kind, mount_id) = match mode {
        "project" => (
            "layout:recovery.project.root",
            "layout:recovery.project.workspace",
            "builtin_mount",
            "builtin:project-workspace",
        ),
        "single" => (
            "layout:recovery.single.root",
            "layout:recovery.single.canvas",
            "canvas_slot",
            "builtin:viewer-canvas",
        ),
        "mosaic" => (
            "layout:recovery.mosaic.root",
            "layout:recovery.mosaic.canvas",
            "canvas_slot",
            "builtin:mosaic-canvas",
        ),
        _ => unreachable!("validated mode"),
    };
    let value = json!({
        "root_id":root_id,
        "nodes":[
            node(root_id, "application", &[content_id]),
            sized_mount(content_id, kind, mount_id, 256.0, 256.0),
        ]
    });
    ShellLayout::from_value(&value, mode, "protected recovery shell layout")
        .expect("built-in recovery layout is valid")
}

fn node(id: &str, kind: &str, children: &[&str]) -> Value {
    json!({"id":id,"type":kind,"children":children})
}

fn mount(id: &str, kind: &str, mount: &str) -> Value {
    json!({"id":id,"type":kind,"mount":mount})
}

fn titled_mount(id: &str, kind: &str, mount: &str, title: &str) -> Value {
    json!({"id":id,"type":kind,"mount":mount,"title":title})
}

fn sized_titled_mount(
    id: &str,
    kind: &str,
    mount: &str,
    title: &str,
    width: f32,
    min_height: f32,
) -> Value {
    json!({
        "id":id,
        "type":kind,
        "mount":mount,
        "title":title,
        "size":{"width":width,"min_height":min_height},
    })
}

fn sized_mount(id: &str, kind: &str, mount: &str, min_width: f32, min_height: f32) -> Value {
    json!({
        "id":id,
        "type":kind,
        "mount":mount,
        "size":{"min_width":min_width,"min_height":min_height},
    })
}

fn panel(id: &str, child: &str, width: f32) -> Value {
    json!({
        "id":id,
        "type":"panel",
        "children":[child],
        "size":{"width":width,"min_width":220.0,"min_height":120.0},
    })
}

fn split(id: &str, first: &str, second: &str, axis: &str, ratio: f32) -> Value {
    json!({
        "id":id,
        "type":"split",
        "children":[first,second],
        "split":{"axis":axis,"ratio":ratio,"resizable":true},
    })
}

fn tabs(id: &str, children: &[&str], selected: &str) -> Value {
    json!({"id":id,"type":"tabs","children":children,"selected_id":selected})
}

pub(super) fn layout_schema() -> Value {
    json!({
        "type":"object",
        "properties":{
            "root_id":{"type":"string","minLength":1,"maxLength":256},
            "nodes":{
                "type":"array","minItems":1,"maxItems":MAX_LAYOUT_NODES,
                "items":{
                    "type":"object",
                    "properties":{
                        "id":{"type":"string","minLength":1,"maxLength":256},
                        "type":{"type":"string","enum":["application","row","column","split","tabs","panel","collapsible","toolbar","status_bar","menu_host","canvas_slot","builtin_mount","extension_mount"]},
                        "parent_id":{"type":["string","null"],"minLength":1,"maxLength":256},
                        "children":{"type":"array","items":{"type":"string","minLength":1,"maxLength":256},"uniqueItems":true},
                        "visible":{"type":"boolean","default":true},
                        "title":{"type":"string","minLength":1,"maxLength":256},
                        "mount":{"type":"string","minLength":1,"maxLength":256},
                        "selected_id":{"type":"string","minLength":1,"maxLength":256},
                        "size":{
                            "type":"object",
                            "properties":{
                                "width":{"type":"number","exclusiveMinimum":0},
                                "height":{"type":"number","exclusiveMinimum":0},
                                "min_width":{"type":"number","minimum":0},
                                "min_height":{"type":"number","minimum":0},
                                "max_width":{"type":"number","exclusiveMinimum":0},
                                "max_height":{"type":"number","exclusiveMinimum":0},
                                "flex":{"type":"number","exclusiveMinimum":0}
                            },
                            "additionalProperties":false
                        },
                        "split":{"type":"object","properties":{"axis":{"type":"string","enum":["horizontal","vertical"],"default":"horizontal"},"ratio":{"type":"number","minimum":0.05,"maximum":0.95},"resizable":{"type":"boolean"}},"required":["ratio"],"additionalProperties":false},
                        "collapsed":{"type":"boolean","default":false}
                        ,"configuration":{"type":"object","default":{},"maxProperties":256}
                        ,"state_bindings":{
                            "type":"object",
                            "properties":{
                                "visible":{
                                    "type":"object",
                                    "properties":{
                                        "type":{"const":"command_state"},
                                        "command_id":{"type":"string","minLength":1,"maxLength":256},
                                        "state":{"type":"string","enum":["visible","enabled","checked"]},
                                        "equals":{"type":"boolean","default":true}
                                    },
                                    "required":["type","command_id","state"],
                                    "additionalProperties":false
                                }
                            },
                            "additionalProperties":false
                        }
                        ,"ownership":{
                            "readOnly":true,
                            "type":"object",
                            "properties":{
                                "scope":{"type":"string","enum":["application","extension"]},
                                "owner_id":{"type":"string","minLength":1},
                                "owner_session_id":{"type":"string","minLength":1},
                                "protected":{"type":"boolean"}
                            },
                            "required":["scope","owner_id","protected"],
                            "additionalProperties":false
                        }
                        ,"readiness":{
                            "readOnly":true,
                            "type":"object",
                            "properties":{
                                "state":{"type":"string","enum":["ready","not_ready","disconnected","incompatible","missing"]},
                                "reason":{"type":["string","null"]},
                                "expected_extension_version":{"type":["string","null"]},
                                "current_extension_version":{"type":["string","null"]}
                            },
                            "required":["state","reason","expected_extension_version","current_extension_version"],
                            "additionalProperties":false
                        }
                    },
                    "required":["id","type"],
                    "additionalProperties":false
                }
            }
        },
        "required":["root_id","nodes"],
        "additionalProperties":false
    })
}

fn extension_owner_id(mount: &str) -> Option<&str> {
    mount.strip_prefix("extension:")?.split('/').next()
}

fn protected_mount(mount: &str) -> bool {
    matches!(
        mount,
        "builtin:project-workspace" | "builtin:viewer-canvas" | "builtin:mosaic-canvas"
    )
}

pub(super) fn component_catalog(mode: Option<&str>) -> Value {
    let components = component_descriptors()
        .into_iter()
        .filter(|component| {
            mode.is_none_or(|mode| {
                component["modes"]
                    .as_array()
                    .is_some_and(|modes| modes.iter().any(|candidate| candidate == mode))
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version":1,
        "mode":mode,
        "components":components,
    })
}

pub(super) fn component_catalog_schema() -> Value {
    json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-component-catalog-v1.json",
        "type":"object",
        "properties":{
            "schema_version":{"const":1},
            "mode":{"type":["string","null"],"enum":["project","single","mosaic",null]},
            "components":{
                "type":"array",
                "items":{
                    "type":"object",
                    "properties":{
                        "id":{"type":"string","minLength":1},
                        "version":{"type":"integer","minimum":1},
                        "title":{"type":"string","minLength":1},
                        "kind":{"type":"string","enum":["toolbar","workspace","canvas","panel"]},
                        "modes":{"type":"array","items":{"type":"string","enum":["project","single","mosaic"]},"uniqueItems":true},
                        "readiness":{"type":"array","items":{"type":"string"},"uniqueItems":true},
                        "legal_parent_types":{"type":"array","items":{"type":"string"},"uniqueItems":true},
                        "singleton":{"type":"boolean"},
                        "configuration_schema":{"type":"object"},
                        "commands":{"type":"array","items":{"type":"string"},"uniqueItems":true},
                        "events":{"type":"array","items":{"type":"string"},"uniqueItems":true},
                        "minimum_size":{"$ref":"#/$defs/size"},
                        "recommended_size":{"$ref":"#/$defs/size"},
                        "persistence":{"type":"string","enum":["session","user","project"]},
                        "ownership":{
                            "type":"object",
                            "properties":{
                                "scope":{"type":"string","enum":["application","extension"]},
                                "owner_id":{"type":"string","minLength":1},
                                "owner_session_id":{"type":["string","null"]},
                                "protected":{"type":"boolean"}
                            },
                            "required":["scope","owner_id","owner_session_id","protected"],
                            "additionalProperties":false
                        }
                    },
                    "required":["id","version","title","kind","modes","readiness","legal_parent_types","singleton","configuration_schema","commands","events","minimum_size","recommended_size","persistence","ownership"],
                    "additionalProperties":false
                }
            }
        },
        "required":["schema_version","mode","components"],
        "additionalProperties":false,
        "$defs":{
            "size":{
                "type":"object",
                "properties":{"width":{"type":"number","minimum":0},"height":{"type":"number","minimum":0}},
                "required":["width","height"],
                "additionalProperties":false
            }
        }
    })
}

fn component_descriptors() -> Vec<Value> {
    vec![
        component(
            "builtin:project-top-bar",
            "Project toolbar",
            "toolbar",
            &["project"],
            &["application", "toolbar"],
            [0.0, 28.0],
            [800.0, 36.0],
            &["app.navigation.show_project"],
            &[],
            "session",
        ),
        component(
            "builtin:project-workspace",
            "Project browser",
            "workspace",
            &["project"],
            &["application", "row", "column", "split", "panel"],
            [320.0, 240.0],
            [1000.0, 700.0],
            &["project.rois.open", "project.rois.select"],
            &["project.rois.selection_changed"],
            "project",
        ),
        component(
            "builtin:viewer-top-bar",
            "Viewer toolbar",
            "toolbar",
            &["single"],
            &["application", "toolbar"],
            [0.0, 28.0],
            [1000.0, 36.0],
            &["viewer.viewports.camera.fit", "viewer.panels.set"],
            &["viewer.ui.changed"],
            "project",
        ),
        component(
            "builtin:viewer-canvas",
            "Image viewport workspace",
            "canvas",
            &["single"],
            &["application", "row", "column", "split", "panel", "tabs"],
            [256.0, 256.0],
            [1200.0, 800.0],
            &["viewer.viewports.camera.set", "viewer.workspace.layout.set"],
            &["viewer.camera.changed", "viewer.workspace.layout.changed"],
            "project",
        ),
        component(
            "builtin:mosaic-top-bar",
            "Mosaic toolbar",
            "toolbar",
            &["mosaic"],
            &["application", "toolbar"],
            [0.0, 28.0],
            [1000.0, 36.0],
            &[
                "mosaic.fit_all",
                "mosaic.ui.set_left_tab",
                "mosaic.ui.set_right_tab",
            ],
            &["mosaic.ui.changed"],
            "project",
        ),
        component(
            "builtin:mosaic-canvas",
            "Mosaic canvas",
            "canvas",
            &["mosaic"],
            &["application", "row", "column", "split", "panel", "tabs"],
            [256.0, 256.0],
            [1200.0, 800.0],
            &["mosaic.focus.set", "mosaic.layout.configure"],
            &["viewer.camera.changed", "mosaic.layout.changed"],
            "project",
        ),
        extension_host_component(
            "builtin:extension-host.top-bar-actions",
            "Extension toolbar actions",
            "toolbar",
            &["project", "single", "mosaic"],
            &["application", "toolbar", "row"],
            [0.0, 28.0],
            [320.0, 36.0],
        ),
        extension_host_component(
            "builtin:extension-host.status-bar",
            "Extension status items",
            "toolbar",
            &["project", "single", "mosaic"],
            &["application", "status_bar", "row"],
            [0.0, 20.0],
            [600.0, 24.0],
        ),
        extension_host_component(
            "builtin:extension-host.left-sections",
            "Extension left sections",
            "panel",
            &["single", "mosaic"],
            &["tabs", "panel", "collapsible", "row", "column", "split"],
            [180.0, 80.0],
            [300.0, 500.0],
        ),
        extension_host_component(
            "builtin:extension-host.right-tabs",
            "Extension right panels",
            "panel",
            &["single", "mosaic"],
            &["tabs", "panel", "collapsible", "row", "column", "split"],
            [180.0, 80.0],
            [320.0, 500.0],
        ),
        extension_host_component(
            "builtin:extension-host.canvas-controls",
            "Extension canvas controls",
            "toolbar",
            &["single", "mosaic"],
            &["toolbar", "panel", "row", "column"],
            [0.0, 28.0],
            [500.0, 36.0],
        ),
        extension_host_component(
            "builtin:extension-host.project-cards",
            "Extension project cards",
            "panel",
            &["project"],
            &["panel", "collapsible", "row", "column", "split"],
            [240.0, 180.0],
            [320.0, 600.0],
        ),
        component(
            "builtin:shell-inspector",
            "Application shell inspector",
            "panel",
            &["project", "single", "mosaic"],
            &["tabs", "panel", "collapsible", "row", "column", "split"],
            [320.0, 180.0],
            [720.0, 520.0],
            &[],
            &["ui.shell.changed"],
            "session",
        ),
        component(
            "builtin:command-toolbar",
            "Application command toolbar",
            "toolbar",
            &["project", "single", "mosaic"],
            &["application", "toolbar", "row", "column"],
            [0.0, 28.0],
            [800.0, 36.0],
            &["ui.commands.execute"],
            &["ui.toolbars.changed", "ui.commands.changed"],
            "session",
        ),
        component(
            "builtin:help",
            "Odon documentation",
            "panel",
            &["project", "single", "mosaic"],
            &["tabs", "panel", "collapsible", "row", "column", "split"],
            [320.0, 240.0],
            [760.0, 520.0],
            &[],
            &[],
            "session",
        ),
        component(
            "builtin:recovery-controls",
            "Application layout recovery",
            "panel",
            &["project", "single", "mosaic"],
            &["tabs", "panel", "collapsible", "row", "column", "split"],
            [260.0, 120.0],
            [420.0, 180.0],
            &["ui.shell.recover"],
            &["ui.shell.changed"],
            "session",
        ),
        component(
            "builtin:channels",
            "Channels",
            "panel",
            &["single", "mosaic"],
            &["tabs", "panel", "collapsible", "row", "column", "split"],
            [220.0, 160.0],
            [360.0, 600.0],
            &[
                "viewer.channels.set_active",
                "viewer.channels.set_visible",
                "viewer.channels.presentation.set",
            ],
            &[
                "viewer.channels.changed",
                "viewer.channels.presentation.changed",
            ],
            "project",
        ),
        component(
            "builtin:viewer-viewport-controls",
            "Viewport controls",
            "toolbar",
            &["single"],
            &["application", "toolbar", "panel", "row", "column"],
            [0.0, 28.0],
            [360.0, 36.0],
            &[
                "viewer.viewports.clone",
                "viewer.viewports.remove",
                "viewer.workspace.layout.set",
                "viewer.workspace.swap",
            ],
            &["viewer.workspace.layout.changed"],
            "project",
        ),
        panel_component(
            "builtin:layers",
            "Layers",
            &["single", "mosaic"],
            &["viewer.viewports.layers.set", "mosaic.rendering.set"],
            &[
                "viewer.viewports.presentation.changed",
                "viewer.rendering.changed",
            ],
        ),
        panel_component(
            "builtin:project",
            "Project",
            &["single", "mosaic"],
            &["project.rois.open", "project.views.apply"],
            &["project.rois.selection_changed", "project.views.changed"],
        ),
        panel_component(
            "builtin:properties",
            "Properties",
            &["single", "mosaic"],
            &["viewer.viewports.layers.set", "mosaic.rendering.set"],
            &[
                "viewer.viewports.presentation.changed",
                "viewer.rendering.changed",
            ],
        ),
        panel_component(
            "builtin:views",
            "Saved views",
            &["single", "mosaic"],
            &["project.views.create", "project.views.apply"],
            &["project.views.changed"],
        ),
        panel_component(
            "builtin:analysis",
            "Analysis",
            &["single"],
            &["viewer.analysis.set", "viewer.analysis.suggest_thresholds"],
            &["viewer.analysis.changed"],
        ),
        panel_component(
            "builtin:measurements",
            "Measurements",
            &["single"],
            &["viewer.measurements.configure", "viewer.measurements.start"],
            &["viewer.measurements.changed"],
        ),
        panel_component(
            "builtin:memory",
            "Memory",
            &["single", "mosaic"],
            &["memory.pin", "memory.unpin"],
            &["memory.changed"],
        ),
        panel_component(
            "builtin:roi-selector",
            "ROI selection",
            &["single"],
            &["project.rois.select", "project.rois.open"],
            &["project.rois.selection_changed"],
        ),
        panel_component(
            "builtin:mosaic-layout",
            "Mosaic layout",
            &["mosaic"],
            &["mosaic.layout.configure"],
            &["mosaic.layout.changed"],
        ),
    ]
}

fn extension_host_component(
    id: &str,
    title: &str,
    kind: &str,
    modes: &[&str],
    legal_parents: &[&str],
    minimum_size: [f32; 2],
    recommended_size: [f32; 2],
) -> Value {
    component(
        id,
        title,
        kind,
        modes,
        legal_parents,
        minimum_size,
        recommended_size,
        &[],
        &["ui.contributions.registered", "ui.extensions.disconnected"],
        "session",
    )
}

fn panel_component(
    id: &str,
    title: &str,
    modes: &[&str],
    commands: &[&str],
    events: &[&str],
) -> Value {
    component(
        id,
        title,
        "panel",
        modes,
        &["tabs", "panel", "collapsible", "row", "column", "split"],
        [220.0, 120.0],
        [360.0, 600.0],
        commands,
        events,
        "project",
    )
}

#[allow(clippy::too_many_arguments)]
fn component(
    id: &str,
    title: &str,
    kind: &str,
    modes: &[&str],
    legal_parents: &[&str],
    minimum_size: [f32; 2],
    recommended_size: [f32; 2],
    commands: &[&str],
    events: &[&str],
    persistence: &str,
) -> Value {
    json!({
        "id":id,
        "version":1,
        "title":title,
        "kind":kind,
        "modes":modes,
        "readiness":["model"],
        "legal_parent_types":legal_parents,
        "singleton":true,
        "configuration_schema":component_configuration_schema(id),
        "commands":commands,
        "events":events,
        "minimum_size":{"width":minimum_size[0],"height":minimum_size[1]},
        "recommended_size":{"width":recommended_size[0],"height":recommended_size[1]},
        "persistence":persistence,
        "ownership":{
            "scope":"application",
            "owner_id":"odon",
            "owner_session_id":Value::Null,
            "protected":protected_mount(id),
        },
    })
}

fn component_configuration_schema(id: &str) -> Value {
    let properties = match id {
        "builtin:project-top-bar" => json!({
            "show_title":{"type":"boolean","default":true},
        }),
        "builtin:viewer-top-bar" => json!({
            "show_title":{"type":"boolean","default":true},
            "show_navigation":{"type":"boolean","default":true},
            "show_panel_controls":{"type":"boolean","default":true},
            "show_viewport_controls":{"type":"boolean","default":true},
            "show_rendering_controls":{"type":"boolean","default":true},
            "show_contrast_controls":{"type":"boolean","default":true},
        }),
        "builtin:mosaic-top-bar" => json!({
            "show_title":{"type":"boolean","default":true},
            "show_navigation":{"type":"boolean","default":true},
            "show_status":{"type":"boolean","default":true},
            "show_panel_controls":{"type":"boolean","default":true},
            "show_rendering_controls":{"type":"boolean","default":true},
            "show_contrast_controls":{"type":"boolean","default":true},
        }),
        _ => json!({}),
    };
    json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "type":"object",
        "properties":properties,
        "additionalProperties":false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_round_trip() {
        for mode in ["project", "single", "mosaic"] {
            let layout = default_layout(mode);
            let reparsed = ShellLayout::from_value(&layout.to_json(), mode, "test").unwrap();
            assert_eq!(layout, reparsed);
        }
    }

    #[test]
    fn desired_tree_rejects_cycles_duplicates_and_missing_canvas() {
        let cycle = json!({"root_id":"layout:root","nodes":[
            {"id":"layout:root","type":"application","children":["layout:a"]},
            {"id":"layout:a","type":"row","children":["layout:root"]}
        ]});
        assert!(ShellLayout::from_value(&cycle, "single", "test").is_err());

        let missing_canvas = json!({"root_id":"layout:root","nodes":[
            {"id":"layout:root","type":"application","children":["layout:layers"]},
            {"id":"layout:layers","type":"builtin_mount","mount":"builtin:layers"}
        ]});
        assert!(ShellLayout::from_value(&missing_canvas, "single", "test").is_err());

        let duplicate_mount = json!({"root_id":"layout:root","nodes":[
            {"id":"layout:root","type":"application","children":["layout:row"]},
            {"id":"layout:row","type":"row","children":["layout:canvas-a","layout:canvas-b"]},
            {"id":"layout:canvas-a","type":"canvas_slot","mount":"builtin:viewer-canvas"},
            {"id":"layout:canvas-b","type":"canvas_slot","mount":"builtin:viewer-canvas"}
        ]});
        assert!(ShellLayout::from_value(&duplicate_mount, "single", "test").is_err());
    }

    #[test]
    fn desired_tree_enforces_catalogue_parent_and_component_kind_rules() {
        let illegal_parent = json!({"root_id":"layout:root","nodes":[
            {"id":"layout:root","type":"application","children":["layout:layers","layout:row"]},
            {"id":"layout:layers","type":"builtin_mount","mount":"builtin:layers"},
            {"id":"layout:row","type":"row","children":["layout:canvas"]},
            {"id":"layout:canvas","type":"canvas_slot","mount":"builtin:viewer-canvas"}
        ]});
        let error = ShellLayout::from_value(&illegal_parent, "single", "test").unwrap_err();
        assert!(
            error
                .message
                .contains("builtin:layers' cannot be placed in parent type 'application'")
        );

        let wrong_mount_kind = json!({"root_id":"layout:root","nodes":[
            {"id":"layout:root","type":"application","children":["layout:row"]},
            {"id":"layout:row","type":"row","children":["layout:canvas"]},
            {"id":"layout:canvas","type":"builtin_mount","mount":"builtin:viewer-canvas"}
        ]});
        let error = ShellLayout::from_value(&wrong_mount_kind, "single", "test").unwrap_err();
        assert!(
            error
                .message
                .contains("canvas component 'builtin:viewer-canvas' requires a canvas_slot")
        );
    }

    #[test]
    fn every_catalogued_builtin_mount_validates_in_every_advertised_legal_parent() {
        for mode in ["project", "single", "mosaic"] {
            for component in component_descriptors().into_iter().filter(|component| {
                component["modes"]
                    .as_array()
                    .is_some_and(|modes| modes.iter().any(|candidate| candidate == mode))
            }) {
                let mount = component["id"].as_str().unwrap();
                let kind = component["kind"].as_str().unwrap();
                for parent in component["legal_parent_types"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(Value::as_str)
                    .map(Option::unwrap)
                {
                    let layout = legal_parent_fixture(mode, mount, kind, parent);
                    ShellLayout::from_value(&layout, mode, "catalogue conformance").unwrap_or_else(
                        |error| {
                            panic!(
                                "{mode} mount {mount} failed in advertised parent {parent}: {error}"
                            )
                        },
                    );
                }
            }
        }
    }

    fn legal_parent_fixture(mode: &str, mount: &str, kind: &str, parent: &str) -> Value {
        let required = required_mount(mode);
        let mount_type = if kind == "canvas" {
            "canvas_slot"
        } else {
            "builtin_mount"
        };
        let test_node = json!({"id":"layout:test.mount","type":mount_type,"mount":mount});
        if parent == "application" {
            let mut children = vec![json!("layout:test.mount")];
            let mut nodes = vec![test_node];
            if mount != required {
                children.push(json!("layout:required"));
                nodes.push(required_mount_node(mode));
            }
            nodes.insert(
                0,
                json!({"id":"layout:root","type":"application","children":children}),
            );
            return json!({"root_id":"layout:root","nodes":nodes});
        }

        let mut parent_children = vec![json!("layout:test.mount")];
        let mut root_children = vec![json!("layout:test.parent")];
        let mut nodes = vec![test_node];
        if parent == "split" {
            if mount == required {
                parent_children.push(json!("layout:filler"));
                nodes.push(json!({
                    "id":"layout:filler",
                    "type":"builtin_mount",
                    "mount":"builtin:help",
                }));
            } else {
                parent_children.push(json!("layout:required"));
                nodes.push(required_mount_node(mode));
            }
        } else if mount != required {
            root_children.push(json!("layout:required"));
            nodes.push(required_mount_node(mode));
        }
        let mut parent_node = json!({
            "id":"layout:test.parent",
            "type":parent,
            "children":parent_children,
        });
        if parent == "tabs" {
            parent_node["selected_id"] = json!("layout:test.mount");
        }
        if parent == "split" {
            parent_node["split"] = json!({"axis":"horizontal","ratio":0.5,"resizable":true});
        }
        nodes.insert(0, parent_node);
        nodes.insert(
            0,
            json!({"id":"layout:root","type":"application","children":root_children}),
        );
        json!({"root_id":"layout:root","nodes":nodes})
    }

    fn required_mount_node(mode: &str) -> Value {
        let mount = required_mount(mode);
        let kind = if mode == "project" {
            "builtin_mount"
        } else {
            "canvas_slot"
        };
        json!({"id":"layout:required","type":kind,"mount":mount})
    }

    #[test]
    fn desired_tree_bounds_retained_mount_configuration() {
        let mut oversized = default_layout("single").to_json();
        let status = oversized["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["id"] == "layout:single.status")
            .unwrap();
        status["configuration"] = json!({
            "values":(0..100)
                .map(|index| format!("{index:03}-{}", "x".repeat(180)))
                .collect::<Vec<_>>()
        });
        let error = ShellLayout::from_value(&oversized, "single", "test").unwrap_err();
        assert!(error.message.contains("per-node limit"));

        let mut excessive_depth = default_layout("single").to_json();
        let mut nested = json!(true);
        for _ in 0..MAX_CONFIGURATION_DEPTH {
            nested = json!({"nested":nested});
        }
        let status = excessive_depth["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["id"] == "layout:single.status")
            .unwrap();
        status["configuration"] = nested;
        let error = ShellLayout::from_value(&excessive_depth, "single", "test").unwrap_err();
        assert!(error.message.contains("maximum depth"));
    }

    #[test]
    fn maximum_size_tree_validates_at_the_boundary_and_rejects_one_more_node() {
        let extension_count = MAX_LAYOUT_NODES - 3;
        let mut children = vec![json!("layout:stress.workspace")];
        let mut nodes = vec![
            json!({
                "id":"layout:stress.root",
                "type":"application",
                "children":["layout:stress.column"],
            }),
            json!({
                "id":"layout:stress.column",
                "type":"column",
                "children":[],
            }),
            json!({
                "id":"layout:stress.workspace",
                "type":"builtin_mount",
                "mount":"builtin:project-workspace",
            }),
        ];
        for index in 0..extension_count {
            let id = format!("layout:stress.extension.{index}");
            children.push(json!(id));
            nodes.push(json!({
                "id":id,
                "type":"extension_mount",
                "mount":format!("extension:org.example.stress/panel-{index}"),
            }));
        }
        nodes[1]["children"] = Value::Array(children);
        let boundary = json!({"root_id":"layout:stress.root","nodes":nodes});
        assert_eq!(
            boundary["nodes"].as_array().unwrap().len(),
            MAX_LAYOUT_NODES
        );
        ShellLayout::from_value(&boundary, "project", "maximum-size test").unwrap();

        const VALIDATION_SAMPLES: u32 = 64;
        let started = std::time::Instant::now();
        for _ in 0..VALIDATION_SAMPLES {
            ShellLayout::from_value(&boundary, "project", "maximum-size timing").unwrap();
        }
        let elapsed = started.elapsed();
        println!(
            "maximum shell tree validation: nodes={MAX_LAYOUT_NODES} samples={VALIDATION_SAMPLES} total_us={} average_us={}",
            elapsed.as_micros(),
            elapsed.as_micros() / u128::from(VALIDATION_SAMPLES),
        );
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "{VALIDATION_SAMPLES} maximum-tree validations took {elapsed:?}; the actor must not stall under boundary input"
        );

        let mut oversized = boundary;
        oversized["nodes"].as_array_mut().unwrap().push(json!({
            "id":"layout:stress.too-many",
            "type":"extension_mount",
            "mount":"extension:org.example.stress/too-many",
        }));
        let error =
            ShellLayout::from_value(&oversized, "project", "maximum-size test").unwrap_err();
        assert_eq!(error.kind, crate::control::ControlErrorKind::InvalidParams);
        assert!(error.message.contains("256-node limit"));
    }
}
