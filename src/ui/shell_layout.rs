//! Renderer-side keyed reconciliation for actor-owned shell layout projections.
//!
//! The actor validates topology. This layer deliberately treats the projection as data and keeps
//! renderer-local state keyed by stable node ID, so replacing a tree does not recreate state for
//! nodes whose keys survive the transaction.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub(crate) struct ShellLayoutReconciler {
    generation: u64,
    nodes: BTreeMap<String, ReconciledNode>,
    last_change: ReconciliationChange,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Geometry state is populated by the recursive layout renderer in this milestone.
struct ReconciledNode {
    descriptor: Value,
    persistent_token: u64,
    last_rect: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ReconciliationChange {
    pub(crate) generation: u64,
    pub(crate) added: Vec<String>,
    pub(crate) removed: Vec<String>,
    pub(crate) retained: Vec<String>,
    pub(crate) updated: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelSide {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativePanelPlan {
    pub(crate) id: String,
    pub(crate) side: PanelSide,
    pub(crate) width: Option<f32>,
    pub(crate) height: Option<f32>,
    pub(crate) min_width: Option<f32>,
    pub(crate) min_height: Option<f32>,
    pub(crate) max_width: Option<f32>,
    pub(crate) max_height: Option<f32>,
    pub(crate) flex: Option<f32>,
    pub(crate) mounts: Vec<String>,
    pub(crate) selected_mount: Option<String>,
    pub(crate) resizable: bool,
}

impl ShellLayoutReconciler {
    pub(crate) fn apply_projection(&mut self, shell: &Value) -> &ReconciliationChange {
        let incoming = shell
            .pointer("/layout/nodes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|node| {
                node.get("id")
                    .and_then(Value::as_str)
                    .map(|id| (id.to_string(), node.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let before = self.nodes.keys().cloned().collect::<BTreeSet<_>>();
        let after = incoming.keys().cloned().collect::<BTreeSet<_>>();
        self.generation = self.generation.wrapping_add(1).max(1);

        let mut next = BTreeMap::new();
        let mut updated = Vec::new();
        for (id, descriptor) in incoming {
            if let Some(mut retained) = self.nodes.remove(&id) {
                if retained.descriptor != descriptor {
                    updated.push(id.clone());
                    retained.descriptor = descriptor;
                }
                next.insert(id, retained);
            } else {
                next.insert(
                    id,
                    ReconciledNode {
                        descriptor,
                        persistent_token: self.generation,
                        last_rect: None,
                    },
                );
            }
        }
        self.nodes = next;
        self.last_change = ReconciliationChange {
            generation: self.generation,
            added: after.difference(&before).cloned().collect(),
            removed: before.difference(&after).cloned().collect(),
            retained: before.intersection(&after).cloned().collect(),
            updated,
        };
        &self.last_change
    }

    #[allow(dead_code)]
    pub(crate) fn persistent_token(&self, id: &str) -> Option<u64> {
        self.nodes.get(id).map(|node| node.persistent_token)
    }

    #[allow(dead_code)]
    pub(crate) fn record_rect(&mut self, id: &str, rect: [f32; 4]) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.last_rect = Some(rect);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rect(&self, id: &str) -> Option<[f32; 4]> {
        self.nodes.get(id).and_then(|node| node.last_rect)
    }

    pub(crate) fn mount_visible(&self, shell: &Value, mount: &str) -> Option<bool> {
        projected_mount_visible(shell, mount)
    }
}

pub(crate) fn projected_mount_visible(shell: &Value, mount: &str) -> Option<bool> {
    let nodes = shell.pointer("/layout/nodes")?.as_array()?;
    let by_id = nodes
        .iter()
        .filter_map(|node| node.get("id").and_then(Value::as_str).map(|id| (id, node)))
        .collect::<BTreeMap<_, _>>();
    let Some(mut node) = nodes
        .iter()
        .find(|node| node.get("mount").and_then(Value::as_str) == Some(mount))
    else {
        return Some(false);
    };
    loop {
        if node.get("visible").and_then(Value::as_bool) == Some(false) {
            return Some(false);
        }
        let Some(parent) = node.get("parent_id").and_then(Value::as_str) else {
            return Some(true);
        };
        node = *by_id.get(parent)?;
    }
}

pub(crate) fn native_panel_plans(shell: &Value, canvas_mount: &str) -> Vec<NativePanelPlan> {
    let Some(layout) = shell.get("layout") else {
        return Vec::new();
    };
    let Some(nodes) = layout.get("nodes").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(root_id) = layout.get("root_id").and_then(Value::as_str) else {
        return Vec::new();
    };
    let by_id = nodes
        .iter()
        .filter_map(|node| node.get("id").and_then(Value::as_str).map(|id| (id, node)))
        .collect::<BTreeMap<_, _>>();
    let Some(canvas_id) = nodes.iter().find_map(|node| {
        (node.get("mount").and_then(Value::as_str) == Some(canvas_mount))
            .then(|| node.get("id").and_then(Value::as_str))
            .flatten()
    }) else {
        return Vec::new();
    };
    let canvas_path = path_from_root(root_id, canvas_id, &by_id);
    let mut traversal = Vec::new();
    collect_node_order(root_id, &by_id, &mut traversal);
    let order = traversal
        .into_iter()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<BTreeMap<_, _>>();

    let mut plans = nodes
        .iter()
        .filter(|node| {
            matches!(
                node.get("type").and_then(Value::as_str),
                Some("panel" | "collapsible")
            )
        })
        .filter_map(|node| {
            let id = node.get("id")?.as_str()?;
            if !node_and_ancestors_visible(node, &by_id) {
                return None;
            }
            let mounts = visible_mounts_below(id, &by_id);
            if mounts.is_empty() || mounts.iter().any(|mount| mount == canvas_mount) {
                return None;
            }
            let panel_path = path_from_root(root_id, id, &by_id);
            let (side, resizable) = relative_side(&panel_path, &canvas_path, &by_id)?;
            let size = node.get("size");
            Some(NativePanelPlan {
                id: id.to_string(),
                side,
                width: size.and_then(|size| size.get("width")).and_then(value_f32),
                height: size.and_then(|size| size.get("height")).and_then(value_f32),
                min_width: size
                    .and_then(|size| size.get("min_width"))
                    .and_then(value_f32),
                min_height: size
                    .and_then(|size| size.get("min_height"))
                    .and_then(value_f32),
                max_width: size
                    .and_then(|size| size.get("max_width"))
                    .and_then(value_f32),
                max_height: size
                    .and_then(|size| size.get("max_height"))
                    .and_then(value_f32),
                flex: size.and_then(|size| size.get("flex")).and_then(value_f32),
                selected_mount: selected_mount_below(id, &by_id),
                mounts,
                resizable,
            })
        })
        .collect::<Vec<_>>();
    plans.sort_by_key(|plan| order.get(plan.id.as_str()).copied().unwrap_or(usize::MAX));
    plans
}

fn collect_node_order<'a>(
    id: &'a str,
    nodes: &BTreeMap<&'a str, &'a Value>,
    order: &mut Vec<&'a str>,
) {
    let Some(node) = nodes.get(id) else {
        return;
    };
    order.push(id);
    if let Some(children) = node.get("children").and_then(Value::as_array) {
        for child in children.iter().filter_map(Value::as_str) {
            collect_node_order(child, nodes, order);
        }
    }
}

fn path_from_root<'a>(
    root_id: &'a str,
    id: &'a str,
    nodes: &BTreeMap<&'a str, &'a Value>,
) -> Vec<&'a str> {
    let mut reversed = vec![id];
    let mut current = id;
    while current != root_id {
        let Some(parent) = nodes
            .get(current)
            .and_then(|node| node.get("parent_id"))
            .and_then(Value::as_str)
        else {
            return Vec::new();
        };
        reversed.push(parent);
        current = parent;
    }
    reversed.reverse();
    reversed
}

fn relative_side(
    panel: &[&str],
    canvas: &[&str],
    nodes: &BTreeMap<&str, &Value>,
) -> Option<(PanelSide, bool)> {
    if panel.is_empty() || canvas.is_empty() {
        return None;
    }
    let divergence = panel
        .iter()
        .zip(canvas)
        .position(|(left, right)| left != right)?;
    if divergence == 0 {
        return None;
    }
    let parent_id = panel[divergence - 1];
    let parent = *nodes.get(parent_id)?;
    let children = parent.get("children")?.as_array()?;
    let panel_index = children
        .iter()
        .position(|child| child.as_str() == Some(panel[divergence]))?;
    let canvas_index = children
        .iter()
        .position(|child| child.as_str() == Some(canvas[divergence]))?;
    let before = panel_index < canvas_index;
    match parent.get("type").and_then(Value::as_str)? {
        "row" => Some((
            if before {
                PanelSide::Left
            } else {
                PanelSide::Right
            },
            true,
        )),
        "column" => Some((
            if before {
                PanelSide::Top
            } else {
                PanelSide::Bottom
            },
            true,
        )),
        "split" => {
            let side = match parent.pointer("/split/axis").and_then(Value::as_str) {
                Some("vertical") => {
                    if before {
                        PanelSide::Top
                    } else {
                        PanelSide::Bottom
                    }
                }
                _ => {
                    if before {
                        PanelSide::Left
                    } else {
                        PanelSide::Right
                    }
                }
            };
            let resizable = parent
                .pointer("/split/resizable")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            Some((side, resizable))
        }
        _ => Some((
            if before {
                PanelSide::Left
            } else {
                PanelSide::Right
            },
            true,
        )),
    }
}

fn visible_mounts_below(id: &str, nodes: &BTreeMap<&str, &Value>) -> Vec<String> {
    let mut mounts = Vec::new();
    collect_mounts(id, nodes, true, &mut mounts);
    mounts
}

fn collect_mounts(
    id: &str,
    nodes: &BTreeMap<&str, &Value>,
    ancestors_visible: bool,
    mounts: &mut Vec<String>,
) {
    let Some(node) = nodes.get(id) else {
        return;
    };
    let visible = ancestors_visible
        && node.get("visible").and_then(Value::as_bool).unwrap_or(true)
        && !(node.get("type").and_then(Value::as_str) == Some("collapsible")
            && node.get("collapsed").and_then(Value::as_bool) == Some(true));
    if !visible {
        return;
    }
    if let Some(mount) = node.get("mount").and_then(Value::as_str) {
        mounts.push(mount.to_string());
    }
    if let Some(children) = node.get("children").and_then(Value::as_array) {
        for child in children.iter().filter_map(Value::as_str) {
            collect_mounts(child, nodes, visible, mounts);
        }
    }
}

fn selected_mount_below(id: &str, nodes: &BTreeMap<&str, &Value>) -> Option<String> {
    let node = *nodes.get(id)?;
    if node.get("type").and_then(Value::as_str) == Some("tabs") {
        let selected = node.get("selected_id").and_then(Value::as_str)?;
        return visible_mounts_below(selected, nodes).into_iter().next();
    }
    node.get("children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find_map(|child| selected_mount_below(child, nodes))
}

fn node_and_ancestors_visible(node: &Value, nodes: &BTreeMap<&str, &Value>) -> bool {
    let mut current = node;
    loop {
        if current.get("visible").and_then(Value::as_bool) == Some(false) {
            return false;
        }
        if current.get("type").and_then(Value::as_str) == Some("collapsible")
            && current.get("collapsed").and_then(Value::as_bool) == Some(true)
        {
            return false;
        }
        let Some(parent) = current.get("parent_id").and_then(Value::as_str) else {
            return true;
        };
        let Some(parent) = nodes.get(parent) else {
            return false;
        };
        current = parent;
    }
}

fn value_f32(value: &Value) -> Option<f32> {
    value
        .as_f64()
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keyed_reconciliation_retains_local_state_for_surviving_nodes() {
        let first = json!({"layout":{"nodes":[
            {"id":"layout:root","type":"application","parent_id":null,"visible":true},
            {"id":"layout:panel","type":"panel","parent_id":"layout:root","visible":true}
        ]}});
        let second = json!({"layout":{"nodes":[
            {"id":"layout:root","type":"application","parent_id":null,"visible":true},
            {"id":"layout:panel","type":"panel","parent_id":"layout:root","visible":false},
            {"id":"layout:canvas","type":"canvas_slot","parent_id":"layout:root","visible":true}
        ]}});
        let mut reconciler = ShellLayoutReconciler::default();
        reconciler.apply_projection(&first);
        let token = reconciler.persistent_token("layout:panel").unwrap();
        reconciler.record_rect("layout:panel", [1.0, 2.0, 3.0, 4.0]);
        let change = reconciler.apply_projection(&second).clone();

        assert_eq!(change.added, vec!["layout:canvas"]);
        assert_eq!(change.retained, vec!["layout:panel", "layout:root"]);
        assert_eq!(change.updated, vec!["layout:panel"]);
        assert_eq!(reconciler.persistent_token("layout:panel"), Some(token));
        assert_eq!(reconciler.rect("layout:panel"), Some([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(reconciler.mount_visible(&second, "missing"), Some(false));
    }

    #[test]
    fn arbitrary_rows_columns_and_tabs_resolve_to_native_panel_regions() {
        let shell = json!({"layout":{
            "root_id":"layout:root",
            "nodes":[
                {"id":"layout:root","type":"application","parent_id":null,"children":["layout:body"],"visible":true},
                {"id":"layout:body","type":"row","parent_id":"layout:root","children":["layout:left","layout:center","layout:right"],"visible":true},
                {"id":"layout:left","type":"panel","parent_id":"layout:body","children":["layout:left-tabs"],"visible":true,"size":{"width":300.0,"min_width":220.0}},
                {"id":"layout:left-tabs","type":"tabs","parent_id":"layout:left","children":["layout:layers","layout:project"],"selected_id":"layout:project","visible":true},
                {"id":"layout:layers","type":"builtin_mount","parent_id":"layout:left-tabs","children":[],"mount":"builtin:layers","visible":true},
                {"id":"layout:project","type":"builtin_mount","parent_id":"layout:left-tabs","children":[],"mount":"builtin:project","visible":true},
                {"id":"layout:center","type":"column","parent_id":"layout:body","children":["layout:top","layout:canvas","layout:bottom"],"visible":true},
                {"id":"layout:top","type":"panel","parent_id":"layout:center","children":["layout:views"],"visible":true,"size":{"height":140.0}},
                {"id":"layout:views","type":"builtin_mount","parent_id":"layout:top","children":[],"mount":"builtin:views","visible":true},
                {"id":"layout:canvas","type":"canvas_slot","parent_id":"layout:center","children":[],"mount":"builtin:viewer-canvas","visible":true},
                {"id":"layout:bottom","type":"panel","parent_id":"layout:center","children":["layout:memory"],"visible":true,"size":{"height":120.0}},
                {"id":"layout:memory","type":"builtin_mount","parent_id":"layout:bottom","children":[],"mount":"builtin:memory","visible":true},
                {"id":"layout:right","type":"panel","parent_id":"layout:body","children":["layout:properties"],"visible":true,"size":{"width":400.0}},
                {"id":"layout:properties","type":"builtin_mount","parent_id":"layout:right","children":[],"mount":"builtin:properties","visible":true}
            ]
        }});
        let plans = native_panel_plans(&shell, "builtin:viewer-canvas");
        assert_eq!(plans.len(), 4);
        let left = plans
            .iter()
            .find(|plan| plan.side == PanelSide::Left)
            .unwrap();
        assert_eq!(left.id, "layout:left");
        assert_eq!(left.width, Some(300.0));
        assert_eq!(left.min_width, Some(220.0));
        assert_eq!(left.mounts, vec!["builtin:layers", "builtin:project"]);
        assert_eq!(left.selected_mount.as_deref(), Some("builtin:project"));
        assert_eq!(
            plans
                .iter()
                .find(|plan| plan.side == PanelSide::Top)
                .unwrap()
                .height,
            Some(140.0)
        );
        assert_eq!(
            plans
                .iter()
                .find(|plan| plan.side == PanelSide::Bottom)
                .unwrap()
                .height,
            Some(120.0)
        );
        assert_eq!(
            plans
                .iter()
                .find(|plan| plan.side == PanelSide::Right)
                .unwrap()
                .width,
            Some(400.0)
        );
    }
}
