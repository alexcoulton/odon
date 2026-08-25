//! Native diagnostics for the actor-owned application shell.

use std::collections::{BTreeMap, BTreeSet};

use eframe::egui;
use serde_json::Value;

pub(crate) type ShellGeometry = BTreeMap<String, [f32; 4]>;

#[derive(Debug, Clone, PartialEq)]
struct InspectorRow {
    depth: usize,
    id: String,
    kind: String,
    mount: Option<String>,
    visible: bool,
    ownership: String,
    mutability: String,
    readiness: String,
    geometry: Option<[f32; 4]>,
}

#[derive(Debug, Clone, PartialEq)]
struct ShellInspection {
    mode: String,
    revision: u64,
    active_region_id: Option<String>,
    focused_node_id: Option<String>,
    rows: Vec<InspectorRow>,
    problems: Vec<String>,
}

pub(crate) fn render(ui: &mut egui::Ui, shell: &Value, geometry: &ShellGeometry) {
    let inspection = inspect(shell, geometry);
    ui.heading("Application shell inspector");
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Mode: {}", inspection.mode));
        ui.separator();
        ui.label(format!("Revision: {}", inspection.revision));
        ui.separator();
        ui.label(format!(
            "Active: {}",
            inspection.active_region_id.as_deref().unwrap_or("none")
        ));
        ui.separator();
        ui.label(format!(
            "Focus: {}",
            inspection.focused_node_id.as_deref().unwrap_or("none")
        ));
    });
    if inspection.problems.is_empty() {
        ui.colored_label(
            egui::Color32::GREEN,
            "No shell validation or readiness problems.",
        );
    } else {
        ui.colored_label(
            egui::Color32::YELLOW,
            format!("{} problem(s)", inspection.problems.len()),
        );
        for problem in &inspection.problems {
            ui.label(format!("- {problem}"));
        }
    }
    ui.separator();
    egui::ScrollArea::both()
        .id_salt("odon-shell-inspector-table")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new("odon-shell-inspector-grid")
                .striped(true)
                .show(ui, |ui| {
                    for heading in [
                        "Node",
                        "Type",
                        "Mount",
                        "Visible",
                        "Ownership",
                        "Mutation",
                        "Ready",
                        "Geometry",
                    ] {
                        ui.strong(heading);
                    }
                    ui.end_row();
                    for row in &inspection.rows {
                        ui.monospace(format!("{}{}", "  ".repeat(row.depth), row.id));
                        ui.label(&row.kind);
                        ui.monospace(row.mount.as_deref().unwrap_or("-"));
                        ui.label(if row.visible { "yes" } else { "no" });
                        ui.label(&row.ownership);
                        ui.label(&row.mutability);
                        ui.label(&row.readiness);
                        ui.monospace(
                            row.geometry
                                .map(|rect| {
                                    format!(
                                        "{:.0},{:.0} {:.0}x{:.0}",
                                        rect[0],
                                        rect[1],
                                        rect[2] - rect[0],
                                        rect[3] - rect[1]
                                    )
                                })
                                .unwrap_or_else(|| "not presented".to_string()),
                        );
                        ui.end_row();
                    }
                });
        });
}

fn inspect(shell: &Value, geometry: &ShellGeometry) -> ShellInspection {
    let mut inspection = ShellInspection {
        mode: shell
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        revision: shell.get("revision").and_then(Value::as_u64).unwrap_or(0),
        active_region_id: shell
            .get("active_region_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        focused_node_id: shell
            .get("focused_node_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        rows: Vec::new(),
        problems: Vec::new(),
    };
    let Some(layout) = shell.get("layout") else {
        inspection
            .problems
            .push("snapshot has no desired layout".to_string());
        return inspection;
    };
    let root_id = layout
        .get("root_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let Some(nodes) = layout.get("nodes").and_then(Value::as_array) else {
        inspection
            .problems
            .push("desired layout has no node array".to_string());
        return inspection;
    };
    let mut by_id = BTreeMap::new();
    for node in nodes {
        let Some(id) = node.get("id").and_then(Value::as_str) else {
            inspection
                .problems
                .push("desired layout contains a node without an ID".to_string());
            continue;
        };
        if by_id.insert(id.to_string(), node).is_some() {
            inspection
                .problems
                .push(format!("duplicate desired-layout node '{id}'"));
        }
    }
    if root_id.is_empty() || !by_id.contains_key(root_id) {
        inspection
            .problems
            .push("desired layout root is missing".to_string());
        return inspection;
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    visit(
        root_id,
        0,
        &by_id,
        geometry,
        &mut visiting,
        &mut visited,
        &mut inspection,
    );
    for id in by_id.keys().filter(|id| !visited.contains(*id)) {
        inspection
            .problems
            .push(format!("node '{id}' is unreachable from root '{root_id}'"));
    }
    inspection
}

#[allow(clippy::too_many_arguments)]
fn visit(
    id: &str,
    depth: usize,
    nodes: &BTreeMap<String, &Value>,
    geometry: &ShellGeometry,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
    inspection: &mut ShellInspection,
) {
    if !visiting.insert(id.to_string()) {
        inspection
            .problems
            .push(format!("cycle detected at node '{id}'"));
        return;
    }
    let Some(node) = nodes.get(id).copied() else {
        inspection
            .problems
            .push(format!("unknown child node '{id}'"));
        visiting.remove(id);
        return;
    };
    if visited.insert(id.to_string()) {
        let kind = node
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let mount = node
            .get("mount")
            .and_then(Value::as_str)
            .map(str::to_string);
        let ownership = ownership_label(node);
        let readiness = node
            .pointer("/readiness/state")
            .and_then(Value::as_str)
            .unwrap_or("ready")
            .to_string();
        if readiness != "ready" {
            inspection.problems.push(format!(
                "node '{id}' is {readiness}: {}",
                node.pointer("/readiness/reason")
                    .and_then(Value::as_str)
                    .unwrap_or("no reason reported")
            ));
        }
        inspection.rows.push(InspectorRow {
            depth,
            id: id.to_string(),
            kind,
            mount,
            visible: node.get("visible").and_then(Value::as_bool).unwrap_or(true),
            ownership,
            mutability: mutability_label(node),
            readiness,
            geometry: geometry.get(id).copied(),
        });
    }
    for child in node
        .get("children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        visit(
            child,
            depth + 1,
            nodes,
            geometry,
            visiting,
            visited,
            inspection,
        );
    }
    visiting.remove(id);
}

fn ownership_label(node: &Value) -> String {
    let scope = node
        .pointer("/ownership/scope")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let owner = node
        .pointer("/ownership/owner_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if node
        .pointer("/ownership/protected")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        format!("{scope}:{owner} (protected)")
    } else {
        format!("{scope}:{owner}")
    }
}

fn mutability_label(node: &Value) -> String {
    if node
        .pointer("/ownership/protected")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return "protected".to_string();
    }
    if node.pointer("/ownership/scope").and_then(Value::as_str) == Some("extension") {
        return "extension_place".to_string();
    }
    let kind = node.get("type").and_then(Value::as_str);
    let mount = node.get("mount").and_then(Value::as_str);
    if matches!(kind, Some("toolbar" | "status_bar" | "menu_host"))
        || mount.is_some_and(|mount| {
            mount.ends_with("-top-bar")
                || matches!(
                    mount,
                    "builtin:extension-host.top-bar-actions"
                        | "builtin:extension-host.status-bar"
                        | "builtin:extension-host.canvas-controls"
                )
        })
    {
        "chrome".to_string()
    } else {
        "compose".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn inspection_reports_tree_geometry_ownership_and_readiness_problems() {
        let shell = json!({
            "mode":"single",
            "revision":9,
            "active_region_id":"canvas",
            "focused_node_id":"extension",
            "layout":{
                "root_id":"root",
                "nodes":[
                    {"id":"root","type":"application","children":["canvas","extension"],"ownership":{"scope":"application","owner_id":"odon","protected":true}},
                    {"id":"canvas","type":"canvas_slot","mount":"builtin:viewer-canvas","ownership":{"scope":"application","owner_id":"odon","protected":true}},
                    {"id":"extension","type":"extension_mount","mount":"extension:org.example/panel","ownership":{"scope":"extension","owner_id":"org.example","protected":false},"readiness":{"state":"disconnected","reason":"client closed"}}
                ]
            }
        });
        let geometry = BTreeMap::from([
            ("root".to_string(), [0.0, 0.0, 800.0, 600.0]),
            ("canvas".to_string(), [0.0, 0.0, 600.0, 600.0]),
        ]);
        let inspection = inspect(&shell, &geometry);
        assert_eq!(inspection.revision, 9);
        assert_eq!(inspection.rows.len(), 3);
        assert_eq!(inspection.rows[1].geometry, Some([0.0, 0.0, 600.0, 600.0]));
        assert_eq!(inspection.rows[2].mutability, "extension_place");
        assert!(inspection.problems[0].contains("disconnected"));
    }

    #[test]
    fn inspection_defensively_reports_invalid_topology() {
        let shell = json!({
            "layout":{
                "root_id":"root",
                "nodes":[
                    {"id":"root","type":"application","children":["missing"]},
                    {"id":"orphan","type":"panel","children":[]}
                ]
            }
        });
        let inspection = inspect(&shell, &BTreeMap::new());
        assert!(
            inspection
                .problems
                .iter()
                .any(|problem| problem.contains("unknown child"))
        );
        assert!(
            inspection
                .problems
                .iter()
                .any(|problem| problem.contains("unreachable"))
        );
    }
}
