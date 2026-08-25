//! Recursive renderer for actor-owned shell layout trees.
//!
//! The model validates the desired tree before it reaches the renderer. This module still parses
//! defensively: an incomplete projection produces no frame, allowing the caller to fall back to
//! its required canvas rather than panic. Geometry and interaction state are keyed by stable node
//! IDs so an accepted tree replacement does not reset unrelated tabs or split drags.

use std::collections::BTreeMap;

use eframe::egui;
use serde_json::{Map, Value, json};

const CONTAINER_GAP: f32 = 4.0;
const SPLIT_HANDLE: f32 = 6.0;
const TAB_HEADER_HEIGHT: f32 = 30.0;
const COLLAPSIBLE_HEADER_HEIGHT: f32 = 28.0;
const MIN_CHILD_EXTENT: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    Horizontal,
    Vertical,
}

impl Axis {
    fn extent(self, rect: egui::Rect) -> f32 {
        match self {
            Self::Horizontal => rect.width(),
            Self::Vertical => rect.height(),
        }
    }

    fn delta(self, vector: egui::Vec2) -> f32 {
        match self {
            Self::Horizontal => vector.x,
            Self::Vertical => vector.y,
        }
    }

    fn cursor(self) -> egui::CursorIcon {
        match self {
            Self::Horizontal => egui::CursorIcon::ResizeHorizontal,
            Self::Vertical => egui::CursorIcon::ResizeVertical,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
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

impl NodeKind {
    fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "application" => Self::Application,
            "row" => Self::Row,
            "column" => Self::Column,
            "split" => Self::Split,
            "tabs" => Self::Tabs,
            "panel" => Self::Panel,
            "collapsible" => Self::Collapsible,
            "toolbar" => Self::Toolbar,
            "status_bar" => Self::StatusBar,
            "menu_host" => Self::MenuHost,
            "canvas_slot" => Self::CanvasSlot,
            "builtin_mount" => Self::BuiltinMount,
            "extension_mount" => Self::ExtensionMount,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Default)]
struct NodeSize {
    width: Option<f32>,
    height: Option<f32>,
    min_width: Option<f32>,
    min_height: Option<f32>,
    max_width: Option<f32>,
    max_height: Option<f32>,
    flex: Option<f32>,
}

impl NodeSize {
    fn desired(&self, axis: Axis) -> Option<f32> {
        match axis {
            Axis::Horizontal => self.width,
            Axis::Vertical => self.height,
        }
    }

    fn min(&self, axis: Axis) -> f32 {
        match axis {
            Axis::Horizontal => self.min_width,
            Axis::Vertical => self.min_height,
        }
        .unwrap_or(MIN_CHILD_EXTENT)
        .max(MIN_CHILD_EXTENT)
    }

    fn max(&self, axis: Axis) -> f32 {
        match axis {
            Axis::Horizontal => self.max_width,
            Axis::Vertical => self.max_height,
        }
        .unwrap_or(f32::INFINITY)
        .max(self.min(axis))
    }
}

#[derive(Debug, Clone)]
struct SplitOptions {
    axis: Axis,
    ratio: f32,
    resizable: bool,
}

#[derive(Debug, Clone)]
struct ProjectedNode {
    id: String,
    kind: NodeKind,
    parent_id: Option<String>,
    children: Vec<String>,
    visible: bool,
    title: Option<String>,
    mount: Option<String>,
    selected_id: Option<String>,
    size: NodeSize,
    split: Option<SplitOptions>,
    collapsed: bool,
    configuration: Value,
}

#[derive(Debug, Clone)]
struct LeafPlacement {
    node_id: String,
    mount: String,
    rect: egui::Rect,
    configuration: Value,
}

#[derive(Debug, Clone)]
struct TabsPlacement {
    node_id: String,
    rect: egui::Rect,
    children: Vec<(String, String)>,
    selected_id: String,
}

#[derive(Debug, Clone)]
struct CollapsiblePlacement {
    node_id: String,
    rect: egui::Rect,
    title: String,
    collapsed: bool,
}

#[derive(Debug, Clone)]
struct SplitPlacement {
    node_id: String,
    rect: egui::Rect,
    axis: Axis,
    ratio: f32,
    min_ratio: f32,
    max_ratio: f32,
    content_extent: f32,
    resizable: bool,
}

#[derive(Debug, Clone, Copy)]
struct SplitUiState {
    revision: u64,
    drag_origin_ratio: f32,
    live_ratio: f32,
}

#[derive(Debug, Clone)]
pub(crate) enum ShellTreeChange {
    Activate {
        node_id: String,
    },
    Select {
        tabs_id: String,
        child_id: String,
    },
    Collapse {
        node_id: String,
        collapsed: bool,
    },
    Split {
        node_id: String,
        axis: &'static str,
        ratio: f32,
        resizable: bool,
    },
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ShellTreeChanges {
    changes: Vec<ShellTreeChange>,
}

impl ShellTreeChanges {
    pub(crate) fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub(crate) fn patch_params(&self) -> Value {
        let mut selected = Map::new();
        let mut collapsed = Map::new();
        let mut splits = Map::new();
        let mut active_region_id = None;
        let mut focused_node_id = None;
        for change in &self.changes {
            match change {
                ShellTreeChange::Activate { node_id } => {
                    active_region_id = Some(node_id.clone());
                    focused_node_id = Some(node_id.clone());
                }
                ShellTreeChange::Select { tabs_id, child_id } => {
                    selected.insert(tabs_id.clone(), Value::String(child_id.clone()));
                }
                ShellTreeChange::Collapse {
                    node_id,
                    collapsed: value,
                } => {
                    collapsed.insert(node_id.clone(), Value::Bool(*value));
                }
                ShellTreeChange::Split {
                    node_id,
                    axis,
                    ratio,
                    resizable,
                } => {
                    splits.insert(
                        node_id.clone(),
                        json!({
                            "axis":axis,
                            "ratio":ratio,
                            "resizable":resizable,
                        }),
                    );
                }
            }
        }
        let mut params = Map::new();
        if !selected.is_empty() {
            params.insert("selected".to_string(), Value::Object(selected));
        }
        if !collapsed.is_empty() {
            params.insert("collapsed".to_string(), Value::Object(collapsed));
        }
        if !splits.is_empty() {
            params.insert("splits".to_string(), Value::Object(splits));
        }
        if let Some(active_region_id) = active_region_id {
            params.insert(
                "active_region_id".to_string(),
                Value::String(active_region_id),
            );
        }
        if let Some(focused_node_id) = focused_node_id {
            params.insert(
                "focused_node_id".to_string(),
                Value::String(focused_node_id),
            );
        }
        Value::Object(params)
    }

    fn activate(&mut self, node_id: &str, active: &str, focused: Option<&str>) {
        if active == node_id && focused == Some(node_id) {
            return;
        }
        self.changes
            .retain(|change| !matches!(change, ShellTreeChange::Activate { .. }));
        self.changes.push(ShellTreeChange::Activate {
            node_id: node_id.to_string(),
        });
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ShellTreeFrame {
    revision: u64,
    root_rect: egui::Rect,
    node_rects: BTreeMap<String, egui::Rect>,
    leaves: Vec<LeafPlacement>,
    tabs: Vec<TabsPlacement>,
    collapsibles: Vec<CollapsiblePlacement>,
    splits: Vec<SplitPlacement>,
    active_region_id: String,
    focused_node_id: Option<String>,
}

impl ShellTreeFrame {
    #[cfg(test)]
    pub(crate) fn from_projection(
        ctx: &egui::Context,
        shell: &Value,
        required_mount: &str,
        rect: egui::Rect,
    ) -> Option<Self> {
        Self::from_projection_with_mount_filter(ctx, shell, required_mount, rect, |_| true)
    }

    pub(crate) fn from_projection_with_mount_filter(
        ctx: &egui::Context,
        shell: &Value,
        required_mount: &str,
        rect: egui::Rect,
        mut mount_available: impl FnMut(&str) -> bool,
    ) -> Option<Self> {
        let layout = shell.get("layout")?;
        let root_id = layout.get("root_id")?.as_str()?.to_string();
        let revision = shell.get("revision").and_then(Value::as_u64).unwrap_or(0);
        let active_region_id = shell.get("active_region_id")?.as_str()?.to_string();
        let focused_node_id = shell
            .get("focused_node_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let mut nodes = layout
            .get("nodes")?
            .as_array()?
            .iter()
            .map(parse_node)
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .map(|node| (node.id.clone(), node))
            .collect::<BTreeMap<_, _>>();
        for node in nodes.values_mut() {
            if node
                .mount
                .as_deref()
                .is_some_and(|mount| !mount_available(mount))
            {
                // Availability is renderer-local. The actor still owns the declared visibility,
                // so a host reappears in the same keyed location when a contribution registers.
                node.visible = false;
            }
        }
        let root = nodes.get(&root_id)?;
        let required_id = nodes
            .values()
            .find(|node| node.mount.as_deref() == Some(required_mount))?
            .id
            .clone();
        if !is_descendant_visible(&required_id, &nodes) {
            return None;
        }

        let content_roots = root.children.clone();
        if content_roots.is_empty()
            || !content_roots
                .iter()
                .any(|id| subtree_contains(id, &required_id, &nodes))
        {
            return None;
        }

        let mut builder = FrameBuilder {
            ctx,
            revision,
            nodes: &nodes,
            node_rects: BTreeMap::new(),
            leaves: Vec::new(),
            tabs: Vec::new(),
            collapsibles: Vec::new(),
            splits: Vec::new(),
        };
        if content_roots.len() == 1 {
            builder.layout_node(&content_roots[0], rect);
        } else {
            builder.layout_children(&content_roots, rect, Axis::Vertical, CONTAINER_GAP);
        }
        if !builder
            .leaves
            .iter()
            .any(|leaf| leaf.mount == required_mount)
        {
            return None;
        }
        Some(Self {
            revision,
            root_rect: rect,
            node_rects: builder.node_rects,
            leaves: builder.leaves,
            tabs: builder.tabs,
            collapsibles: builder.collapsibles,
            splits: builder.splits,
            active_region_id,
            focused_node_id,
        })
    }

    pub(crate) fn show(
        &self,
        ui: &mut egui::Ui,
        mut render_mount: impl FnMut(&mut egui::Ui, &str, &Value),
    ) -> ShellTreeChanges {
        let mut changes = ShellTreeChanges::default();
        ui.allocate_rect(self.root_rect, egui::Sense::hover());

        for tabs in &self.tabs {
            let mut header = ui.new_child(
                egui::UiBuilder::new()
                    .id_salt(("odon-shell-tabs", tabs.node_id.as_str()))
                    .max_rect(tabs.rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            header.set_clip_rect(tabs.rect);
            header
                .painter()
                .rect_filled(tabs.rect, 0.0, header.visuals().panel_fill);
            for (child_id, title) in &tabs.children {
                let response = header.selectable_label(child_id == &tabs.selected_id, title);
                if response.clicked() && child_id != &tabs.selected_id {
                    let state_id =
                        egui::Id::new(("odon-shell-tabs-selected", tabs.node_id.as_str()));
                    header.ctx().data_mut(|data| {
                        data.insert_temp(state_id, (self.revision, child_id.clone()))
                    });
                    changes.changes.push(ShellTreeChange::Select {
                        tabs_id: tabs.node_id.clone(),
                        child_id: child_id.clone(),
                    });
                }
                if response.clicked() {
                    changes.activate(
                        child_id,
                        &self.active_region_id,
                        self.focused_node_id.as_deref(),
                    );
                }
            }
        }

        for collapsible in &self.collapsibles {
            let mut header = ui.new_child(
                egui::UiBuilder::new()
                    .id_salt(("odon-shell-collapsible", collapsible.node_id.as_str()))
                    .max_rect(collapsible.rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            header.set_clip_rect(collapsible.rect);
            let marker = if collapsible.collapsed { "▶" } else { "▼" };
            if header
                .button(format!("{marker} {}", collapsible.title))
                .clicked()
            {
                let collapsed = !collapsible.collapsed;
                let state_id =
                    egui::Id::new(("odon-shell-collapsed", collapsible.node_id.as_str()));
                header
                    .ctx()
                    .data_mut(|data| data.insert_temp(state_id, (self.revision, collapsed)));
                changes.changes.push(ShellTreeChange::Collapse {
                    node_id: collapsible.node_id.clone(),
                    collapsed,
                });
                changes.activate(
                    &collapsible.node_id,
                    &self.active_region_id,
                    self.focused_node_id.as_deref(),
                );
            }
        }

        for leaf in &self.leaves {
            let mut child = ui.new_child(
                egui::UiBuilder::new()
                    .id_salt(("odon-shell-mount", leaf.node_id.as_str()))
                    .max_rect(leaf.rect)
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            child.set_clip_rect(leaf.rect);
            render_mount(&mut child, &leaf.mount, &leaf.configuration);
            if child.rect_contains_pointer(leaf.rect)
                && child.input(|input| input.pointer.any_click())
            {
                changes.activate(
                    &leaf.node_id,
                    &self.active_region_id,
                    self.focused_node_id.as_deref(),
                );
            }
        }

        for split in &self.splits {
            let mut handle_ui = ui.new_child(
                egui::UiBuilder::new()
                    .id_salt(("odon-shell-split", split.node_id.as_str()))
                    .max_rect(split.rect),
            );
            handle_ui.set_clip_rect(split.rect);
            let sense = if split.resizable {
                egui::Sense::drag()
            } else {
                egui::Sense::hover()
            };
            let response = handle_ui.allocate_rect(split.rect, sense);
            let stroke = if response.dragged() || response.hovered() {
                handle_ui.visuals().widgets.active.fg_stroke
            } else {
                handle_ui.visuals().widgets.inactive.fg_stroke
            };
            let points = match split.axis {
                Axis::Horizontal => [
                    egui::pos2(split.rect.center().x, split.rect.top()),
                    egui::pos2(split.rect.center().x, split.rect.bottom()),
                ],
                Axis::Vertical => [
                    egui::pos2(split.rect.left(), split.rect.center().y),
                    egui::pos2(split.rect.right(), split.rect.center().y),
                ],
            };
            handle_ui.painter().line_segment(points, stroke);
            if !split.resizable {
                continue;
            }
            response.clone().on_hover_cursor(split.axis.cursor());
            let state_id = egui::Id::new(("odon-shell-split-state", split.node_id.as_str()));
            let mut state = handle_ui
                .ctx()
                .data(|data| data.get_temp::<SplitUiState>(state_id))
                .filter(|state| state.revision == self.revision)
                .unwrap_or(SplitUiState {
                    revision: self.revision,
                    drag_origin_ratio: split.ratio,
                    live_ratio: split.ratio,
                });
            if response.drag_started() {
                state.drag_origin_ratio = split.ratio;
            }
            if response.dragged() || response.drag_stopped() {
                let delta = split.axis.delta(response.drag_delta());
                state.live_ratio = (state.drag_origin_ratio + delta / split.content_extent)
                    .clamp(split.min_ratio, split.max_ratio);
                handle_ui
                    .ctx()
                    .data_mut(|data| data.insert_temp(state_id, state));
                handle_ui.ctx().request_repaint();
            }
            if response.drag_stopped() {
                if (state.live_ratio - split.ratio).abs() > f32::EPSILON {
                    changes.changes.push(ShellTreeChange::Split {
                        node_id: split.node_id.clone(),
                        axis: match split.axis {
                            Axis::Horizontal => "horizontal",
                            Axis::Vertical => "vertical",
                        },
                        ratio: state.live_ratio,
                        resizable: split.resizable,
                    });
                }
                changes.activate(
                    &split.node_id,
                    &self.active_region_id,
                    self.focused_node_id.as_deref(),
                );
            }
        }

        changes
    }

    pub(crate) fn node_rects(&self) -> impl Iterator<Item = (&str, egui::Rect)> {
        self.node_rects
            .iter()
            .map(|(id, rect)| (id.as_str(), *rect))
    }
}

struct FrameBuilder<'a> {
    ctx: &'a egui::Context,
    revision: u64,
    nodes: &'a BTreeMap<String, ProjectedNode>,
    node_rects: BTreeMap<String, egui::Rect>,
    leaves: Vec<LeafPlacement>,
    tabs: Vec<TabsPlacement>,
    collapsibles: Vec<CollapsiblePlacement>,
    splits: Vec<SplitPlacement>,
}

impl FrameBuilder<'_> {
    fn layout_node(&mut self, id: &str, rect: egui::Rect) {
        let Some(node) = self.nodes.get(id) else {
            return;
        };
        if !node_effectively_visible(id, self.nodes) || rect.width() <= 0.0 || rect.height() <= 0.0
        {
            return;
        }
        self.node_rects.insert(id.to_string(), rect);
        match node.kind {
            NodeKind::Application => {
                self.layout_children(&node.children, rect, Axis::Vertical, CONTAINER_GAP)
            }
            NodeKind::Row => {
                self.layout_children(&node.children, rect, Axis::Horizontal, CONTAINER_GAP)
            }
            NodeKind::Column => {
                self.layout_children(&node.children, rect, Axis::Vertical, CONTAINER_GAP)
            }
            NodeKind::Split => self.layout_split(node, rect),
            NodeKind::Tabs => self.layout_tabs(node, rect),
            NodeKind::Panel => {
                if let Some(child) = node.children.first() {
                    self.layout_node(child, rect.shrink(1.0));
                }
            }
            NodeKind::Collapsible => self.layout_collapsible(node, rect),
            NodeKind::Toolbar | NodeKind::MenuHost => {
                self.layout_children(&node.children, rect, Axis::Horizontal, CONTAINER_GAP)
            }
            NodeKind::StatusBar => {
                self.layout_children(&node.children, rect, Axis::Horizontal, CONTAINER_GAP)
            }
            NodeKind::CanvasSlot | NodeKind::BuiltinMount | NodeKind::ExtensionMount => {
                if let Some(mount) = node.mount.as_ref() {
                    self.leaves.push(LeafPlacement {
                        node_id: node.id.clone(),
                        mount: mount.clone(),
                        rect,
                        configuration: node.configuration.clone(),
                    });
                }
            }
        }
    }

    fn layout_children(&mut self, ids: &[String], rect: egui::Rect, axis: Axis, gap: f32) {
        let visible = ids
            .iter()
            .filter_map(|id| {
                let node = self.nodes.get(id)?;
                node_effectively_visible(id, self.nodes).then_some(node)
            })
            .collect::<Vec<_>>();
        if visible.is_empty() {
            return;
        }
        let extents = distribute_extents(&visible, axis, axis.extent(rect), gap);
        let mut cursor = match axis {
            Axis::Horizontal => rect.left(),
            Axis::Vertical => rect.top(),
        };
        for (index, (node, extent)) in visible.into_iter().zip(extents).enumerate() {
            let child_rect = match axis {
                Axis::Horizontal => egui::Rect::from_min_max(
                    egui::pos2(cursor, rect.top()),
                    egui::pos2((cursor + extent).min(rect.right()), rect.bottom()),
                ),
                Axis::Vertical => egui::Rect::from_min_max(
                    egui::pos2(rect.left(), cursor),
                    egui::pos2(rect.right(), (cursor + extent).min(rect.bottom())),
                ),
            };
            self.layout_node(&node.id, child_rect);
            cursor += extent;
            if index + 1 < ids.len() {
                cursor += gap;
            }
        }
    }

    fn layout_split(&mut self, node: &ProjectedNode, rect: egui::Rect) {
        let Some(split) = node.split.as_ref() else {
            return;
        };
        let first_visible = node
            .children
            .first()
            .is_some_and(|id| node_effectively_visible(id, self.nodes));
        let second_visible = node
            .children
            .get(1)
            .is_some_and(|id| node_effectively_visible(id, self.nodes));
        match (first_visible, second_visible) {
            (true, false) => {
                if let Some(first) = node.children.first() {
                    self.layout_node(first, rect);
                }
                return;
            }
            (false, true) => {
                if let Some(second) = node.children.get(1) {
                    self.layout_node(second, rect);
                }
                return;
            }
            (false, false) => return,
            (true, true) => {}
        }
        let state_id = egui::Id::new(("odon-shell-split-state", node.id.as_str()));
        let ratio = self
            .ctx
            .data(|data| data.get_temp::<SplitUiState>(state_id))
            .filter(|state| state.revision == self.revision)
            .map(|state| state.live_ratio)
            .unwrap_or(split.ratio)
            .clamp(0.05, 0.95);
        let content_extent = (split.axis.extent(rect) - SPLIT_HANDLE).max(2.0);
        let (min_ratio, max_ratio) =
            split_ratio_bounds(node, split.axis, content_extent, self.nodes);
        let ratio = ratio.clamp(min_ratio, max_ratio);
        let first_extent = (content_extent * ratio).max(MIN_CHILD_EXTENT);
        let divider_start = match split.axis {
            Axis::Horizontal => rect.left() + first_extent,
            Axis::Vertical => rect.top() + first_extent,
        };
        let (first, divider, second) = match split.axis {
            Axis::Horizontal => (
                egui::Rect::from_min_max(rect.min, egui::pos2(divider_start, rect.bottom())),
                egui::Rect::from_min_max(
                    egui::pos2(divider_start, rect.top()),
                    egui::pos2(divider_start + SPLIT_HANDLE, rect.bottom()),
                ),
                egui::Rect::from_min_max(
                    egui::pos2(divider_start + SPLIT_HANDLE, rect.top()),
                    rect.max,
                ),
            ),
            Axis::Vertical => (
                egui::Rect::from_min_max(rect.min, egui::pos2(rect.right(), divider_start)),
                egui::Rect::from_min_max(
                    egui::pos2(rect.left(), divider_start),
                    egui::pos2(rect.right(), divider_start + SPLIT_HANDLE),
                ),
                egui::Rect::from_min_max(
                    egui::pos2(rect.left(), divider_start + SPLIT_HANDLE),
                    rect.max,
                ),
            ),
        };
        if let Some(first_id) = node.children.first() {
            self.layout_node(first_id, first);
        }
        if let Some(second_id) = node.children.get(1) {
            self.layout_node(second_id, second);
        }
        self.splits.push(SplitPlacement {
            node_id: node.id.clone(),
            rect: divider,
            axis: split.axis,
            ratio,
            min_ratio,
            max_ratio,
            content_extent,
            resizable: split.resizable,
        });
    }

    fn layout_tabs(&mut self, node: &ProjectedNode, rect: egui::Rect) {
        let visible_children = node
            .children
            .iter()
            .filter(|id| node_effectively_visible(id, self.nodes))
            .cloned()
            .collect::<Vec<_>>();
        if visible_children.is_empty() {
            return;
        }
        let state_id = egui::Id::new(("odon-shell-tabs-selected", node.id.as_str()));
        let selected = self
            .ctx
            .data(|data| data.get_temp::<(u64, String)>(state_id))
            .filter(|(revision, selected)| {
                *revision == self.revision && visible_children.iter().any(|id| id == selected)
            })
            .map(|(_, selected)| selected)
            .or_else(|| {
                node.selected_id
                    .clone()
                    .filter(|selected| visible_children.contains(selected))
            })
            .or_else(|| visible_children.first().cloned());
        let Some(selected) = selected else {
            return;
        };
        let header_height = TAB_HEADER_HEIGHT.min(rect.height().max(0.0));
        let header = egui::Rect::from_min_max(
            rect.min,
            egui::pos2(rect.right(), rect.top() + header_height),
        );
        let body = egui::Rect::from_min_max(egui::pos2(rect.left(), header.bottom()), rect.max);
        self.tabs.push(TabsPlacement {
            node_id: node.id.clone(),
            rect: header,
            children: visible_children
                .iter()
                .filter_map(|id| {
                    self.nodes
                        .get(id)
                        .map(|child| (id.clone(), node_title(child, self.nodes)))
                })
                .collect(),
            selected_id: selected.clone(),
        });
        self.layout_node(&selected, body.shrink(1.0));
    }

    fn layout_collapsible(&mut self, node: &ProjectedNode, rect: egui::Rect) {
        let state_id = egui::Id::new(("odon-shell-collapsed", node.id.as_str()));
        let collapsed = self
            .ctx
            .data(|data| data.get_temp::<(u64, bool)>(state_id))
            .filter(|(revision, _)| *revision == self.revision)
            .map(|(_, collapsed)| collapsed)
            .unwrap_or(node.collapsed);
        let header_height = COLLAPSIBLE_HEADER_HEIGHT.min(rect.height().max(0.0));
        let header = egui::Rect::from_min_max(
            rect.min,
            egui::pos2(rect.right(), rect.top() + header_height),
        );
        self.collapsibles.push(CollapsiblePlacement {
            node_id: node.id.clone(),
            rect: header,
            title: node
                .title
                .clone()
                .unwrap_or_else(|| node_title(node, self.nodes)),
            collapsed,
        });
        if !collapsed && let Some(child) = node.children.first() {
            let body = egui::Rect::from_min_max(egui::pos2(rect.left(), header.bottom()), rect.max);
            self.layout_node(child, body.shrink(1.0));
        }
    }
}

fn parse_node(value: &Value) -> Option<ProjectedNode> {
    let size = value.get("size");
    let kind = NodeKind::parse(value.get("type")?.as_str()?)?;
    let mount = value
        .get("mount")
        .and_then(Value::as_str)
        .map(str::to_string);
    let intrinsic_minimum = mount
        .as_deref()
        .and_then(odon::model::shell_component_minimum_size);
    let chrome_height = application_chrome_height(kind, mount.as_deref());
    let split = value.get("split").and_then(|split| {
        Some(SplitOptions {
            axis: match split.get("axis").and_then(Value::as_str) {
                Some("vertical") => Axis::Vertical,
                _ => Axis::Horizontal,
            },
            ratio: value_f32(split.get("ratio")?)?,
            resizable: split
                .get("resizable")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        })
    });
    Some(ProjectedNode {
        id: value.get("id")?.as_str()?.to_string(),
        kind,
        parent_id: value
            .get("parent_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        children: value
            .get("children")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        visible: value
            .get("visible")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        title: value
            .get("title")
            .and_then(Value::as_str)
            .map(str::to_string),
        mount,
        selected_id: value
            .get("selected_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        size: NodeSize {
            width: size.and_then(|size| size.get("width")).and_then(value_f32),
            height: size
                .and_then(|size| size.get("height"))
                .and_then(value_f32)
                .or(chrome_height),
            min_width: size
                .and_then(|size| size.get("min_width"))
                .and_then(value_f32)
                .or_else(|| intrinsic_minimum.map(|minimum| minimum[0])),
            min_height: size
                .and_then(|size| size.get("min_height"))
                .and_then(value_f32)
                .or_else(|| intrinsic_minimum.map(|minimum| minimum[1])),
            max_width: size
                .and_then(|size| size.get("max_width"))
                .and_then(value_f32),
            max_height: size
                .and_then(|size| size.get("max_height"))
                .and_then(value_f32)
                .or_else(|| chrome_height.map(|height| height * 2.0)),
            flex: size.and_then(|size| size.get("flex")).and_then(value_f32),
        },
        split,
        collapsed: value
            .get("collapsed")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        configuration: value
            .get("configuration")
            .filter(|configuration| configuration.is_object())
            .cloned()
            .unwrap_or_else(|| json!({})),
    })
}

fn value_f32(value: &Value) -> Option<f32> {
    let value = value.as_f64()?;
    value.is_finite().then_some(value as f32)
}

fn distribute_extents(
    nodes: &[&ProjectedNode],
    axis: Axis,
    total_extent: f32,
    gap: f32,
) -> Vec<f32> {
    let gap_total = gap * nodes.len().saturating_sub(1) as f32;
    let available = (total_extent - gap_total).max(nodes.len() as f32 * MIN_CHILD_EXTENT);
    let mut extents = nodes
        .iter()
        .map(|node| {
            node.size
                .desired(axis)
                .unwrap_or_else(|| node.size.min(axis))
                .clamp(node.size.min(axis), node.size.max(axis))
        })
        .collect::<Vec<_>>();
    let used = extents.iter().sum::<f32>();
    if used < available {
        let primary = nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| node.size.desired(axis).is_none().then_some(index))
            .collect::<Vec<_>>();
        grow_extents(nodes, axis, &mut extents, &primary, available - used);
        let remaining = available - extents.iter().sum::<f32>();
        if remaining > 0.01 {
            let all = (0..nodes.len()).collect::<Vec<_>>();
            grow_extents(nodes, axis, &mut extents, &all, remaining);
        }
    } else if used > available {
        shrink_extents(nodes, axis, &mut extents, used - available);
        let sum = extents.iter().sum::<f32>();
        if sum > available && sum > 0.0 {
            // The parent is smaller than the sum of declared minimums. Preserve a usable frame
            // rather than overflowing; the actor-owned constraints remain unchanged and will
            // apply again when more space becomes available.
            let scale = available / sum;
            for extent in &mut extents {
                *extent = (*extent * scale).max(MIN_CHILD_EXTENT);
            }
        }
    }
    extents
}

fn grow_extents(
    nodes: &[&ProjectedNode],
    axis: Axis,
    extents: &mut [f32],
    candidates: &[usize],
    mut remaining: f32,
) {
    let mut active = candidates.to_vec();
    while remaining > 0.01 && !active.is_empty() {
        let weight_total = active
            .iter()
            .map(|index| nodes[*index].size.flex.unwrap_or(1.0).max(0.0001))
            .sum::<f32>();
        let before = remaining;
        active.retain(|index| {
            let weight = nodes[*index].size.flex.unwrap_or(1.0).max(0.0001);
            let share = before * weight / weight_total;
            let capacity = (nodes[*index].size.max(axis) - extents[*index]).max(0.0);
            let added = share.min(capacity);
            extents[*index] += added;
            remaining -= added;
            capacity - added > 0.01
        });
        if (before - remaining).abs() < 0.01 {
            break;
        }
    }
}

fn shrink_extents(nodes: &[&ProjectedNode], axis: Axis, extents: &mut [f32], mut excess: f32) {
    let mut active = (0..nodes.len()).collect::<Vec<_>>();
    while excess > 0.01 && !active.is_empty() {
        let capacity_total = active
            .iter()
            .map(|index| (extents[*index] - nodes[*index].size.min(axis)).max(0.0))
            .sum::<f32>();
        if capacity_total <= 0.01 {
            break;
        }
        let before = excess;
        active.retain(|index| {
            let capacity = (extents[*index] - nodes[*index].size.min(axis)).max(0.0);
            let removed = (before * capacity / capacity_total).min(capacity);
            extents[*index] -= removed;
            excess -= removed;
            capacity - removed > 0.01
        });
    }
}

fn split_ratio_bounds(
    node: &ProjectedNode,
    axis: Axis,
    content_extent: f32,
    nodes: &BTreeMap<String, ProjectedNode>,
) -> (f32, f32) {
    let first = node.children.first().and_then(|id| nodes.get(id));
    let second = node.children.get(1).and_then(|id| nodes.get(id));
    let Some((first, second)) = first.zip(second) else {
        return (0.05, 0.95);
    };
    let lower = (first.size.min(axis) / content_extent)
        .max(1.0 - second.size.max(axis) / content_extent)
        .max(0.05);
    let upper = (first.size.max(axis) / content_extent)
        .min(1.0 - second.size.min(axis) / content_extent)
        .min(0.95);
    if lower <= upper {
        (lower, upper)
    } else {
        // Constraints cannot all fit in the available viewport. The 5–95% actor invariant is the
        // safe interactive fallback; geometry will honor the constraints again at a larger size.
        (0.05, 0.95)
    }
}

fn application_chrome_height(kind: NodeKind, mount: Option<&str>) -> Option<f32> {
    match kind {
        NodeKind::Toolbar | NodeKind::MenuHost => Some(36.0),
        NodeKind::StatusBar => Some(24.0),
        _ if mount.is_some_and(|mount| mount.ends_with("-top-bar")) => Some(36.0),
        _ => None,
    }
}

fn subtree_contains(
    root_id: &str,
    descendant_id: &str,
    nodes: &BTreeMap<String, ProjectedNode>,
) -> bool {
    if root_id == descendant_id {
        return true;
    }
    nodes.get(root_id).is_some_and(|node| {
        node.children
            .iter()
            .any(|child| subtree_contains(child, descendant_id, nodes))
    })
}

fn is_descendant_visible(id: &str, nodes: &BTreeMap<String, ProjectedNode>) -> bool {
    let mut current = id;
    loop {
        let Some(node) = nodes.get(current) else {
            return false;
        };
        if !node.visible || (node.kind == NodeKind::Collapsible && node.collapsed) {
            return false;
        }
        let Some(parent) = node.parent_id.as_deref() else {
            return true;
        };
        current = parent;
    }
}

fn node_effectively_visible(id: &str, nodes: &BTreeMap<String, ProjectedNode>) -> bool {
    let Some(node) = nodes.get(id) else {
        return false;
    };
    if !node.visible {
        return false;
    }
    if node.kind == NodeKind::Collapsible && node.collapsed {
        return true;
    }
    match node.kind {
        NodeKind::CanvasSlot | NodeKind::BuiltinMount | NodeKind::ExtensionMount => true,
        _ => node
            .children
            .iter()
            .any(|child| node_effectively_visible(child, nodes)),
    }
}

fn node_title(node: &ProjectedNode, nodes: &BTreeMap<String, ProjectedNode>) -> String {
    if let Some(title) = node.title.as_ref() {
        return title.clone();
    }
    if let Some(mount) = node.mount.as_deref() {
        return humanize_mount(mount);
    }
    node.children
        .iter()
        .find_map(|id| nodes.get(id))
        .map(|child| node_title(child, nodes))
        .unwrap_or_else(|| humanize_mount(&node.id))
}

fn humanize_mount(value: &str) -> String {
    let tail = value
        .rsplit([':', '.'])
        .next()
        .unwrap_or(value)
        .replace(['-', '_'], " ");
    let mut chars = tail.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => "Untitled".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(nodes: Value) -> Value {
        json!({
            "revision":7,
            "active_region_id":"root",
            "focused_node_id":Value::Null,
            "layout":{
                "root_id":"root",
                "nodes":nodes,
            }
        })
    }

    fn node(id: &str, kind: &str, children: &[&str]) -> Value {
        json!({
            "id":id,
            "type":kind,
            "children":children,
            "visible":true,
            "size":{},
            "collapsed":false,
        })
    }

    #[test]
    fn activation_changes_are_semantic_and_suppress_projection_no_ops() {
        let mut changes = ShellTreeChanges::default();
        changes.activate("canvas", "root", None);
        assert_eq!(changes.patch_params()["active_region_id"], "canvas");
        assert_eq!(changes.patch_params()["focused_node_id"], "canvas");

        let mut no_op = ShellTreeChanges::default();
        no_op.activate("canvas", "canvas", Some("canvas"));
        assert!(no_op.is_empty());
    }

    fn mount(id: &str, kind: &str, mount: &str) -> Value {
        json!({
            "id":id,
            "type":kind,
            "children":[],
            "visible":true,
            "mount":mount,
            "size":{},
            "collapsed":false,
        })
    }

    #[test]
    fn nested_rows_and_columns_produce_distinct_leaf_geometry() {
        let ctx = egui::Context::default();
        let mut top = mount("top", "builtin_mount", "builtin:viewer-top-bar");
        top["configuration"] = json!({"show_title":false});
        let value = shell(json!([
            node("root", "application", &["top", "body"]),
            top,
            node("body", "row", &["left", "right-column"]),
            mount("left", "builtin_mount", "builtin:layers"),
            node("right-column", "column", &["canvas", "properties"]),
            mount("canvas", "canvas_slot", "builtin:viewer-canvas"),
            mount("properties", "builtin_mount", "builtin:properties"),
        ]));
        let frame = ShellTreeFrame::from_projection(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 600.0)),
        )
        .unwrap();
        let rects = frame
            .leaves
            .iter()
            .map(|leaf| (leaf.mount.as_str(), leaf.rect))
            .collect::<BTreeMap<_, _>>();
        assert!((rects["builtin:viewer-top-bar"].height() - 36.0).abs() < 0.1);
        assert_eq!(frame.leaves[0].configuration["show_title"], false);
        assert!(rects["builtin:viewer-top-bar"].bottom() < rects["builtin:layers"].top());
        assert!(rects["builtin:layers"].right() < rects["builtin:viewer-canvas"].left());
        assert!(rects["builtin:viewer-canvas"].bottom() < rects["builtin:properties"].top());
    }

    #[test]
    fn application_chrome_hosts_are_vertical_and_intrinsically_sized() {
        let ctx = egui::Context::default();
        let value = shell(json!([
            node(
                "root",
                "application",
                &["menu", "toolbar", "canvas", "status"]
            ),
            node("menu", "menu_host", &["menu-item"]),
            mount("menu-item", "extension_mount", "extension:menu"),
            node("toolbar", "toolbar", &["tool"]),
            mount("tool", "extension_mount", "extension:tool"),
            mount("canvas", "canvas_slot", "builtin:viewer-canvas"),
            node("status", "status_bar", &["status-item"]),
            mount("status-item", "extension_mount", "extension:status"),
        ]));
        let frame = ShellTreeFrame::from_projection(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 600.0)),
        )
        .unwrap();
        let rects = frame
            .leaves
            .iter()
            .map(|leaf| (leaf.mount.as_str(), leaf.rect))
            .collect::<BTreeMap<_, _>>();
        assert!((rects["extension:menu"].height() - 36.0).abs() < 0.1);
        assert!((rects["extension:tool"].height() - 36.0).abs() < 0.1);
        assert!((rects["extension:status"].height() - 24.0).abs() < 0.1);
        assert!(rects["extension:menu"].bottom() < rects["extension:tool"].top());
        assert!(rects["extension:tool"].bottom() < rects["builtin:viewer-canvas"].top());
        assert!(rects["builtin:viewer-canvas"].bottom() < rects["extension:status"].top());
    }

    #[test]
    fn nested_split_uses_axis_and_ratio() {
        let ctx = egui::Context::default();
        let mut split = node("split", "split", &["layers", "canvas"]);
        split["split"] = json!({
            "axis":"horizontal",
            "ratio":0.25,
            "resizable":true,
        });
        let value = shell(json!([
            node("root", "application", &["split"]),
            split,
            mount("layers", "builtin_mount", "builtin:layers"),
            mount("canvas", "canvas_slot", "builtin:viewer-canvas"),
        ]));
        let frame = ShellTreeFrame::from_projection(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 500.0)),
        )
        .unwrap();
        let rects = frame
            .leaves
            .iter()
            .map(|leaf| (leaf.mount.as_str(), leaf.rect))
            .collect::<BTreeMap<_, _>>();
        assert!((rects["builtin:layers"].width() - 248.5).abs() < 0.1);
        assert!(rects["builtin:viewer-canvas"].width() > 740.0);
        assert_eq!(frame.splits.len(), 1);
    }

    #[test]
    fn split_and_row_geometry_honor_feasible_size_constraints() {
        let ctx = egui::Context::default();
        let mut split = node("split", "split", &["left", "canvas"]);
        split["split"] = json!({
            "axis":"horizontal",
            "ratio":0.1,
            "resizable":true,
        });
        let mut left = mount("left", "builtin_mount", "builtin:layers");
        left["size"] = json!({"min_width":300.0,"max_width":360.0});
        let mut canvas = mount("canvas", "canvas_slot", "builtin:viewer-canvas");
        canvas["size"] = json!({"min_width":400.0});
        let value = shell(json!([
            node("root", "application", &["split"]),
            split,
            left,
            canvas,
        ]));
        let frame = ShellTreeFrame::from_projection(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 500.0)),
        )
        .unwrap();
        let rects = frame
            .leaves
            .iter()
            .map(|leaf| (leaf.mount.as_str(), leaf.rect))
            .collect::<BTreeMap<_, _>>();
        assert!(rects["builtin:layers"].width() >= 299.9);
        assert!(rects["builtin:layers"].width() <= 360.1);
        assert!(rects["builtin:viewer-canvas"].width() >= 400.0);
        assert!(frame.splits[0].min_ratio >= 300.0 / 994.0);

        let mut fixed = mount("fixed", "builtin_mount", "builtin:layers");
        fixed["size"] = json!({"width":200.0,"min_width":180.0,"max_width":220.0});
        let mut flexible = mount("flex", "builtin_mount", "builtin:properties");
        flexible["size"] = json!({"min_width":100.0,"max_width":500.0,"flex":2.0});
        let extents = distribute_extents(
            &[
                &parse_node(&fixed).unwrap(),
                &parse_node(&flexible).unwrap(),
            ],
            Axis::Horizontal,
            604.0,
            4.0,
        );
        assert!((extents[0] - 200.0).abs() < 0.1);
        assert!((extents[1] - 400.0).abs() < 0.1);
    }

    #[test]
    fn selected_tab_is_the_only_mounted_child() {
        let ctx = egui::Context::default();
        let mut tabs = node("tabs", "tabs", &["layers", "canvas"]);
        tabs["selected_id"] = json!("canvas");
        let value = shell(json!([
            node("root", "application", &["tabs"]),
            tabs,
            mount("layers", "builtin_mount", "builtin:layers"),
            mount("canvas", "canvas_slot", "builtin:viewer-canvas"),
        ]));
        let frame = ShellTreeFrame::from_projection(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 500.0)),
        )
        .unwrap();
        assert_eq!(frame.leaves.len(), 1);
        assert_eq!(frame.leaves[0].mount, "builtin:viewer-canvas");
        assert_eq!(frame.tabs[0].selected_id, "canvas");
    }

    #[test]
    fn hidden_panel_subtree_releases_its_split_extent() {
        let ctx = egui::Context::default();
        let mut split = node("split", "split", &["panel", "canvas"]);
        split["split"] = json!({
            "axis":"horizontal",
            "ratio":0.3,
            "resizable":true,
        });
        let panel = node("panel", "panel", &["tabs"]);
        let mut tabs = node("tabs", "tabs", &["layers", "project"]);
        tabs["selected_id"] = json!("layers");
        let mut layers = mount("layers", "builtin_mount", "builtin:layers");
        layers["visible"] = json!(false);
        let mut project = mount("project", "builtin_mount", "builtin:project");
        project["visible"] = json!(false);
        let value = shell(json!([
            node("root", "application", &["split"]),
            split,
            panel,
            tabs,
            layers,
            project,
            mount("canvas", "canvas_slot", "builtin:viewer-canvas"),
        ]));
        let frame = ShellTreeFrame::from_projection(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 500.0)),
        )
        .unwrap();
        assert_eq!(frame.leaves.len(), 1);
        assert_eq!(frame.leaves[0].mount, "builtin:viewer-canvas");
        assert_eq!(frame.leaves[0].rect.width(), 800.0);
        assert!(frame.splits.is_empty());
    }

    #[test]
    fn unavailable_default_extension_hosts_release_geometry_without_mutating_actor_state() {
        let ctx = egui::Context::default();
        let value = shell(json!([
            node("root", "application", &["controls", "canvas"]),
            mount(
                "controls",
                "builtin_mount",
                "builtin:extension-host.canvas-controls"
            ),
            mount("canvas", "canvas_slot", "builtin:viewer-canvas"),
        ]));
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 500.0));
        let without_host = ShellTreeFrame::from_projection_with_mount_filter(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            rect,
            |mount| mount != "builtin:extension-host.canvas-controls",
        )
        .unwrap();
        assert_eq!(without_host.leaves.len(), 1);
        assert_eq!(without_host.leaves[0].rect.height(), 500.0);
        assert_eq!(value["layout"]["nodes"][0]["visible"], true);

        let with_host = ShellTreeFrame::from_projection_with_mount_filter(
            &ctx,
            &value,
            "builtin:viewer-canvas",
            rect,
            |_| true,
        )
        .unwrap();
        assert_eq!(with_host.leaves.len(), 2);
        assert!(with_host.leaves[1].rect.top() > 0.0);
    }

    #[test]
    fn changes_batch_into_one_layout_patch() {
        let changes = ShellTreeChanges {
            changes: vec![
                ShellTreeChange::Select {
                    tabs_id: "tabs".to_string(),
                    child_id: "layers".to_string(),
                },
                ShellTreeChange::Collapse {
                    node_id: "panel".to_string(),
                    collapsed: true,
                },
                ShellTreeChange::Split {
                    node_id: "split".to_string(),
                    axis: "vertical",
                    ratio: 0.4,
                    resizable: true,
                },
            ],
        };
        let patch = changes.patch_params();
        assert_eq!(patch["selected"], json!({"tabs":"layers"}));
        assert_eq!(patch["collapsed"], json!({"panel":true}));
        assert_eq!(patch["splits"]["split"]["axis"], "vertical");
        assert_eq!(patch["splits"]["split"]["resizable"], true);
        assert!((patch["splits"]["split"]["ratio"].as_f64().unwrap() - 0.4).abs() < 1e-6);
    }
}
