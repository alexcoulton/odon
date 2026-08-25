//! Native realization of the actor-owned mosaic shell projection.

use super::*;

impl MosaicViewerApp {
    pub(crate) fn set_extension_ui_registry(
        &mut self,
        registry: std::sync::Arc<odon::control::UiRegistry>,
    ) {
        self.extension_ui_registry = Some(registry);
    }

    pub(crate) fn apply_control_shell_projection(&mut self, shell: &serde_json::Value) {
        self.control_shell_layout.apply_projection(shell);
        for plan in crate::ui::shell_layout::native_panel_plans(shell, "builtin:mosaic-canvas") {
            match plan.selected_mount.as_deref() {
                Some("builtin:layers") => {
                    self.left_tab = LeftTab::Layers;
                }
                Some("builtin:project") => {
                    self.left_tab = LeftTab::Project;
                }
                Some("builtin:properties") => {
                    self.right_tab = RightTab::Properties;
                }
                Some("builtin:views") => {
                    self.right_tab = RightTab::Views;
                }
                Some("builtin:mosaic-layout") => {
                    self.right_tab = RightTab::Layout;
                }
                Some("builtin:memory") => {
                    self.right_tab = RightTab::Memory;
                }
                _ => {}
            }
        }
        self.control_shell_projection = shell.clone();
    }

    pub(super) fn shell_node_visible(&self, id: &str, default: bool) -> bool {
        if let Some(mount) = legacy_mount(id)
            && let Some(visible) = self
                .control_shell_layout
                .mount_visible(&self.control_shell_projection, mount)
        {
            return visible;
        }
        shell_node(&self.control_shell_projection, id)
            .and_then(|node| node.get("visible"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(default)
    }

    pub(super) fn shell_left_tabs(&self) -> Vec<left_panel::TabSpec<LeftTab>> {
        let has_layout = self.control_shell_projection.get("layout").is_some();
        let panel_plan = layout_panel_plan(
            &self.control_shell_projection,
            crate::ui::shell_layout::PanelSide::Left,
        );
        let layout_order = panel_plan
            .as_ref()
            .map(|plan| plan.mounts.clone())
            .unwrap_or_default();
        let desired_width = panel_plan.as_ref().and_then(|plan| plan.width);
        let mut order = if has_layout {
            layout_order
                .into_iter()
                .filter_map(|mount| match mount.as_str() {
                    "builtin:layers" => Some("builtin:mosaic.layers".to_string()),
                    "builtin:project" => Some("builtin:mosaic.project".to_string()),
                    _ => None,
                })
                .collect()
        } else {
            ordered_children(&self.control_shell_projection, "builtin:mosaic.left-tabs")
        };
        if order.is_empty() && !has_layout {
            order = vec![
                "builtin:mosaic.layers".into(),
                "builtin:mosaic.project".into(),
            ];
        }
        order
            .iter()
            .filter_map(|id| match id.as_str() {
                "builtin:mosaic.layers" => Some(left_panel::TabSpec {
                    tab: LeftTab::Layers,
                    label: "Layers",
                    panel_key: "layers",
                    default_width: desired_width.unwrap_or(360.0),
                    scroll: true,
                }),
                "builtin:mosaic.project" => Some(left_panel::TabSpec {
                    tab: LeftTab::Project,
                    label: "Project",
                    panel_key: "project",
                    default_width: desired_width.unwrap_or(420.0),
                    scroll: false,
                }),
                _ => None,
            })
            .collect()
    }

    pub(super) fn shell_right_tabs(&self) -> Vec<right_panel::TabSpec<RightTab>> {
        let has_layout = self.control_shell_projection.get("layout").is_some();
        let panel_plan = layout_panel_plan(
            &self.control_shell_projection,
            crate::ui::shell_layout::PanelSide::Right,
        );
        let layout_order = panel_plan
            .as_ref()
            .map(|plan| plan.mounts.clone())
            .unwrap_or_default();
        let mut order = if has_layout {
            layout_order
                .into_iter()
                .filter_map(|mount| match mount.as_str() {
                    "builtin:properties" => Some("builtin:mosaic.properties".to_string()),
                    "builtin:views" => Some("builtin:mosaic.views".to_string()),
                    "builtin:mosaic-layout" => Some("builtin:mosaic.layout".to_string()),
                    "builtin:memory" => Some("builtin:mosaic.memory".to_string()),
                    _ => None,
                })
                .collect()
        } else {
            ordered_children(&self.control_shell_projection, "builtin:mosaic.right-tabs")
        };
        if order.is_empty() && !has_layout {
            order = ["properties", "views", "layout", "memory"]
                .into_iter()
                .map(|id| format!("builtin:mosaic.{id}"))
                .collect();
        }
        order
            .iter()
            .filter_map(|id| {
                let (tab, label) = match id.as_str() {
                    "builtin:mosaic.properties" => (RightTab::Properties, "Properties"),
                    "builtin:mosaic.views" => (RightTab::Views, "Views"),
                    "builtin:mosaic.layout" => (RightTab::Layout, "Layout"),
                    "builtin:mosaic.memory" => (RightTab::Memory, "Memory"),
                    _ => return None,
                };
                Some(right_panel::TabSpec {
                    tab,
                    label,
                    scroll: true,
                })
            })
            .collect()
    }

    pub(super) fn shell_right_panel_width(&self, default: f32) -> f32 {
        layout_panel_plan(
            &self.control_shell_projection,
            crate::ui::shell_layout::PanelSide::Right,
        )
        .and_then(|plan| plan.width)
        .unwrap_or(default)
    }

    pub(super) fn ui_actor_shell_tree(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> bool {
        let projection = self.control_shell_projection.clone();
        let extension_registry = self.extension_ui_registry.clone();
        let Some(frame) = crate::ui::shell_tree::ShellTreeFrame::from_projection_with_mount_filter(
            ctx,
            &projection,
            "builtin:mosaic-canvas",
            ui.available_rect_before_wrap(),
            |mount| {
                extension_registry
                    .as_ref()
                    .is_none_or(|registry| registry.shell_mount_available(mount, &projection))
            },
        ) else {
            return false;
        };
        let shell_geometry = frame
            .node_rects()
            .map(|(id, rect)| {
                (
                    id.to_string(),
                    [rect.min.x, rect.min.y, rect.max.x, rect.max.y],
                )
            })
            .collect::<crate::ui::shell_inspector::ShellGeometry>();
        let changes = frame.show(ui, |mount_ui, mount, configuration| match mount {
            "builtin:mosaic-top-bar" => {
                self.ui_mosaic_top_bar(mount_ui, ctx, configuration)
            }
            "builtin:command-toolbar" => {
                if let Some(invocation) =
                    crate::ui::command_toolbar::render(mount_ui, &projection)
                {
                    let mut params = serde_json::json!({"command_id":invocation.command_id});
                    if let Some(checked) = invocation.checked {
                        params["checked"] = serde_json::json!(checked);
                    }
                    self.submit_native_control_intent(
                        "ui.commands.execute",
                        params,
                    );
                }
            }
            "builtin:mosaic-canvas" => self.ui_canvas(mount_ui, ctx),
            "builtin:shell-inspector" => crate::ui::shell_inspector::render(
                mount_ui,
                &projection,
                &shell_geometry,
            ),
            "builtin:help" => crate::ui::help::render_help_browser(mount_ui, "mosaic-shell"),
            mount if mount.starts_with("extension:") => {
                if !extension_registry
                    .as_ref()
                    .is_some_and(|registry| {
                        registry.render_shell_mount_in_layout(mount_ui, mount, Some(&projection))
                    })
                {
                    mount_ui.weak(format!(
                        "Extension shell mount '{mount}' is not connected to a registered contribution."
                    ));
                }
            }
            mount if mount.starts_with("builtin:extension-host.") => {
                if !extension_registry.as_ref().is_some_and(|registry| {
                    registry.render_shell_mount_in_layout(mount_ui, mount, Some(&projection))
                }) {
                    mount_ui.weak("No extension contributions are registered for this host.");
                }
            }
            "builtin:project" => self.ui_shell_builtin(mount, mount_ui, ctx),
            _ => {
                egui::ScrollArea::vertical()
                    .id_salt(("odon-shell-mount-scroll", mount))
                    .auto_shrink([false, false])
                    .show(mount_ui, |body| self.ui_shell_builtin(mount, body, ctx));
            }
        });
        for (id, rect) in frame.node_rects() {
            self.control_shell_layout
                .record_rect(id, [rect.min.x, rect.min.y, rect.max.x, rect.max.y]);
        }
        if !changes.is_empty() {
            self.submit_native_control_intent("ui.shell.patch_layout", changes.patch_params());
        }
        true
    }
}

fn legacy_mount(id: &str) -> Option<&'static str> {
    match id {
        "builtin:mosaic.top-bar" => Some("builtin:mosaic-top-bar"),
        "builtin:mosaic.canvas" => Some("builtin:mosaic-canvas"),
        "builtin:mosaic.layers" => Some("builtin:layers"),
        "builtin:mosaic.project" => Some("builtin:project"),
        "builtin:mosaic.properties" => Some("builtin:properties"),
        "builtin:mosaic.views" => Some("builtin:views"),
        "builtin:mosaic.layout" => Some("builtin:mosaic-layout"),
        "builtin:mosaic.memory" => Some("builtin:memory"),
        _ => None,
    }
}

fn shell_node<'a>(shell: &'a serde_json::Value, id: &str) -> Option<&'a serde_json::Value> {
    shell
        .get("nodes")?
        .as_array()?
        .iter()
        .find(|node| node.get("id").and_then(serde_json::Value::as_str) == Some(id))
}

fn ordered_children(shell: &serde_json::Value, id: &str) -> Vec<String> {
    shell_node(shell, id)
        .and_then(|node| node.get("children"))
        .and_then(serde_json::Value::as_array)
        .map(|children| {
            children
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn layout_panel_plan(
    shell: &serde_json::Value,
    side: crate::ui::shell_layout::PanelSide,
) -> Option<crate::ui::shell_layout::NativePanelPlan> {
    crate::ui::shell_layout::native_panel_plans(shell, "builtin:mosaic-canvas")
        .into_iter()
        .find(|plan| plan.side == side)
}
