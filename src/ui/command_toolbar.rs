//! Native egui realization of the actor-owned command-toolbar presentation.

use std::collections::HashMap;

use eframe::egui;
use serde_json::Value;

use super::CommandPresentationInvocation;

pub(crate) fn render(
    ui: &mut egui::Ui,
    shell_projection: &Value,
) -> Option<CommandPresentationInvocation> {
    let surface = shell_projection.get("_command_surface")?;
    let commands = surface
        .get("commands")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|command| Some((command.get("id")?.as_str()?, command)))
        .collect::<HashMap<_, _>>();
    let groups = surface
        .pointer("/toolbar/groups")
        .and_then(Value::as_array)?;
    let mut invoked = None;
    ui.horizontal(|ui| {
        for (group_index, group) in groups.iter().enumerate() {
            if group_index > 0 {
                ui.separator();
            }
            if let Some(title) = group.get("title").and_then(Value::as_str) {
                ui.strong(title);
            }
            for item in group
                .get("items")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(command_id) = item.get("command_id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(command) = commands.get(command_id).copied() else {
                    continue;
                };
                if command.pointer("/state/visible").and_then(Value::as_bool) == Some(false) {
                    continue;
                }
                let label = item
                    .get("label")
                    .and_then(Value::as_str)
                    .or_else(|| command.get("title").and_then(Value::as_str))
                    .unwrap_or(command_id);
                let icon = item
                    .get("icon")
                    .and_then(Value::as_str)
                    .or_else(|| command.get("icon").and_then(Value::as_str));
                let show_label = item
                    .get("show_label")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let button_text = match (icon, show_label) {
                    (Some(icon), true) => format!("{icon} {label}"),
                    (Some(icon), false) => icon.to_string(),
                    (None, _) => label.to_string(),
                };
                let enabled = command_enabled(command);
                let checked = command.pointer("/state/checked").and_then(Value::as_bool);
                let mut tooltip = item
                    .get("tooltip")
                    .and_then(Value::as_str)
                    .or_else(|| command.get("description").and_then(Value::as_str))
                    .unwrap_or("")
                    .to_string();
                if !enabled {
                    let reason = command
                        .pointer("/state/reasons")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !reason.is_empty() {
                        if !tooltip.is_empty() {
                            tooltip.push_str("\n\nUnavailable: ");
                        }
                        tooltip.push_str(&reason);
                    }
                }
                let response = ui
                    .add_enabled(
                        enabled,
                        egui::Button::new(button_text.clone()).selected(checked.unwrap_or(false)),
                    )
                    .on_hover_text(tooltip.clone());
                if let Some(checked) = checked {
                    response.widget_info(|| {
                        egui::WidgetInfo::selected(
                            egui::WidgetType::Button,
                            enabled,
                            checked,
                            &button_text,
                        )
                    });
                }
                if !tooltip.is_empty() {
                    response.ctx.accesskit_node_builder(response.id, |node| {
                        node.set_description(tooltip.clone());
                    });
                }
                if response.clicked() {
                    invoked = Some(command_invocation(command_id, command));
                }
            }
        }
    });
    invoked
}

fn command_enabled(command: &Value) -> bool {
    command
        .pointer("/state/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

fn command_invocation(command_id: &str, command: &Value) -> CommandPresentationInvocation {
    CommandPresentationInvocation {
        command_id: command_id.to_string(),
        checked: command
            .pointer("/state/checked")
            .and_then(Value::as_bool)
            .map(|checked| !checked),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn interactive_projection(checked: bool, enabled: bool) -> Value {
        json!({
            "_command_surface":{
                "commands":[{
                    "id":"viewer.scale_bar.toggle",
                    "title":"Scale bar",
                    "description":"Toggle the scale bar.",
                    "state":{
                        "visible":true,
                        "enabled":enabled,
                        "checked":checked,
                        "reasons":if enabled { json!([]) } else { json!(["Viewer unavailable."]) },
                    }
                }],
                "toolbar":{"groups":[{
                    "id":"toolbar-group:view",
                    "items":[{
                        "id":"toolbar-item:scale",
                        "command_id":"viewer.scale_bar.toggle",
                        "label":"Scale bar",
                        "tooltip":"Show or hide the scale bar."
                    }]
                }]}
            }
        })
    }

    #[test]
    fn absent_toolbar_projection_is_a_safe_no_op() {
        let context = egui::Context::default();
        context.begin_pass(Default::default());
        egui::CentralPanel::default().show(&context, |ui| {
            assert_eq!(render(ui, &json!({})), None);
        });
        let _ = context.end_pass();
    }

    #[test]
    fn all_handler_types_share_mode_and_readiness_enablement() {
        for handler in [
            json!({"type":"native","action":"settings"}),
            json!({"type":"control","method":"ui.shell.recover","params":{}}),
            json!({"type":"event","event":"measure"}),
        ] {
            let command = json!({
                "handler":handler,
                "availability":{"modes":["single"]},
                "readiness":{"state":"ready"},
            });
            let mut enabled = command.clone();
            enabled["state"] = json!({"visible":true,"enabled":true,"checked":null});
            assert!(command_enabled(&enabled));
            enabled["state"]["enabled"] = json!(false);
            assert!(!command_enabled(&enabled));
        }
        assert!(!command_enabled(&json!({
            "handler":{"type":"event","event":"measure"},
            "availability":{"modes":["single"]},
            "readiness":{"state":"disconnected"},
            "state":{"visible":true,"enabled":false,"checked":null},
        }),));
    }

    #[test]
    fn checked_toolbar_presentations_submit_the_toggled_state() {
        assert_eq!(
            command_invocation(
                "viewer.scale_bar.toggle",
                &json!({"state":{"checked":true}}),
            ),
            CommandPresentationInvocation {
                command_id: "viewer.scale_bar.toggle".to_string(),
                checked: Some(false),
            }
        );
        assert_eq!(
            command_invocation("project.save", &json!({"state":{"checked":null}})).checked,
            None
        );
    }

    #[test]
    fn toolbar_buttons_publish_accessible_state_and_accept_accesskit_clicks() {
        use egui::accesskit::{Action, ActionRequest, Role, Toggled};

        let context = egui::Context::default();
        context.enable_accesskit();
        context.begin_pass(Default::default());
        egui::CentralPanel::default().show(&context, |ui| {
            assert_eq!(render(ui, &interactive_projection(true, true)), None);
        });
        let output = context.end_pass();
        let update = output.platform_output.accesskit_update.unwrap();
        let (target, node) = update
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some("Scale bar"))
            .expect("toolbar button is present in the accessibility tree");
        assert_eq!(node.role(), Role::Button);
        assert_eq!(node.toggled(), Some(Toggled::True));
        assert_eq!(node.description(), Some("Show or hide the scale bar."));

        context.begin_pass(egui::RawInput {
            events: vec![egui::Event::AccessKitActionRequest(ActionRequest {
                action: Action::Click,
                target: *target,
                data: None,
            })],
            ..Default::default()
        });
        let mut invocation = None;
        egui::CentralPanel::default().show(&context, |ui| {
            invocation = render(ui, &interactive_projection(true, true));
        });
        let _ = context.end_pass();
        assert_eq!(
            invocation,
            Some(CommandPresentationInvocation {
                command_id: "viewer.scale_bar.toggle".to_string(),
                checked: Some(false),
            })
        );
    }

    #[test]
    fn toolbar_buttons_are_reconciled_and_keyboard_navigable() {
        let context = egui::Context::default();
        context.begin_pass(Default::default());
        egui::CentralPanel::default().show(&context, |ui| {
            assert_eq!(render(ui, &interactive_projection(false, false)), None);
        });
        let _ = context.end_pass();

        context.begin_pass(egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Tab,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        });
        egui::CentralPanel::default().show(&context, |ui| {
            assert_eq!(render(ui, &interactive_projection(false, true)), None);
        });
        let _ = context.end_pass();

        context.begin_pass(egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        });
        let mut invocation = None;
        egui::CentralPanel::default().show(&context, |ui| {
            invocation = render(ui, &interactive_projection(false, true));
        });
        let _ = context.end_pass();
        assert_eq!(
            invocation,
            Some(CommandPresentationInvocation {
                command_id: "viewer.scale_bar.toggle".to_string(),
                checked: Some(true),
            })
        );
    }
}
