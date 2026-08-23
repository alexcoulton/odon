use std::time::Instant;

use eframe::egui;

use super::*;

fn event_policy_kind(component: &Component) -> &str {
    component
        .event_policy
        .as_ref()
        .and_then(|policy| policy.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("immediate")
}

#[derive(Debug, Clone)]
pub(in crate::control::ui) struct Interaction {
    pub(in crate::control::ui) extension_id: String,
    pub(in crate::control::ui) owner_session_id: String,
    pub(in crate::control::ui) component_id: String,
    pub(in crate::control::ui) kind: String,
    pub(in crate::control::ui) value: Value,
    pub(in crate::control::ui) action: Option<Value>,
    pub(in crate::control::ui) event_policy: Option<Value>,
    pub(in crate::control::ui) occurred_at: Instant,
}

impl Interaction {
    pub(in crate::control::ui) fn key(&self) -> String {
        format!("{}:{}", self.extension_id, self.component_id)
    }

    pub(in crate::control::ui) fn ready(&self, now: Instant, last: Option<Instant>) -> bool {
        let policy = self
            .event_policy
            .as_ref()
            .and_then(|policy| policy.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("immediate");
        let requested_ms = self
            .event_policy
            .as_ref()
            .and_then(|policy| policy.get("milliseconds"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if policy == "debounce" {
            return now.duration_since(self.occurred_at)
                >= Duration::from_millis(requested_ms.max(33));
        }
        let interval = if policy == "throttle" {
            requested_ms.max(33)
        } else {
            33
        };
        last.is_none_or(|last| now.duration_since(last) >= Duration::from_millis(interval))
    }
}

pub(super) fn render_component(
    ui: &mut egui::Ui,
    component: &mut Component,
    extension_id: &str,
    owner_session_id: &str,
    connected: bool,
    retain_native_actions: bool,
    interactions: &mut Vec<Interaction>,
) {
    if !component.visible {
        return;
    }
    let interactive = matches!(
        component.kind.as_str(),
        "button"
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
    );
    let works_without_python = component
        .action
        .as_ref()
        .and_then(|action| action.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| matches!(kind, "command" | "bind"));
    let enabled = component.enabled
        && (!interactive || connected || (retain_native_actions && works_without_python));
    ui.add_enabled_ui(enabled, |ui| match component.kind.as_str() {
        "panel" | "column" => {
            for child in &mut component.children {
                render_component(
                    ui,
                    child,
                    extension_id,
                    owner_session_id,
                    connected,
                    retain_native_actions,
                    interactions,
                );
            }
        }
        "row" => {
            ui.horizontal(|ui| {
                for child in &mut component.children {
                    render_component(
                        ui,
                        child,
                        extension_id,
                        owner_session_id,
                        connected,
                        retain_native_actions,
                        interactions,
                    );
                }
            });
        }
        "grid" => {
            egui::Grid::new((extension_id, &component.id))
                .num_columns(component.columns.unwrap_or(2))
                .show(ui, |ui| {
                    let columns = component.columns.unwrap_or(2);
                    for (index, child) in component.children.iter_mut().enumerate() {
                        render_component(
                            ui,
                            child,
                            extension_id,
                            owner_session_id,
                            connected,
                            retain_native_actions,
                            interactions,
                        );
                        if (index + 1) % columns == 0 {
                            ui.end_row();
                        }
                    }
                });
        }
        "tabs" => {
            let mut selected = component
                .value
                .as_str()
                .filter(|id| component.children.iter().any(|child| child.id == *id))
                .map(str::to_string)
                .or_else(|| component.children.first().map(|child| child.id.clone()));
            ui.horizontal_wrapped(|ui| {
                for child in &component.children {
                    ui.selectable_value(
                        &mut selected,
                        Some(child.id.clone()),
                        child
                            .title
                            .as_deref()
                            .or(child.label.as_deref())
                            .unwrap_or(&child.id),
                    );
                }
            });
            ui.separator();
            component.value = selected.clone().map(Value::String).unwrap_or(Value::Null);
            if let Some(child) = component
                .children
                .iter_mut()
                .find(|child| Some(&child.id) == selected.as_ref())
            {
                render_component(
                    ui,
                    child,
                    extension_id,
                    owner_session_id,
                    connected,
                    retain_native_actions,
                    interactions,
                );
            }
        }
        "scroll" => {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for child in &mut component.children {
                    render_component(
                        ui,
                        child,
                        extension_id,
                        owner_session_id,
                        connected,
                        retain_native_actions,
                        interactions,
                    );
                }
            });
        }
        "group" => {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                if let Some(label) = &component.label {
                    ui.strong(label);
                }
                for child in &mut component.children {
                    render_component(
                        ui,
                        child,
                        extension_id,
                        owner_session_id,
                        connected,
                        retain_native_actions,
                        interactions,
                    );
                }
            });
        }
        "collapsible" => {
            ui.collapsing(component.label.as_deref().unwrap_or("Details"), |ui| {
                for child in &mut component.children {
                    render_component(
                        ui,
                        child,
                        extension_id,
                        owner_session_id,
                        connected,
                        retain_native_actions,
                        interactions,
                    );
                }
            });
        }
        "text" | "markdown" | "status" => {
            ui.label(
                component
                    .value
                    .as_str()
                    .or(component.label.as_deref())
                    .unwrap_or_default(),
            );
        }
        "warning" | "error" => {
            let color = if component.kind == "error" {
                egui::Color32::LIGHT_RED
            } else {
                egui::Color32::YELLOW
            };
            ui.colored_label(
                color,
                component
                    .value
                    .as_str()
                    .or(component.label.as_deref())
                    .unwrap_or_default(),
            );
        }
        "spinner" => {
            ui.horizontal(|ui| {
                ui.spinner();
                if let Some(label) = &component.label {
                    ui.label(label);
                }
            });
        }
        "separator" => {
            ui.separator();
        }
        "spacer" => {
            ui.add_space(component.value.as_f64().unwrap_or(8.0) as f32);
        }
        "button" => {
            if ui
                .button(component.label.as_deref().unwrap_or("Action"))
                .clicked()
            {
                interactions.push(interaction(
                    component,
                    extension_id,
                    owner_session_id,
                    "action",
                ));
            }
        }
        "toggle" | "checkbox" => {
            let mut value = component.value.as_bool().unwrap_or(false);
            if ui
                .checkbox(&mut value, component.label.as_deref().unwrap_or_default())
                .changed()
            {
                component.value = json!(value);
                interactions.push(interaction(
                    component,
                    extension_id,
                    owner_session_id,
                    "input",
                ));
            }
        }
        "slider" | "number" | "integer" => {
            let mut value = component.value.as_f64().unwrap_or_default();
            let range = component.minimum.unwrap_or(0.0)..=component.maximum.unwrap_or(1.0);
            let response = ui.add(
                egui::Slider::new(&mut value, range)
                    .text(component.label.as_deref().unwrap_or_default()),
            );
            let changed = response.changed();
            if changed {
                if component.kind == "integer" {
                    value = value.round();
                }
                component.value = json!(value);
            }
            let on_commit = event_policy_kind(component) == "commit";
            if (changed && (!on_commit || !response.dragged()))
                || (on_commit && response.drag_stopped())
            {
                interactions.push(interaction(
                    component,
                    extension_id,
                    owner_session_id,
                    "input",
                ));
            }
        }
        "text_input" => {
            let mut value = component.value.as_str().unwrap_or_default().to_string();
            let response = ui
                .horizontal(|ui| {
                    if let Some(label) = &component.label {
                        ui.label(label);
                    }
                    ui.text_edit_singleline(&mut value)
                })
                .inner;
            let changed = response.changed();
            if changed {
                component.value = json!(value);
            }
            let on_commit = event_policy_kind(component) == "commit";
            let enter_pressed =
                response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (changed && !on_commit) || (on_commit && (response.lost_focus() || enter_pressed)) {
                interactions.push(interaction(
                    component,
                    extension_id,
                    owner_session_id,
                    "input",
                ));
            }
        }
        "select" | "radio" => {
            let mut value = component.value.clone();
            egui::ComboBox::from_id_salt((extension_id, &component.id))
                .selected_text(value.as_str().unwrap_or("Select…"))
                .show_ui(ui, |ui| {
                    for option in &component.options {
                        ui.selectable_value(
                            &mut value,
                            option.clone(),
                            option.as_str().unwrap_or("option"),
                        );
                    }
                });
            if value != component.value {
                component.value = value;
                interactions.push(interaction(
                    component,
                    extension_id,
                    owner_session_id,
                    "input",
                ));
            }
        }
        "multi_select" => {
            let mut values = component.value.as_array().cloned().unwrap_or_default();
            let mut changed = false;
            egui::ComboBox::from_id_salt((extension_id, &component.id))
                .selected_text(format!("{} selected", values.len()))
                .show_ui(ui, |ui| {
                    for option in &component.options {
                        let mut selected = values.contains(option);
                        if ui
                            .checkbox(&mut selected, option.as_str().unwrap_or("option"))
                            .changed()
                        {
                            changed = true;
                            if selected {
                                values.push(option.clone());
                            } else {
                                values.retain(|value| value != option);
                            }
                        }
                    }
                });
            if changed {
                component.value = Value::Array(values);
                interactions.push(interaction(
                    component,
                    extension_id,
                    owner_session_id,
                    "input",
                ));
            }
        }
        "color" => {
            let mut color = parse_color(component.value.as_str()).unwrap_or(egui::Color32::WHITE);
            let response = ui
                .horizontal(|ui| {
                    if let Some(label) = &component.label {
                        ui.label(label);
                    }
                    ui.color_edit_button_srgba(&mut color)
                })
                .inner;
            if response.changed() {
                component.value = Value::String(format!(
                    "#{:02x}{:02x}{:02x}{:02x}",
                    color.r(),
                    color.g(),
                    color.b(),
                    color.a()
                ));
                interactions.push(interaction(
                    component,
                    extension_id,
                    owner_session_id,
                    "input",
                ));
            }
        }
        "progress" => {
            ui.add(
                egui::ProgressBar::new(component.value.as_f64().unwrap_or_default() as f32)
                    .show_percentage(),
            );
        }
        _ => {
            ui.label(format!("Unsupported component: {}", component.kind));
        }
    });
    if let Some(help) = &component.help {
        ui.small(help);
    }
}

fn parse_color(value: Option<&str>) -> Option<egui::Color32> {
    let value = value?.strip_prefix('#')?;
    if !matches!(value.len(), 6 | 8) {
        return None;
    }
    let byte = |offset| u8::from_str_radix(&value[offset..offset + 2], 16).ok();
    Some(egui::Color32::from_rgba_unmultiplied(
        byte(0)?,
        byte(2)?,
        byte(4)?,
        if value.len() == 8 { byte(6)? } else { 255 },
    ))
}

fn interaction(
    component: &Component,
    extension_id: &str,
    owner_session_id: &str,
    kind: &str,
) -> Interaction {
    Interaction {
        extension_id: extension_id.to_string(),
        owner_session_id: owner_session_id.to_string(),
        component_id: component.id.clone(),
        kind: kind.to_string(),
        value: component.value.clone(),
        action: component.action.clone(),
        event_policy: component.event_policy.clone(),
        occurred_at: Instant::now(),
    }
}
