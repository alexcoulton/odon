//! Searchable native realization of the actor-owned command catalogue and palette presentation.

use eframe::egui;
use serde_json::Value;

use super::CommandPresentationInvocation;

const STATE_ID: &str = "odon.command-palette.state";

#[derive(Clone, Default)]
struct PaletteState {
    open: bool,
    query: String,
    selected: usize,
    focus_search: bool,
}

pub(crate) fn show(
    ctx: &egui::Context,
    command_surface: &Value,
) -> Option<CommandPresentationInvocation> {
    let palette = command_surface.get("palette")?;
    let mut state = ctx
        .data_mut(|data| data.get_temp::<PaletteState>(egui::Id::new(STATE_ID)))
        .unwrap_or_default();
    if super::command_shortcuts::consume(ctx, palette.get("shortcut")) {
        state.open = true;
        state.focus_search = true;
        state.query.clear();
        state.selected = 0;
    }
    if !state.open {
        ctx.data_mut(|data| data.insert_temp(egui::Id::new(STATE_ID), state));
        return None;
    }

    let max_results = palette
        .get("max_results")
        .and_then(Value::as_u64)
        .unwrap_or(20) as usize;
    let show_descriptions = palette
        .get("show_descriptions")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let title = palette
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("Commands");
    let placeholder = palette
        .get("placeholder")
        .and_then(Value::as_str)
        .unwrap_or("Search commands…");
    let mut open = true;
    let mut close_requested = false;
    let mut invoked = None;
    egui::Window::new(title)
        .id(egui::Id::new("odon.command-palette.window"))
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_width(560.0)
        .show(ctx, |ui| {
            let search = ui.add(
                egui::TextEdit::singleline(&mut state.query)
                    .hint_text(placeholder)
                    .desired_width(f32::INFINITY),
            );
            if state.focus_search {
                search.request_focus();
                state.focus_search = false;
            }
            if search.changed() {
                state.selected = 0;
            }
            let query = state.query.trim().to_ascii_lowercase();
            let mut commands = command_surface
                .get("commands")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|command| command_available(command))
                .filter(|command| command_matches(command, &query))
                .collect::<Vec<_>>();
            commands.sort_by_key(|command| {
                command
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            });
            commands.truncate(max_results);
            state.selected = state.selected.min(commands.len().saturating_sub(1));
            if ui.input(|input| input.key_pressed(egui::Key::ArrowDown)) && !commands.is_empty() {
                state.selected = (state.selected + 1).min(commands.len() - 1);
            }
            if ui.input(|input| input.key_pressed(egui::Key::ArrowUp)) {
                state.selected = state.selected.saturating_sub(1);
            }
            if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                close_requested = true;
            }
            ui.separator();
            if commands.is_empty() {
                ui.weak("No available commands match this search.");
            } else {
                egui::ScrollArea::vertical()
                    .max_height(420.0)
                    .show(ui, |ui| {
                        for (index, command) in commands.iter().enumerate() {
                            let command_id = command
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown");
                            let title = command
                                .get("title")
                                .and_then(Value::as_str)
                                .unwrap_or(command_id);
                            let shortcut =
                                super::command_shortcuts::neutral_label(command.get("shortcut"));
                            let label = shortcut
                                .map(|shortcut| format!("{title}    {shortcut}"))
                                .unwrap_or_else(|| title.to_string());
                            let response = ui.add(
                                egui::Button::new(label)
                                    .selected(index == state.selected)
                                    .min_size(egui::vec2(ui.available_width(), 24.0)),
                            );
                            if response.clicked() {
                                invoked = Some(command_invocation(command_id, command));
                            }
                            if show_descriptions
                                && let Some(description) =
                                    command.get("description").and_then(Value::as_str)
                            {
                                ui.small(description);
                            }
                        }
                    });
                if ui.input(|input| input.key_pressed(egui::Key::Enter))
                    && let Some(command) = commands.get(state.selected)
                    && let Some(command_id) = command.get("id").and_then(Value::as_str)
                {
                    invoked = Some(command_invocation(command_id, command));
                }
            }
        });
    if invoked.is_some() || close_requested {
        open = false;
    }
    state.open = open;
    ctx.data_mut(|data| data.insert_temp(egui::Id::new(STATE_ID), state));
    invoked
}

fn command_available(command: &Value) -> bool {
    command
        .pointer("/state/visible")
        .and_then(Value::as_bool)
        .unwrap_or(true)
        && command
            .pointer("/state/enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true)
}

fn command_invocation(command_id: &str, command: &Value) -> CommandPresentationInvocation {
    let checked = command
        .pointer("/state/checked")
        .and_then(Value::as_bool)
        .map(|checked| !checked);
    CommandPresentationInvocation {
        command_id: command_id.to_string(),
        checked,
    }
}

fn command_matches(command: &Value, query: &str) -> bool {
    query.is_empty()
        || ["id", "title", "description"]
            .into_iter()
            .filter_map(|field| command.get(field).and_then(Value::as_str))
            .any(|value| value.to_ascii_lowercase().contains(query))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn search_filters_mode_readiness_and_command_text() {
        let ready = json!({
            "id":"viewer.fit",
            "title":"Fit Image",
            "description":"Fit the image to the viewport.",
            "availability":{"modes":["single"]},
            "state":{"visible":true,"enabled":true,"checked":null},
        });
        assert!(command_available(&ready));
        assert!(command_matches(&ready, "viewport"));
        assert!(!command_matches(&ready, "measure"));
        assert!(!command_available(&json!({
            "availability":{"modes":["single"]},
            "readiness":{"state":"disconnected"},
            "state":{"visible":true,"enabled":false,"checked":null},
        }),));
    }

    #[test]
    fn shortcut_labels_are_platform_neutral() {
        assert_eq!(
            super::super::command_shortcuts::neutral_label(Some(
                &json!({"key":"p","modifiers":["primary","shift"]}),
            )),
            Some("Primary+Shift+P".to_string()),
        );
    }
}
