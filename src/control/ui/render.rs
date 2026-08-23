use std::collections::HashMap;
use std::time::{Duration, Instant};

use eframe::egui;
use serde_json::json;

use super::*;

mod components;

pub(super) use components::Interaction;
use components::render_component;

impl UiRegistry {
    pub fn render(&self, ctx: &egui::Context) {
        let mut interactions = Vec::new();
        let mut native_removed_extensions = Vec::new();
        {
            let mut state = self.state.lock().expect("UI registry poisoned");
            if !state.extensions.is_empty() {
                let mut remove = Vec::new();
                let mut reset = false;
                egui::Window::new("Odon Extensions")
                    .id(egui::Id::new("odon-extension-diagnostics"))
                    .default_open(false)
                    .resizable(true)
                    .show(ctx, |ui| {
                        ui.label("Python UI extensions");
                        for extension in state.extensions.values() {
                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "{} {} ({})",
                                    extension.name,
                                    extension.version,
                                    if extension.connected {
                                        "connected"
                                    } else {
                                        "disconnected"
                                    }
                                ));
                                if ui.button("Remove").clicked() {
                                    remove.push(extension.id.clone());
                                }
                            });
                        }
                        ui.separator();
                        if ui.button("Remove all extension UI").clicked() {
                            reset = true;
                        }
                    });
                if reset {
                    native_removed_extensions.extend(state.extensions.keys().cloned());
                    state.extensions.clear();
                    state.contributions.clear();
                } else {
                    state
                        .contributions
                        .retain(|item| !remove.contains(&item.extension_id));
                    for id in remove {
                        state.extensions.remove(&id);
                        native_removed_extensions.push(id);
                    }
                }
            }
            let extension_states = state
                .extensions
                .iter()
                .map(|(id, extension)| {
                    (
                        id.clone(),
                        (
                            extension.connected,
                            extension.owner_session_id.clone(),
                            matches!(extension.disconnect_policy, DisconnectPolicy::Retain),
                        ),
                    )
                })
                .collect::<HashMap<_, _>>();

            if has_location(&state.contributions, "top_bar.actions") {
                egui::TopBottomPanel::top("odon-extension-top-bar").show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        render_location(
                            ui,
                            &mut state.contributions,
                            "top_bar.actions",
                            &extension_states,
                            &mut interactions,
                            false,
                        );
                    });
                });
            }
            if has_location(&state.contributions, "status_bar") {
                egui::TopBottomPanel::bottom("odon-extension-status-bar").show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        render_location(
                            ui,
                            &mut state.contributions,
                            "status_bar",
                            &extension_states,
                            &mut interactions,
                            false,
                        );
                    });
                });
            }
            if has_location(&state.contributions, "left.sections") {
                egui::SidePanel::left("odon-extension-left-sections")
                    .default_width(260.0)
                    .resizable(true)
                    .show(ctx, |ui| {
                        ui.heading("Extensions");
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            render_location(
                                ui,
                                &mut state.contributions,
                                "left.sections",
                                &extension_states,
                                &mut interactions,
                                true,
                            );
                        });
                    });
            }

            let right_tabs = state
                .contributions
                .iter()
                .filter(|item| item.location == "right.tabs")
                .map(|item| (item.contribution_id.clone(), contribution_title(item)))
                .collect::<Vec<_>>();
            if !right_tabs.is_empty() {
                let mut selected = state
                    .selected_right_tab
                    .clone()
                    .filter(|id| right_tabs.iter().any(|(candidate, _)| candidate == id))
                    .unwrap_or_else(|| right_tabs[0].0.clone());
                egui::SidePanel::right("odon-extension-right-tabs")
                    .default_width(300.0)
                    .resizable(true)
                    .show(ctx, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for (id, title) in &right_tabs {
                                ui.selectable_value(&mut selected, id.clone(), title);
                            }
                        });
                        ui.separator();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            if let Some(contribution) = state
                                .contributions
                                .iter_mut()
                                .find(|item| item.contribution_id == selected)
                            {
                                render_contribution(
                                    ui,
                                    contribution,
                                    &extension_states,
                                    &mut interactions,
                                    false,
                                );
                            }
                        });
                    });
                state.selected_right_tab = Some(selected);
            }

            if has_location(&state.contributions, "canvas.controls") {
                egui::Area::new(egui::Id::new("odon-extension-canvas-controls"))
                    .anchor(egui::Align2::CENTER_TOP, [0.0, 48.0])
                    .show(ctx, |ui| {
                        egui::Frame::window(ui.style()).show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                render_location(
                                    ui,
                                    &mut state.contributions,
                                    "canvas.controls",
                                    &extension_states,
                                    &mut interactions,
                                    false,
                                );
                            });
                        });
                    });
            }
            if has_location(&state.contributions, "project.cards") {
                egui::Window::new("Extension project cards")
                    .id(egui::Id::new("odon-extension-project-cards"))
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        render_location(
                            ui,
                            &mut state.contributions,
                            "project.cards",
                            &extension_states,
                            &mut interactions,
                            true,
                        );
                    });
            }
        }
        if !native_removed_extensions.is_empty() {
            let revision = self.events.next_revision();
            for extension_id in native_removed_extensions {
                self.events.publish(
                    "ui.extensions.removed",
                    &extension_id,
                    revision,
                    json!({"extension_id": extension_id, "reason": "removed_in_native_ui"}),
                    None,
                    None,
                );
            }
        }
        let now = Instant::now();
        let mut deferred = self
            .deferred_interactions
            .lock()
            .expect("deferred UI interactions poisoned");
        for interaction in interactions {
            deferred.insert(interaction.key(), interaction);
        }
        let mut last_emitted = self.last_emitted.lock().expect("UI rate state poisoned");
        let ready = deferred
            .iter()
            .filter(|(key, interaction)| interaction.ready(now, last_emitted.get(*key).copied()))
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        let interactions = ready
            .into_iter()
            .filter_map(|key| {
                last_emitted.insert(key.clone(), now);
                deferred.remove(&key)
            })
            .collect::<Vec<_>>();
        if !deferred.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
        drop(last_emitted);
        drop(deferred);
        for interaction in interactions {
            let revision = self.events.next_revision();
            self.events.publish(
                format!(
                    "ui.extension:{}.{kind}",
                    interaction.extension_id,
                    kind = interaction.kind
                ),
                format!("ui:{}", interaction.component_id),
                revision,
                json!({
                    "component_id": interaction.component_id,
                    "value": interaction.value,
                    "action": interaction.action,
                }),
                None,
                None,
            );
            if let Some(action) = interaction.action {
                self.pending_actions
                    .lock()
                    .expect("UI action queue poisoned")
                    .push(UiAction {
                        extension_id: interaction.extension_id,
                        owner_session_id: interaction.owner_session_id,
                        component_id: interaction.component_id,
                        action,
                        value: interaction.value,
                    });
            }
        }
    }
}

fn has_location(contributions: &[ContributionSnapshot], location: &str) -> bool {
    contributions.iter().any(|item| item.location == location)
}

fn contribution_title(contribution: &ContributionSnapshot) -> String {
    contribution
        .root
        .title
        .clone()
        .or_else(|| contribution.root.label.clone())
        .unwrap_or_else(|| contribution.extension_id.clone())
}

fn render_location(
    ui: &mut egui::Ui,
    contributions: &mut [ContributionSnapshot],
    location: &str,
    extension_states: &HashMap<String, (bool, String, bool)>,
    interactions: &mut Vec<Interaction>,
    grouped: bool,
) {
    for contribution in contributions
        .iter_mut()
        .filter(|item| item.location == location)
    {
        render_contribution(ui, contribution, extension_states, interactions, grouped);
    }
}

fn render_contribution(
    ui: &mut egui::Ui,
    contribution: &mut ContributionSnapshot,
    extension_states: &HashMap<String, (bool, String, bool)>,
    interactions: &mut Vec<Interaction>,
    grouped: bool,
) {
    let title = contribution_title(contribution);
    let (connected, owner_session_id, retain_native_actions) = extension_states
        .get(&contribution.extension_id)
        .cloned()
        .unwrap_or((false, String::new(), false));
    let mut render = |ui: &mut egui::Ui, interactions: &mut Vec<Interaction>| {
        if !connected {
            ui.colored_label(egui::Color32::YELLOW, "Extension disconnected");
        }
        render_component(
            ui,
            &mut contribution.root,
            &contribution.extension_id,
            &owner_session_id,
            connected,
            retain_native_actions,
            interactions,
        );
    };
    if grouped {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.strong(title);
            render(ui, interactions);
        });
        ui.add_space(6.0);
    } else {
        render(ui, interactions);
    }
}
