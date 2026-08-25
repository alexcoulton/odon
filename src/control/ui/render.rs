use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use eframe::egui;
use serde_json::json;

use super::*;

mod components;

pub(super) use components::Interaction;
use components::render_component;

impl UiRegistry {
    pub fn render(&self, ctx: &egui::Context, _shell: Option<&serde_json::Value>) {
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
    }

    pub fn render_shell_mount(&self, ui: &mut egui::Ui, shell_mount: &str) -> bool {
        self.render_shell_mount_in_layout(ui, shell_mount, None)
    }

    pub fn shell_mount_available(&self, shell_mount: &str, shell: &serde_json::Value) -> bool {
        let Some(location) = default_host_location(shell_mount) else {
            return true;
        };
        let state = self.state.lock().expect("UI registry poisoned");
        let mounted = mounted_extension_contributions(Some(shell));
        has_location(&state.contributions, location, &mounted)
    }

    pub fn render_shell_mount_in_layout(
        &self,
        ui: &mut egui::Ui,
        shell_mount: &str,
        shell: Option<&serde_json::Value>,
    ) -> bool {
        let mut interactions = Vec::new();
        let rendered = {
            let mut state = self.state.lock().expect("UI registry poisoned");
            let extension_states = extension_states(&state);
            if let Some(location) = default_host_location(shell_mount) {
                let mounted = mounted_extension_contributions(shell);
                let grouped = matches!(location, "left.sections" | "right.tabs" | "project.cards");
                render_location(
                    ui,
                    &mut state.contributions,
                    location,
                    &extension_states,
                    &mut interactions,
                    grouped,
                    &mounted,
                );
                true
            } else if let Some(contribution) = state
                .contributions
                .iter_mut()
                .find(|contribution| contribution.shell_mount == shell_mount)
            {
                render_contribution(
                    ui,
                    contribution,
                    &extension_states,
                    &mut interactions,
                    false,
                );
                true
            } else {
                false
            }
        };
        self.commit_interactions(ui.ctx(), interactions);
        rendered
    }

    pub(in crate::control::ui) fn commit_interactions(
        &self,
        ctx: &egui::Context,
        interactions: Vec<Interaction>,
    ) {
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

fn extension_states(state: &State) -> HashMap<String, (bool, String, bool)> {
    state
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
        .collect()
}

fn default_host_location(shell_mount: &str) -> Option<&'static str> {
    Some(match shell_mount {
        "builtin:extension-host.top-bar-actions" => "top_bar.actions",
        "builtin:extension-host.status-bar" => "status_bar",
        "builtin:extension-host.left-sections" => "left.sections",
        "builtin:extension-host.right-tabs" => "right.tabs",
        "builtin:extension-host.canvas-controls" => "canvas.controls",
        "builtin:extension-host.project-cards" => "project.cards",
        _ => return None,
    })
}

fn has_location(
    contributions: &[ContributionSnapshot],
    location: &str,
    shell_mounts: &HashSet<String>,
) -> bool {
    contributions
        .iter()
        .any(|item| item.location == location && !shell_mounts.contains(&item.shell_mount))
}

fn mounted_extension_contributions(shell: Option<&serde_json::Value>) -> HashSet<String> {
    shell
        .and_then(|shell| shell.pointer("/layout/nodes"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|node| {
            node.get("type").and_then(serde_json::Value::as_str) == Some("extension_mount")
        })
        .filter_map(|node| node.get("mount").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect()
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
    shell_mounts: &HashSet<String>,
) {
    for contribution in contributions
        .iter_mut()
        .filter(|item| item.location == location && !shell_mounts.contains(&item.shell_mount))
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
    let (transport_connected, owner_session_id, retain_native_actions) = extension_states
        .get(&contribution.extension_id)
        .cloned()
        .unwrap_or((false, String::new(), false));
    let connected = transport_connected && contribution.readiness == "ready";
    let retain_native_actions = retain_native_actions && contribution.readiness == "disconnected";
    let mut render = |ui: &mut egui::Ui, interactions: &mut Vec<Interaction>| {
        if contribution.readiness != "ready" {
            let message = match contribution.readiness.as_str() {
                "disconnected" => "Extension disconnected",
                "incompatible" => "Extension version is incompatible with this retained mount",
                "not_ready" => "Extension is connected but not ready",
                _ => "Extension mount is unavailable",
            };
            ui.colored_label(egui::Color32::YELLOW, message);
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
