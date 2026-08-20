use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{ControlError, ControlErrorKind, EventHub};

const MAX_COMPONENTS: usize = 512;
const MAX_DEPTH: usize = 16;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DisconnectPolicy {
    Remove,
    Disable,
    Retain,
}

impl Default for DisconnectPolicy {
    fn default() -> Self {
        Self::Remove
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub help: Option<String>,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub value: Value,
    #[serde(default)]
    pub minimum: Option<f64>,
    #[serde(default)]
    pub maximum: Option<f64>,
    #[serde(default)]
    pub options: Vec<Value>,
    #[serde(default)]
    pub columns: Option<usize>,
    #[serde(default)]
    pub action: Option<Value>,
    #[serde(default)]
    pub event_policy: Option<Value>,
    #[serde(default)]
    pub children: Vec<Component>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterExtension {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub disconnect_policy: DisconnectPolicy,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExtensionSnapshot {
    pub id: String,
    pub name: String,
    pub version: String,
    pub requested_capabilities: Vec<String>,
    pub granted_capabilities: Vec<String>,
    pub disconnect_policy: DisconnectPolicy,
    pub owner_session_id: String,
    pub connected: bool,
    pub revision: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterContribution {
    pub extension_id: String,
    #[serde(default)]
    pub contribution_id: Option<String>,
    pub location: String,
    pub root: Component,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContributionSnapshot {
    pub contribution_id: String,
    pub extension_id: String,
    pub location: String,
    pub root: Component,
    pub revision: u64,
}

#[derive(Debug, Default)]
struct State {
    extensions: HashMap<String, ExtensionSnapshot>,
    contributions: Vec<ContributionSnapshot>,
    selected_right_tab: Option<String>,
}

#[derive(Debug)]
pub struct UiRegistry {
    state: Mutex<State>,
    events: Arc<EventHub>,
    pending_actions: Mutex<Vec<UiAction>>,
    deferred_interactions: Mutex<HashMap<String, Interaction>>,
    last_emitted: Mutex<HashMap<String, Instant>>,
}

#[derive(Debug, Clone)]
pub struct UiAction {
    pub extension_id: String,
    pub owner_session_id: String,
    pub component_id: String,
    pub action: Value,
    pub value: Value,
}

impl UiRegistry {
    pub fn shared(events: Arc<EventHub>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            events,
            pending_actions: Mutex::new(Vec::new()),
            deferred_interactions: Mutex::new(HashMap::new()),
            last_emitted: Mutex::new(HashMap::new()),
        })
    }

    pub fn register_extension(
        &self,
        params: Value,
        session_id: &str,
    ) -> Result<ExtensionSnapshot, ControlError> {
        let request: RegisterExtension = serde_json::from_value(params).map_err(|error| {
            ControlError::invalid_params(
                "ui.extensions.register",
                format!("invalid extension: {error}"),
            )
        })?;
        if request.id.trim().is_empty()
            || !request.id.contains('.')
            || request.id.chars().any(char::is_whitespace)
        {
            return Err(ControlError::invalid_params(
                "ui.extensions.register",
                "extension id must be a non-empty reverse-domain-style identifier",
            ));
        }
        if request.name.trim().is_empty() || request.version.trim().is_empty() {
            return Err(ControlError::invalid_params(
                "ui.extensions.register",
                "extension name and version must not be empty",
            ));
        }
        let allowed = [
            "ui.panels",
            "ui.actions",
            "viewer.read",
            "viewer.write",
            "viewer.layers.read",
            "viewer.layers.write",
            "data.read",
            "data.write",
        ];
        let granted_capabilities = request
            .capabilities
            .iter()
            .filter(|capability| allowed.contains(&capability.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let mut state = self.state.lock().expect("UI registry poisoned");
        if let Some(existing) = state.extensions.get(&request.id) {
            if existing.connected {
                return Err(ControlError::new(
                    ControlErrorKind::Conflict,
                    format!("extension '{}' is already registered", request.id),
                ));
            }
            state.extensions.remove(&request.id);
            state
                .contributions
                .retain(|item| item.extension_id != request.id);
        }
        let revision = self.events.next_revision();
        let snapshot = ExtensionSnapshot {
            id: request.id.clone(),
            name: request.name,
            version: request.version,
            requested_capabilities: request.capabilities,
            granted_capabilities,
            disconnect_policy: request.disconnect_policy,
            owner_session_id: session_id.to_string(),
            connected: true,
            revision,
        };
        state
            .extensions
            .insert(request.id.clone(), snapshot.clone());
        drop(state);
        self.publish(
            "ui.extensions.registered",
            &request.id,
            revision,
            &snapshot,
            session_id,
        );
        Ok(snapshot)
    }

    pub fn list_extensions(&self) -> Vec<ExtensionSnapshot> {
        let mut extensions = self
            .state
            .lock()
            .expect("UI registry poisoned")
            .extensions
            .values()
            .cloned()
            .collect::<Vec<_>>();
        extensions.sort_by(|left, right| left.id.cmp(&right.id));
        extensions
    }

    pub fn register_contribution(
        &self,
        params: Value,
        session_id: &str,
    ) -> Result<ContributionSnapshot, ControlError> {
        let request: RegisterContribution = serde_json::from_value(params).map_err(|error| {
            ControlError::invalid_params(
                "ui.contributions.register",
                format!("invalid contribution: {error}"),
            )
        })?;
        if !matches!(
            request.location.as_str(),
            "left.sections"
                | "right.tabs"
                | "top_bar.actions"
                | "canvas.controls"
                | "status_bar"
                | "project.cards"
        ) {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "unsupported UI insertion location",
            ));
        }
        validate_tree(&request.root)?;
        let contribution_id = request.contribution_id.unwrap_or_else(|| {
            format!(
                "contribution:{}",
                crate::control::discovery::random_uuid_like()
                    .unwrap_or_else(|_| "unavailable".to_string())
            )
        });
        let mut state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(&request.extension_id)
            .ok_or_else(|| not_found("extension", &request.extension_id))?;
        ensure_owner(extension, session_id)?;
        ensure_contribution_capabilities(&request.root, extension)?;
        if state
            .contributions
            .iter()
            .any(|item| item.contribution_id == contribution_id)
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "contribution ID already exists",
            ));
        }
        let revision = self.events.next_revision();
        let snapshot = ContributionSnapshot {
            contribution_id: contribution_id.clone(),
            extension_id: request.extension_id.clone(),
            location: request.location,
            root: request.root,
            revision,
        };
        state.contributions.push(snapshot.clone());
        drop(state);
        self.publish(
            "ui.contributions.registered",
            &request.extension_id,
            revision,
            &snapshot,
            session_id,
        );
        Ok(snapshot)
    }

    pub fn list_contributions(&self) -> Vec<ContributionSnapshot> {
        self.state
            .lock()
            .expect("UI registry poisoned")
            .contributions
            .clone()
    }

    pub fn sync_native_bindings(&self, native_state: &Value, layers: &[super::LayerSnapshot]) {
        let mut state = self.state.lock().expect("UI registry poisoned");
        for contribution in &mut state.contributions {
            sync_component_binding(&mut contribution.root, native_state, layers);
        }
    }

    pub fn patch_values(
        &self,
        contribution_id: &str,
        values: &HashMap<String, Value>,
        if_revision: Option<u64>,
        session_id: &str,
    ) -> Result<ContributionSnapshot, ControlError> {
        let mut state = self.state.lock().expect("UI registry poisoned");
        let index = state
            .contributions
            .iter()
            .position(|item| item.contribution_id == contribution_id)
            .ok_or_else(|| not_found("contribution", contribution_id))?;
        let extension_id = state.contributions[index].extension_id.clone();
        ensure_owner(
            state
                .extensions
                .get(&extension_id)
                .ok_or_else(|| not_found("extension", &extension_id))?,
            session_id,
        )?;
        let contribution = &mut state.contributions[index];
        if if_revision.is_some_and(|expected| expected != contribution.revision) {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "UI contribution revision conflict",
            )
            .with_data(json!({"current_revision": contribution.revision})));
        }
        let known = component_ids(&contribution.root);
        if values.keys().any(|id| !known.contains(id)) {
            return Err(ControlError::invalid_params(
                "ui.contributions.patch_values",
                "value patch references an unknown component ID",
            ));
        }
        patch_component_values(&mut contribution.root, values);
        let revision = self.events.next_revision();
        contribution.revision = revision;
        let snapshot = contribution.clone();
        drop(state);
        self.publish(
            "ui.contributions.changed",
            &extension_id,
            revision,
            &snapshot,
            session_id,
        );
        Ok(snapshot)
    }

    pub fn remove_extension(
        &self,
        extension_id: &str,
        session_id: &str,
    ) -> Result<(), ControlError> {
        let mut state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(extension_id)
            .ok_or_else(|| not_found("extension", extension_id))?;
        ensure_owner(extension, session_id)?;
        state.extensions.remove(extension_id);
        state
            .contributions
            .retain(|item| item.extension_id != extension_id);
        let revision = self.events.next_revision();
        drop(state);
        self.events.publish(
            "ui.extensions.removed",
            extension_id,
            revision,
            json!({"extension_id": extension_id}),
            Some(session_id.to_string()),
            None,
        );
        Ok(())
    }

    pub fn remove_contribution(
        &self,
        contribution_id: &str,
        session_id: &str,
    ) -> Result<(), ControlError> {
        let mut state = self.state.lock().expect("UI registry poisoned");
        let index = state
            .contributions
            .iter()
            .position(|item| item.contribution_id == contribution_id)
            .ok_or_else(|| not_found("contribution", contribution_id))?;
        let extension_id = state.contributions[index].extension_id.clone();
        ensure_owner(
            state
                .extensions
                .get(&extension_id)
                .ok_or_else(|| not_found("extension", &extension_id))?,
            session_id,
        )?;
        state.contributions.remove(index);
        let revision = self.events.next_revision();
        drop(state);
        self.events.publish(
            "ui.contributions.removed",
            contribution_id,
            revision,
            json!({"contribution_id": contribution_id, "extension_id": extension_id}),
            Some(session_id.to_string()),
            None,
        );
        Ok(())
    }

    pub fn cleanup_session(&self, session_id: &str) {
        let mut state = self.state.lock().expect("UI registry poisoned");
        let owned = state
            .extensions
            .values()
            .filter(|extension| extension.owner_session_id == session_id)
            .map(|extension| (extension.id.clone(), extension.disconnect_policy.clone()))
            .collect::<Vec<_>>();
        if owned.is_empty() {
            return;
        }
        let revision = self.events.next_revision();
        let remove = owned
            .iter()
            .filter(|(_, policy)| matches!(policy, DisconnectPolicy::Remove))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let disconnected = owned
            .iter()
            .filter(|(_, policy)| !matches!(policy, DisconnectPolicy::Remove))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in &disconnected {
            if let Some(extension) = state.extensions.get_mut(id) {
                extension.connected = false;
                extension.revision = revision;
            }
        }
        state
            .contributions
            .retain(|item| !remove.contains(&item.extension_id));
        for id in &remove {
            state.extensions.remove(id);
        }
        drop(state);
        for id in remove {
            self.events.publish(
                "ui.extensions.removed",
                &id,
                revision,
                json!({"extension_id": id, "reason": "session_disconnected"}),
                Some(session_id.to_string()),
                None,
            );
        }
        for id in disconnected {
            self.events.publish(
                "ui.extensions.disconnected",
                &id,
                revision,
                json!({"extension_id": id}),
                Some(session_id.to_string()),
                None,
            );
        }
    }

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

    pub fn drain_actions(&self) -> Vec<UiAction> {
        std::mem::take(
            &mut *self
                .pending_actions
                .lock()
                .expect("UI action queue poisoned"),
        )
    }

    fn publish(
        &self,
        event: &str,
        source: &str,
        revision: u64,
        value: &impl Serialize,
        session_id: &str,
    ) {
        self.events.publish(
            event,
            source,
            revision,
            serde_json::to_value(value).unwrap_or_else(|_| json!({})),
            Some(session_id.to_string()),
            None,
        );
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

fn event_policy_kind(component: &Component) -> &str {
    component
        .event_policy
        .as_ref()
        .and_then(|policy| policy.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("immediate")
}

#[derive(Debug, Clone)]
struct Interaction {
    extension_id: String,
    owner_session_id: String,
    component_id: String,
    kind: String,
    value: Value,
    action: Option<Value>,
    event_policy: Option<Value>,
    occurred_at: Instant,
}

impl Interaction {
    fn key(&self) -> String {
        format!("{}:{}", self.extension_id, self.component_id)
    }

    fn ready(&self, now: Instant, last: Option<Instant>) -> bool {
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

fn render_component(
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

fn validate_tree(root: &Component) -> Result<(), ControlError> {
    let mut ids = HashSet::new();
    let mut count = 0;
    validate_component(root, 0, &mut count, &mut ids)
}

fn validate_component(
    component: &Component,
    depth: usize,
    count: &mut usize,
    ids: &mut HashSet<String>,
) -> Result<(), ControlError> {
    *count += 1;
    if depth > MAX_DEPTH || *count > MAX_COMPONENTS {
        return Err(ControlError::new(
            ControlErrorKind::ResourceLimit,
            "component tree exceeds safety limits",
        )
        .with_data(json!({
            "max_components": MAX_COMPONENTS,
            "max_depth": MAX_DEPTH,
        })));
    }
    if component.id.trim().is_empty() || !ids.insert(component.id.clone()) {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            "component IDs must be non-empty and unique",
        ));
    }
    if !matches!(
        component.kind.as_str(),
        "panel"
            | "column"
            | "row"
            | "grid"
            | "tabs"
            | "scroll"
            | "group"
            | "collapsible"
            | "text"
            | "markdown"
            | "status"
            | "warning"
            | "error"
            | "spinner"
            | "separator"
            | "spacer"
            | "button"
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
            | "progress"
    ) {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            format!("unsupported component type '{}'", component.kind),
        ));
    }
    if component.kind == "grid"
        && component
            .columns
            .is_some_and(|value| !(1..=16).contains(&value))
    {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            "grid columns must be between 1 and 16",
        ));
    }
    if matches!(component.kind.as_str(), "slider" | "number" | "integer")
        && component
            .minimum
            .zip(component.maximum)
            .is_some_and(|(min, max)| !min.is_finite() || !max.is_finite() || min >= max)
    {
        return Err(ControlError::invalid_params(
            "ui.contributions.register",
            "numeric component bounds must be finite and increasing",
        ));
    }
    if let Some(policy) = component.event_policy.as_ref() {
        let kind = policy.get("type").and_then(Value::as_str).ok_or_else(|| {
            ControlError::invalid_params(
                "ui.contributions.register",
                "event_policy.type is required",
            )
        })?;
        if !matches!(kind, "commit" | "immediate" | "throttle" | "debounce") {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "unsupported event policy",
            ));
        }
        if matches!(kind, "throttle" | "debounce")
            && !policy
                .get("milliseconds")
                .and_then(Value::as_u64)
                .is_some_and(|value| (1..=60_000).contains(&value))
        {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "throttle/debounce milliseconds must be between 1 and 60000",
            ));
        }
    }
    for child in &component.children {
        validate_component(child, depth + 1, count, ids)?;
    }
    Ok(())
}

fn component_ids(root: &Component) -> HashSet<String> {
    let mut ids = HashSet::new();
    fn collect(component: &Component, ids: &mut HashSet<String>) {
        ids.insert(component.id.clone());
        for child in &component.children {
            collect(child, ids);
        }
    }
    collect(root, &mut ids);
    ids
}

fn patch_component_values(component: &mut Component, values: &HashMap<String, Value>) {
    if let Some(value) = values.get(&component.id) {
        component.value = value.clone();
    }
    for child in &mut component.children {
        patch_component_values(child, values);
    }
}

fn sync_component_binding(
    component: &mut Component,
    native_state: &Value,
    layers: &[super::LayerSnapshot],
) {
    let binding = component
        .action
        .as_ref()
        .filter(|action| action.get("type").and_then(Value::as_str) == Some("bind"));
    if let Some(binding) = binding {
        let target = binding.get("target").and_then(Value::as_str);
        let property = binding.get("property").and_then(Value::as_str);
        let value = match (target, property) {
            (Some("viewer.layers"), Some(property)) => binding
                .get("layer_id")
                .and_then(Value::as_str)
                .and_then(|id| layers.iter().find(|layer| layer.layer_id == id))
                .and_then(|layer| match property {
                    "visible" => Some(json!(layer.visible)),
                    "opacity" => Some(json!(layer.opacity)),
                    _ => None,
                }),
            (Some("viewer.channels"), Some("active")) => native_state
                .pointer("/channels/channels")
                .and_then(Value::as_array)
                .and_then(|channels| {
                    channels.iter().find(|channel| {
                        channel.get("selected").and_then(Value::as_bool) == Some(true)
                    })
                })
                .and_then(|channel| channel.get("name").cloned()),
            (Some("viewer.channels"), Some("visible")) => native_state
                .pointer("/channels/channels")
                .and_then(Value::as_array)
                .map(|channels| {
                    Value::Array(
                        channels
                            .iter()
                            .filter(|channel| {
                                channel.get("visible").and_then(Value::as_bool) == Some(true)
                            })
                            .filter_map(|channel| channel.get("name").cloned())
                            .collect(),
                    )
                }),
            (Some("viewer.camera"), Some("zoom")) => native_state
                .pointer("/camera/camera/zoom_screen_per_lvl0_px")
                .cloned(),
            (Some("viewer"), Some("smooth_pixels")) => native_state
                .pointer("/smooth/smooth_pixels/smooth")
                .cloned(),
            _ => None,
        };
        if let Some(value) = value {
            component.value = value;
        }
    }
    for child in &mut component.children {
        sync_component_binding(child, native_state, layers);
    }
}

fn ensure_contribution_capabilities(
    root: &Component,
    extension: &ExtensionSnapshot,
) -> Result<(), ControlError> {
    let granted = &extension.granted_capabilities;
    if !granted.iter().any(|capability| capability == "ui.panels") {
        return Err(ControlError::new(
            ControlErrorKind::PermissionDenied,
            "the extension was not granted ui.panels",
        ));
    }
    fn check(component: &Component, granted: &[String]) -> Result<(), ControlError> {
        if let Some(action) = component.action.as_ref() {
            let kind = action.get("type").and_then(Value::as_str).ok_or_else(|| {
                ControlError::invalid_params(
                    "ui.contributions.register",
                    "component action.type is required",
                )
            })?;
            match kind {
                "emit" => {
                    if !action
                        .get("event")
                        .and_then(Value::as_str)
                        .is_some_and(|event| !event.trim().is_empty())
                    {
                        return Err(ControlError::invalid_params(
                            "ui.contributions.register",
                            "emit actions require a non-empty event",
                        ));
                    }
                }
                "command" => {
                    let method = action
                        .get("method")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            ControlError::invalid_params(
                                "ui.contributions.register",
                                "command actions require a method",
                            )
                        })?;
                    if crate::control::registry::method(method).is_none() {
                        return Err(ControlError::invalid_params(
                            "ui.contributions.register",
                            format!("unknown native command '{method}'"),
                        ));
                    }
                    if !granted
                        .iter()
                        .any(|capability| capability == "viewer.write")
                    {
                        return Err(ControlError::new(
                            ControlErrorKind::PermissionDenied,
                            "native command actions require viewer.write",
                        ));
                    }
                }
                "bind" => {
                    let target = action.get("target").and_then(Value::as_str);
                    if !matches!(
                        target,
                        Some("viewer.layers" | "viewer.channels" | "viewer.camera" | "viewer")
                    ) {
                        return Err(ControlError::new(
                            ControlErrorKind::Unsupported,
                            "unsupported native binding target",
                        ));
                    }
                    let property = action.get("property").and_then(Value::as_str);
                    let property_supported = match target {
                        Some("viewer.layers") => {
                            matches!(property, Some("opacity" | "visible"))
                                && action
                                    .get("layer_id")
                                    .and_then(Value::as_str)
                                    .is_some_and(|id| !id.trim().is_empty())
                        }
                        Some("viewer.channels") => matches!(property, Some("active" | "visible")),
                        Some("viewer.camera") => property == Some("zoom"),
                        Some("viewer") => property == Some("smooth_pixels"),
                        _ => false,
                    };
                    if !property_supported {
                        return Err(ControlError::new(
                            ControlErrorKind::Unsupported,
                            "unsupported native binding property or missing layer_id",
                        ));
                    }
                    let capability = if target == Some("viewer.layers") {
                        "viewer.layers.write"
                    } else {
                        "viewer.write"
                    };
                    if !granted.iter().any(|granted| granted == capability) {
                        return Err(ControlError::new(
                            ControlErrorKind::PermissionDenied,
                            format!("native bindings to this target require {capability}"),
                        ));
                    }
                }
                _ => {
                    return Err(ControlError::invalid_params(
                        "ui.contributions.register",
                        "unsupported component action type",
                    ));
                }
            }
        }
        for child in &component.children {
            check(child, granted)?;
        }
        Ok(())
    }
    check(root, granted)
}

fn ensure_owner(extension: &ExtensionSnapshot, session_id: &str) -> Result<(), ControlError> {
    if extension.owner_session_id != session_id {
        Err(ControlError::new(
            ControlErrorKind::PermissionDenied,
            "extension is owned by another control session",
        ))
    } else {
        Ok(())
    }
}

fn not_found(kind: &str, id: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::ResourceNotFound,
        format!("{kind} '{id}' was not found"),
    )
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Ownership;
    use std::collections::BTreeMap;

    #[test]
    fn extension_trees_validate_and_patch_atomically() {
        let registry = UiRegistry::shared(EventHub::shared());
        registry
            .register_extension(
                json!({
                    "id": "org.example.test", "name": "Test", "version": "1.0",
                    "capabilities": ["ui.panels", "unknown"]
                }),
                "session",
            )
            .expect("extension");
        let contribution = registry
            .register_contribution(
                json!({
                    "extension_id": "org.example.test", "location": "right.tabs",
                                "root": {"id": "root", "type": "panel", "children": [
                                    {"id": "threshold", "type": "slider", "value": 0.5,
                                     "minimum": 0.0, "maximum": 1.0,
                                     "event_policy": {"type": "debounce", "milliseconds": 100}}
                    ]}
                }),
                "session",
            )
            .expect("contribution");
        let patched = registry
            .patch_values(
                &contribution.contribution_id,
                &HashMap::from([("threshold".to_string(), json!(0.8))]),
                Some(contribution.revision),
                "session",
            )
            .expect("patch");
        assert_eq!(patched.root.children[0].value, 0.8);
        assert!(
            registry
                .patch_values(
                    &patched.contribution_id,
                    &HashMap::from([("missing".into(), json!(1))]),
                    None,
                    "session"
                )
                .is_err()
        );
    }

    #[test]
    fn native_bindings_reflect_viewer_and_layer_state() {
        let mut root: Component = serde_json::from_value(json!({
            "id": "root", "type": "column", "children": [
                {"id": "opacity", "type": "slider", "minimum": 0.0, "maximum": 1.0,
                 "action": {"type": "bind", "target": "viewer.layers",
                            "layer_id": "layer:test", "property": "opacity"}},
                {"id": "channel", "type": "select", "options": ["DAPI", "CD3"],
                 "action": {"type": "bind", "target": "viewer.channels", "property": "active"}}
            ]
        }))
        .expect("component");
        let layers = vec![super::super::LayerSnapshot {
            layer_id: "layer:test".into(),
            name: "Test".into(),
            kind: "labels".into(),
            data_resource_id: "resource:test".into(),
            visible: true,
            opacity: 0.4,
            ownership: Ownership::Session,
            owner_session_id: "session".into(),
            style: BTreeMap::new(),
            provenance: BTreeMap::new(),
            order: 0,
            revision: 1,
        }];
        sync_component_binding(
            &mut root,
            &json!({"channels": {"channels": [
                {"name": "DAPI", "selected": false},
                {"name": "CD3", "selected": true}
            ]}}),
            &layers,
        );
        assert_eq!(root.children[0].value, 0.4);
        assert_eq!(root.children[1].value, "CD3");
    }

    #[test]
    fn disconnected_extensions_can_be_reclaimed_by_a_new_session() {
        let registry = UiRegistry::shared(EventHub::shared());
        registry
            .register_extension(
                json!({
                    "id": "org.example.reconnect", "name": "Reconnect", "version": "1",
                    "capabilities": ["ui.panels"], "disconnect_policy": "disable"
                }),
                "first",
            )
            .expect("first registration");
        registry.cleanup_session("first");
        let extension = registry
            .register_extension(
                json!({
                    "id": "org.example.reconnect", "name": "Reconnect", "version": "1",
                    "capabilities": ["ui.panels"], "disconnect_policy": "disable"
                }),
                "second",
            )
            .expect("replacement registration");
        assert!(extension.connected);
        assert_eq!(extension.owner_session_id, "second");
    }
}
