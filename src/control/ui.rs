use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{ControlError, ControlErrorKind, EventHub};

mod render;
mod validation;

use render::Interaction;
use validation::{
    component_ids, ensure_contribution_capabilities, patch_component_values,
    sync_component_binding, validate_tree,
};

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
mod tests;
