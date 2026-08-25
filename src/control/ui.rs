use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;
use serde_json::{Value, json};

use super::{ControlError, ControlErrorKind, EventHub};

mod commands;
mod layout_templates;
mod render;
mod shell_catalog;
mod types;
mod validation;

use render::Interaction;
pub use types::*;
use validation::{
    component_ids, ensure_contribution_capabilities, patch_component_values,
    sync_component_binding, validate_tree,
};

#[derive(Debug, Default)]
struct State {
    extensions: HashMap<String, ExtensionSnapshot>,
    contributions: Vec<ContributionSnapshot>,
    extension_layouts: Vec<ExtensionLayoutSnapshot>,
    session_capabilities: HashMap<String, BTreeSet<String>>,
}

#[derive(Debug)]
pub struct UiRegistry {
    state: Mutex<State>,
    events: Arc<EventHub>,
    pending_actions: Mutex<Vec<UiAction>>,
    deferred_interactions: Mutex<HashMap<String, Interaction>>,
    last_emitted: Mutex<HashMap<String, Instant>>,
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

    pub fn set_session_capabilities(&self, session_id: &str, capabilities: &[String]) {
        self.state
            .lock()
            .expect("UI registry poisoned")
            .session_capabilities
            .insert(
                session_id.to_string(),
                capabilities.iter().cloned().collect(),
            );
    }

    pub fn session_capabilities(&self, session_id: &str) -> Vec<String> {
        self.state
            .lock()
            .expect("UI registry poisoned")
            .session_capabilities
            .get(session_id)
            .into_iter()
            .flatten()
            .cloned()
            .collect()
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
        if request.readiness_reason.as_ref().is_some_and(|reason| {
            reason.is_empty() || reason.len() > 256 || reason.chars().any(char::is_control)
        }) {
            return Err(ControlError::invalid_params(
                "ui.extensions.register",
                "readiness_reason must contain 1 to 256 non-control bytes",
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
            ready: request.ready,
            readiness_reason: request.readiness_reason,
            revision,
        };
        state
            .extensions
            .insert(request.id.clone(), snapshot.clone());
        for contribution in state
            .contributions
            .iter_mut()
            .filter(|contribution| contribution.extension_id == request.id)
        {
            contribution.ownership = extension_ownership(&request.id, session_id);
            contribution.readiness =
                extension_content_readiness(&contribution.extension_version, &snapshot).to_string();
        }
        for layout in state
            .extension_layouts
            .iter_mut()
            .filter(|layout| layout.extension_id == request.id)
        {
            layout.ownership = extension_ownership(&request.id, session_id);
            layout.readiness =
                extension_content_readiness(&layout.extension_version, &snapshot).to_string();
        }
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

    pub fn set_extension_readiness(
        &self,
        params: Value,
        session_id: &str,
    ) -> Result<ExtensionSnapshot, ControlError> {
        let request: SetExtensionReadiness = serde_json::from_value(params).map_err(|error| {
            ControlError::invalid_params(
                "ui.extensions.set_readiness",
                format!("invalid readiness update: {error}"),
            )
        })?;
        if request.reason.as_ref().is_some_and(|reason| {
            reason.is_empty() || reason.len() > 256 || reason.chars().any(char::is_control)
        }) {
            return Err(ControlError::invalid_params(
                "ui.extensions.set_readiness",
                "reason must contain 1 to 256 non-control bytes",
            ));
        }
        let mut state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get_mut(&request.extension_id)
            .ok_or_else(|| not_found("extension", &request.extension_id))?;
        ensure_owner(extension, session_id)?;
        let changed =
            extension.ready != request.ready || extension.readiness_reason != request.reason;
        if !changed {
            return Ok(extension.clone());
        }
        let revision = self.events.next_revision();
        extension.ready = request.ready;
        extension.readiness_reason = request.reason;
        extension.revision = revision;
        let snapshot = extension.clone();
        for contribution in state
            .contributions
            .iter_mut()
            .filter(|item| item.extension_id == request.extension_id)
        {
            contribution.readiness =
                extension_content_readiness(&contribution.extension_version, &snapshot).to_string();
            contribution.revision = revision;
        }
        for layout in state
            .extension_layouts
            .iter_mut()
            .filter(|item| item.extension_id == request.extension_id)
        {
            layout.readiness =
                extension_content_readiness(&layout.extension_version, &snapshot).to_string();
            layout.revision = revision;
        }
        drop(state);
        self.publish(
            "ui.extensions.readiness_changed",
            &request.extension_id,
            revision,
            &snapshot,
            session_id,
        );
        Ok(snapshot)
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
            "shell"
                | "left.sections"
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
        if contribution_id.trim().is_empty()
            || contribution_id.len() > 128
            || contribution_id.chars().any(char::is_whitespace)
        {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "contribution_id must be 1–128 characters without whitespace",
            ));
        }
        let shell_mount = contribution_shell_mount(&request.extension_id, &contribution_id);
        if shell_mount.len() > 256 {
            return Err(ControlError::invalid_params(
                "ui.contributions.register",
                "extension and contribution IDs are too long for a shell mount ID",
            ));
        }
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
            extension_version: extension.version.clone(),
            shell_mount,
            location: request.location,
            root: request.root,
            ownership: extension_ownership(&request.extension_id, session_id),
            readiness: extension_content_readiness(&extension.version, extension).to_string(),
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
        state
            .extension_layouts
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

    pub fn cleanup_session(&self, session_id: &str) -> UiSessionCleanup {
        let mut state = self.state.lock().expect("UI registry poisoned");
        state.session_capabilities.remove(session_id);
        let owned = state
            .extensions
            .values()
            .filter(|extension| extension.owner_session_id == session_id)
            .map(|extension| (extension.id.clone(), extension.disconnect_policy.clone()))
            .collect::<Vec<_>>();
        if owned.is_empty() {
            return UiSessionCleanup::default();
        }
        let unavailable_mounts = state
            .contributions
            .iter()
            .filter(|contribution| owned.iter().any(|(id, _)| id == &contribution.extension_id))
            .map(|contribution| contribution.shell_mount.clone())
            .collect::<Vec<_>>();
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
            for contribution in state
                .contributions
                .iter_mut()
                .filter(|item| item.extension_id == *id)
            {
                contribution.readiness = "disconnected".to_string();
                contribution.revision = revision;
            }
            for layout in state
                .extension_layouts
                .iter_mut()
                .filter(|item| item.extension_id == *id)
            {
                layout.readiness = "disconnected".to_string();
                layout.revision = revision;
            }
        }
        state
            .contributions
            .retain(|item| !remove.contains(&item.extension_id));
        state
            .extension_layouts
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
        UiSessionCleanup {
            unavailable_mounts,
            extensions: owned
                .into_iter()
                .map(|(extension_id, disconnect_policy)| UiExtensionCleanup {
                    extension_id,
                    disconnect_policy,
                })
                .collect(),
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

fn contribution_shell_mount(extension_id: &str, contribution_id: &str) -> String {
    format!("extension:{extension_id}/{contribution_id}")
}

fn extension_ownership(extension_id: &str, session_id: &str) -> Value {
    json!({
        "scope":"extension",
        "owner_id":extension_id,
        "owner_session_id":session_id,
        "protected":false,
    })
}

fn extension_content_readiness<'a>(
    retained_version: &str,
    extension: &'a ExtensionSnapshot,
) -> &'a str {
    if !extension.connected {
        "disconnected"
    } else if retained_version != extension.version {
        "incompatible"
    } else if !extension.ready {
        "not_ready"
    } else {
        "ready"
    }
}

pub(super) fn contribution_modes(location: &str) -> &'static [&'static str] {
    match location {
        "project.cards" => &["project"],
        "left.sections" | "right.tabs" | "canvas.controls" => &["single", "mosaic"],
        _ => &["project", "single", "mosaic"],
    }
}

pub(super) fn contribution_kind(location: &str) -> &'static str {
    match location {
        "top_bar.actions" | "status_bar" | "canvas.controls" => "toolbar",
        _ => "panel",
    }
}

pub(super) fn contribution_legal_parent_types(location: &str) -> &'static [&'static str] {
    match location {
        "top_bar.actions" => &["toolbar", "row", "column", "panel"],
        "status_bar" => &["status_bar", "row", "column", "panel"],
        "canvas.controls" => &["toolbar", "row", "column", "panel"],
        _ => &["tabs", "panel", "collapsible", "row", "column", "split"],
    }
}

#[cfg(test)]
mod tests;
